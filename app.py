import json
import os

import gradio as gr
import requests
from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool, set_tracing_disabled
from pypdf import PdfReader


KIMI_MODEL = "kimi-k2.6"
KIMI_BASE_URL = "https://api.moonshot.ai/v1"
OLOSTEP_SEARCH_URL = "https://api.olostep.com/v1/searches"
OLOSTEP_SCRAPE_URL = "https://api.olostep.com/v1/scrapes"
MAX_AGENT_TURNS = 25
MAX_SEARCH_RESULTS = 10
MAX_PAGE_READS = 3
MAX_SCRAPED_CHARS = 8000

DEFAULT_PREFERENCES = """Remote data science, AI writer, or technical writer roles in AI, machine learning, data science, or cloud.
Prefer roles focused on technical content, tutorials, developer education, research writing, and AI product storytelling."""

AGENT_INSTRUCTIONS = """
You are JobFit AI, a focused job-search agent.

Tool plan:
- Call search_jobs exactly once with limit 10.
- Read at most 3 direct job pages with read_job_page.
- After reading up to 3 pages, stop using tools and write the report.
- Search again only if the first search returns zero usable jobs.
- Avoid broad search pages, expired jobs, and LinkedIn unless no better source exists.

Report rules:
- Keep the report simple, clear, and practical.
- Use short bullets.
- Do not use em dashes.
- Do not use contractions.
- Do not add text before or after the report.
- Include at least 5 jobs in Ranked Jobs and Job Notes if search results contain at least 5 usable jobs.
- If only 3 pages were scraped, fill the remaining entries from search results using title, URL, and description.
- Add rejected jobs at the end with name, link, and rejection reason.
- Every job must include a clickable Markdown link.
- Every job must have one apply decision: Apply, Maybe, or Do not apply.

Use exactly this Markdown structure:

# JobFit AI Report

## Best Match

- **Role:** <job title>
- **Company:** <company>
- **Apply decision:** Apply / Maybe / Do not apply
- **Fit score:** <score>/100
- **Link:** [Apply here](<job url>)

**Why this is the best match:**

- <specific reason>
- <specific reason>
- <specific reason>

## Ranked Jobs

| Rank | Role | Company | Apply? | Fit | Link |
| --- | --- | --- | --- | --- | --- |
| 1 | <role> | <company> | Apply / Maybe / Do not apply | <score>/100 | [Apply here](<url>) |

## Job Notes

### 1. <Role> at <Company>

- **Apply decision:** Apply / Maybe / Do not apply
- **Fit score:** <score>/100
- **Link:** [Apply here](<job url>)

**Why it fits:**

- <bullet>
- <bullet>

**Concerns:**

- <bullet>
- <bullet>

**Application angle:**

- <how the person should position their CV/application>

## Rejected Jobs

| Job | Link | Reason |
| --- | --- | --- |
| <job title or source title> | [Open](<url>) | <short reason it was rejected> |
""".strip()

RUN_PROMPT_TEMPLATE = """
Find current job postings for this candidate and rank them by fit.

Keep the run simple:
- one search
- up to three page reads
- final report

The final report must follow AGENT_INSTRUCTIONS exactly.
Use simple wording. Do not use em dashes. Do not use contractions.
Rank at least 5 jobs when search returns at least 5 usable results.
Use search-result backups when only 3 pages are read.
Include rejected jobs with name, link, and reason.

Candidate CV:
{cv_text}

Preferences:
{preferences}
""".strip()

set_tracing_disabled(True)


def read_cv(path: str, max_chars: int = 12000) -> tuple[str, list[str]]:
    reader = PdfReader(path)
    text = ""
    logs = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        logs.append(f"Read CV page {page_number}: {len(page_text)} characters")
        text += page_text + "\n"
    return text[:max_chars], logs


@function_tool
def search_jobs(query: str, limit: int = MAX_SEARCH_RESULTS) -> str:
    """Search the web for job listings and return compact JSON results."""
    safe_limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))
    response = requests.post(
        OLOSTEP_SEARCH_URL,
        headers={"Authorization": f"Bearer {os.environ['OLOSTEP_API_KEY']}", "Content-Type": "application/json"},
        json={"query": query},
        timeout=60,
    )
    response.raise_for_status()
    links = response.json().get("result", {}).get("links", [])[:safe_limit]
    results = [
        {"title": item.get("title", "Untitled"), "url": item.get("url"), "description": item.get("description", "")}
        for item in links
        if isinstance(item, dict) and item.get("url")
    ]
    return json.dumps(results, ensure_ascii=False)


@function_tool
def read_job_page(url: str) -> str:
    """Scrape one job listing URL and return markdown text."""
    response = requests.post(
        OLOSTEP_SCRAPE_URL,
        headers={"Authorization": f"Bearer {os.environ['OLOSTEP_API_KEY']}", "Content-Type": "application/json"},
        json={"url_to_scrape": url, "formats": ["markdown"]},
        timeout=120,
    )
    response.raise_for_status()
    markdown = response.json().get("result", {}).get("markdown_content") or ""
    return markdown[:MAX_SCRAPED_CHARS]


def build_agent() -> Agent:
    kimi_client = AsyncOpenAI(api_key=os.environ["MOONSHOT_API_KEY"], base_url=KIMI_BASE_URL)
    kimi_model = OpenAIChatCompletionsModel(model=KIMI_MODEL, openai_client=kimi_client)
    return Agent(
        name="JobFit AI",
        model=kimi_model,
        model_settings=ModelSettings(
            tool_choice="auto",
            parallel_tool_calls=True,
            extra_body={"thinking": {"type": "disabled"}},
        ),
        tools=[search_jobs, read_job_page],
        instructions=AGENT_INSTRUCTIONS,
    )


async def run_jobfit(cv_file: str, preferences: str):
    logs = []

    def emit(message: str, report: str = ""):
        logs.append(message)
        return "\n".join(logs), report

    if not cv_file:
        yield emit("Upload a CV PDF first.")
        return
    if not preferences.strip():
        yield emit("Enter job preferences first.")
        return

    try:
        yield emit("Reading CV...")
        cv_text, cv_logs = read_cv(cv_file)
        for log in cv_logs:
            yield emit(log)

        prompt = RUN_PROMPT_TEMPLATE.format(cv_text=cv_text, preferences=preferences)
        result = Runner.run_streamed(
            build_agent(),
            prompt,
            max_turns=MAX_AGENT_TURNS,
            run_config=RunConfig(workflow_name="JobFit AI Kimi Search", tracing_disabled=True),
        )

        yield emit("Starting agent run...")
        async for event in result.stream_events():
            if event.type != "run_item_stream_event":
                continue
            item = event.item
            if event.name == "tool_called":
                raw = item.raw_item
                tool_name = raw.get("name") if isinstance(raw, dict) else getattr(raw, "name", "tool")
                arguments = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", "")
                yield emit(f"Tool call: {tool_name}")
                if arguments:
                    yield emit(f"Parameters: {str(arguments).replace(chr(10), ' ')[:500]}")
            elif event.name == "tool_output":
                yield emit(f"Tool output: {len(str(item.output)):,} chars")

        report = result.final_output
        yield emit("Done!", report)
    except Exception as exc:
        yield emit(f"Error: {exc}")


THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="sky",
    neutral_hue="slate",
)


with gr.Blocks(title="JobFit AI") as demo:
    gr.Markdown(
        """
# JobFit AI

Upload a CV, describe the job preferences, and generate a ranked list of direct job listings to apply for.
"""
    )

    with gr.Row():
        cv_input = gr.File(label="CV PDF", file_types=[".pdf"], type="filepath")
        preferences_input = gr.Textbox(
            label="Job preferences",
            value=DEFAULT_PREFERENCES,
            lines=8,
            placeholder="Describe role type, industry, remote/location preferences, seniority, and topics.",
        )
    run_button = gr.Button("Generate JobFit Report", variant="primary", elem_id="run-button")
    logs_output = gr.Textbox(
        label="Progress log",
        lines=8,
        max_lines=8,
        interactive=False,
        elem_id="progress-log",
        visible=False,
    )

    gr.Markdown("## Final Report")
    report_output = gr.Markdown(label="Report")

    def start_run():
        return gr.update(interactive=False), gr.update(visible=True)

    def end_run():
        return gr.update(interactive=True)

    run_event = run_button.click(
        fn=start_run,
        outputs=[run_button, logs_output],
        show_progress="hidden",
    )
    run_event.then(
        fn=run_jobfit,
        inputs=[cv_input, preferences_input],
        outputs=[logs_output, report_output],
        show_progress="hidden",
    ).then(
        fn=end_run,
        outputs=run_button,
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue().launch(theme=THEME)
