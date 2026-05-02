import json
import os
import re

import gradio as gr
import requests
from openai import OpenAI
from pypdf import PdfReader


KIMI_MODEL = "kimi-k2.6"
KIMI_BASE_URL = "https://api.moonshot.ai/v1"
OLOSTEP_SEARCH_URL = "https://api.olostep.com/v1/searches"
OLOSTEP_SCRAPE_URL = "https://api.olostep.com/v1/scrapes"

DEFAULT_PREFERENCES = """Remote data science, AI writer, or technical writer roles in AI, machine learning, data science, or cloud.
Prefer roles focused on technical content, tutorials, developer education, research writing, and AI product storytelling."""

MAX_SEARCH_RESULTS = 20
MAX_SOURCES_TO_SCRAPE = 10
MAX_SCRAPED_CHARS = 10000
MIN_REPORT_ENTRIES = 5

# Domains that block scrapers — skip them during scraping
UNSCRAPEABLE_DOMAINS = [
    "linkedin.com",
]

# Only filter out broad search/category pages, not specific job listings
SEARCH_PAGE_HINTS = [
    "/search/",
    "/jobs/search",
    "/content/search",
    "/remote/content/search",
    "?q=",
    "?query=",
]

CATEGORY_PATHS = ("/jobs", "/careers", "/openings", "/positions")

DIRECT_JOB_HINTS = [
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "workable.com",
    "smartrecruiters.com",
    "bamboohr.com",
    "myworkdayjobs.com",
    "/careers/",
    "/jobs/",
    "/job/",
    "/positions/",
    "/openings/",
]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Set the {name} environment variable.")
    return value


def kimi_client() -> OpenAI:
    return OpenAI(api_key=require_env("MOONSHOT_API_KEY"), base_url=KIMI_BASE_URL)


def ask_kimi(prompt: str, system_prompt: str = "You are JobFit AI, a precise job-search and job-fit assistant.") -> str:
    response = kimi_client().chat.completions.create(
        model=KIMI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content.strip()


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def read_cv(path: str, max_chars: int = 12000) -> tuple[str, list[str]]:
    reader = PdfReader(path)
    text = ""
    logs = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        logs.append(f"Read CV page {page_number}: {len(page_text)} characters")
        text += page_text + "\n"
    return text[:max_chars], logs


def olostep_headers() -> dict:
    return {
        "Authorization": f"Bearer {require_env('OLOSTEP_API_KEY')}",
        "Content-Type": "application/json",
    }


def is_specific_job_listing(url: str) -> bool:
    """Check if URL points to a specific job listing (not a broad category/search page)."""
    normalized = url.lower().rstrip("/")

    # Filter out search pages
    if any(hint in normalized for hint in SEARCH_PAGE_HINTS):
        return False

    # Filter out bare category pages (e.g. /jobs, /careers with nothing after)
    if any(normalized.endswith(cat) for cat in CATEGORY_PATHS):
        return False

    # Must match at least one job-related URL pattern
    return any(hint in normalized for hint in DIRECT_JOB_HINTS)


def search_jobs(query: str, limit: int = MAX_SEARCH_RESULTS) -> list[dict]:
    response = requests.post(
        OLOSTEP_SEARCH_URL,
        headers=olostep_headers(),
        json={"query": query},
        timeout=60,
    )
    response.raise_for_status()

    links = response.json().get("result", {}).get("links", [])
    results = []
    for index, link in enumerate(links[:limit], start=1):
        if isinstance(link, dict) and link.get("url"):
            results.append({
                "index": index,
                "title": link.get("title", "Untitled"),
                "url": link["url"],
                "description": link.get("description", ""),
            })
    return results


def filter_job_listings(sources: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into specific job listings vs. broad/non-job pages."""
    job_listings = []
    other = []
    for source in sources:
        if is_specific_job_listing(source["url"]):
            job_listings.append(source)
        else:
            other.append(source)

    for index, source in enumerate(job_listings, start=1):
        source["index"] = index
    return job_listings, other


def deduplicate_sources(*result_sets: list[dict]) -> list[dict]:
    merged = []
    seen_urls = set()
    for results in result_sets:
        for result in results:
            normalized_url = result["url"].rstrip("/").lower()
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                merged.append(result)
    for index, result in enumerate(merged, start=1):
        result["index"] = index
    return merged


def scrape_url(source: dict) -> dict:
    response = requests.post(
        OLOSTEP_SCRAPE_URL,
        headers=olostep_headers(),
        json={"url_to_scrape": source["url"], "formats": ["markdown"]},
        timeout=120,
    )
    response.raise_for_status()
    markdown = response.json().get("result", {}).get("markdown_content") or ""
    return {**source, "scraped_chars": len(markdown), "markdown": markdown[:MAX_SCRAPED_CHARS]}


def select_sources_with_kimi(sources: list[dict], preferences: str) -> tuple[list[dict], list[dict]]:
    prompt = f"""
Select at least {MIN_REPORT_ENTRIES} and up to {MAX_SOURCES_TO_SCRAPE} direct job listing pages to scrape.

Candidate preferences:
{preferences}

Sources:
{json.dumps(sources, indent=2)}

Choose only real, specific job posting pages a candidate can apply to directly.
Reject broad job boards, search pages, category pages, spam, and unrelated pages.

Return JSON:
{{
  "selected": [{{"index": 1, "reason": "short reason"}}],
  "skipped": [{{"index": 2, "reason": "short reason"}}]
}}
"""
    selection = parse_json_response(ask_kimi(prompt))
    selected = []
    for item in selection.get("selected", []):
        match = next((s for s in sources if s["index"] == item.get("index")), None)
        if match:
            selected.append({**match, "selection_reason": item.get("reason", "Selected by Kimi")})
    return selected[:MAX_SOURCES_TO_SCRAPE], selection.get("skipped", [])


def generate_search_query(cv_text: str, preferences: str) -> str:
    prompt = f"""
Create one short web search query to find relevant job postings.

Candidate CV excerpt:
{cv_text[:5000]}

Job preferences:
{preferences}

Rules: 6-10 words. Focus on role title + 2-3 must-have skills.
No site: operators, OR operators, parentheses, quotes, company names, or ATS names.
No filler like "hiring immediately", "apply now", "full time", or years.

Return only the search query.
"""
    query = ask_kimi(prompt).strip().strip('"').strip("'")
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"\b(AND|OR)\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\bsite:\S+", " ", query)
    query = re.sub(r"[()\"']", " ", query)
    query = re.sub(r"\b(2024|2025|2026|apply now|hiring immediately|full time)\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip()

    words = query.split()
    if len(words) > 12:
        query = " ".join(words[:12])
    if len(query) > 100:
        query = query[:100].rsplit(" ", 1)[0]
    if not query:
        query = "remote AI technical writer careers"
    return query


def generate_report(cv_text: str, preferences: str, scraped_sources: list[dict], skipped_sources: list[dict], additional_listings=None) -> str:
    analysis_input = [
        {"title": s["title"], "url": s["url"], "scraped_chars": s["scraped_chars"], "markdown": s["markdown"]}
        for s in scraped_sources
    ]

    additional_section = ""
    if additional_listings:
        additional_brief = [
            {"title": s["title"], "url": s["url"], "description": s.get("description", "")}
            for s in additional_listings
        ]
        additional_section = (
            f"\nAdditional job listings (not scraped in full, include in ranking"
            f" to reach at least {MIN_REPORT_ENTRIES} entries):\n"
            f"{json.dumps(additional_brief, ensure_ascii=False, indent=2)}\n"
        )

    prompt = f"""
Rank these job listings for the candidate.

Candidate CV:
{cv_text}

Job preferences:
{preferences}

Scraped sources:
{json.dumps(analysis_input, ensure_ascii=False, indent=2)}

Skipped sources:
{json.dumps(skipped_sources, ensure_ascii=False, indent=2)}
{additional_section}
IMPORTANT: You MUST include at least {MIN_REPORT_ENTRIES} job listings in the Ranked Summary table and Details section.
Include ALL scraped sources. If a source lacks full details, still include it with whatever info is available.
For additional listings that were not fully scraped, include them with basic info (title, URL) and estimate the fit score.

Write a clean Markdown report with this structure:

# JobFit AI Results

## Best Job To Apply For

**Role:** <job title>
**Company:** <company>
**Fit Score:** <score>/100
**URL:** [Apply](<url>)

**Why this is the best fit:**
- <reason 1>
- <reason 2>

## Ranked Summary

| Rank | Role | Company | Fit | Action |
| --- | --- | --- | --- | --- |
| 1 | <role> | <company> | <score>/100 | Apply / Maybe / Skip |

## Details

### 1. <Job title>

**Company:** <company>
**Fit Score:** <score>/100
**URL:** [Apply](<url>)

**Why it matches:**
- <bullet>

**Concerns:**
- <bullet>

**Application angle:**
- <bullet>

## Skipped Sources

- **<title>:** <why skipped>
"""
    return normalize_report_markdown(ask_kimi(prompt))


def normalize_report_markdown(markdown_text: str) -> str:
    text = markdown_text.replace("\r\n", "\n")
    labels = ["Role", "Company", "Fit Score", "Recommendation", "URL"]
    for label in labels:
        text = re.sub(rf"(?<!\n)\s+(\*\*{re.escape(label)}:\*\*)", r"\n\1", text)

    normalized_lines = []
    previous_blank = False
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(f"**{label}:**") for label in labels):
            normalized_lines.append(f"- {stripped.rstrip()}")
            previous_blank = False
            continue
        if stripped.startswith("#"):
            if normalized_lines and not previous_blank:
                normalized_lines.append("")
            normalized_lines.append(stripped)
            normalized_lines.append("")
            previous_blank = True
            continue
        if not stripped:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line.rstrip())
        previous_blank = False
    return "\n".join(normalized_lines).strip() + "\n"



def run_jobfit(cv_file: str, preferences: str):
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

        yield emit("Generating search query...")
        query = generate_search_query(cv_text, preferences)
        yield emit(f"Query: {query}")

        yield emit("Searching for jobs...")
        raw_sources = search_jobs(query)
        job_listings, other = filter_job_listings(raw_sources)
        yield emit(f"Found {len(job_listings)} direct job listings, {len(other)} other results")

        # One retry with modified query if too few results
        if len(job_listings) < 8:
            retry_query = f"{query} careers apply remote"
            yield emit(f"Too few results. Retrying with: {retry_query}")
            retry_sources = search_jobs(retry_query)
            raw_sources = deduplicate_sources(raw_sources, retry_sources)
            job_listings, other = filter_job_listings(raw_sources)
            yield emit(f"After retry: {len(job_listings)} job listings found")

        if not job_listings:
            yield emit("No job listings found. Try broader preferences.")
            return

        yield emit(f"Selecting top {MAX_SOURCES_TO_SCRAPE} listings to analyze...")
        try:
            selected, skipped = select_sources_with_kimi(job_listings, preferences)
        except Exception:
            selected = job_listings[:MAX_SOURCES_TO_SCRAPE]
            skipped = []

        if not selected:
            selected = job_listings[:MAX_SOURCES_TO_SCRAPE]

        # Filter out domains that can't be scraped
        scrapeable = [s for s in selected if not any(d in s["url"].lower() for d in UNSCRAPEABLE_DOMAINS)]
        filtered_out = len(selected) - len(scrapeable)
        if filtered_out:
            yield emit(f"Skipped {filtered_out} unscrapeable source(s) (LinkedIn)")
            selected = scrapeable

        yield emit(f"Scraping {len(selected)} job page(s)...")
        scraped = []
        for i, source in enumerate(selected, start=1):
            yield emit(f"Scraping {i}/{len(selected)}: {source['title']}")
            try:
                result = scrape_url(source)
                if result["scraped_chars"] > 0:
                    scraped.append(result)
                    yield emit(f"  Got {result['scraped_chars']} characters")
            except Exception as exc:
                yield emit(f"  Scrape failed: {exc}")

        if not scraped:
            rows = [f"| {i+1} | {s['title']} | [Open]({s['url']}) |" for i, s in enumerate(selected[:10])]
            report = f"# JobFit AI Results\n\n## Job Links Found\n\n| # | Title | URL |\n| --- | --- | --- |\n{'  '.join(rows)}\n"
            yield emit("Scraping failed. Direct links saved.", report)
            return

        # Collect unscraped listings as backup data for the report
        scraped_urls = {s["url"] for s in scraped}
        additional = [
            s for s in job_listings
            if s["url"] not in scraped_urls
            and not any(d in s["url"].lower() for d in UNSCRAPEABLE_DOMAINS)
        ]

        yield emit("Ranking jobs and writing report...")
        report = generate_report(cv_text, preferences, scraped, skipped, additional)

        yield emit("Done!", report)
    except Exception as exc:
        yield emit(f"Error: {exc}")


CSS = """
.gradio-container { max-width: 1180px !important; }
#run-button { height: 44px; }
#progress-log textarea {
    max-height: 190px !important;
    overflow-y: auto !important;
}
"""

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
    demo.queue().launch(css=CSS, theme=THEME)
