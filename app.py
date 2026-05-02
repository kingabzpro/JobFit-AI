import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import requests
from openai import OpenAI
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape


KIMI_MODEL = "kimi-k2.6"
KIMI_BASE_URL = "https://api.moonshot.ai/v1"
OLOSTEP_SEARCH_URL = "https://api.olostep.com/v1/searches"
OLOSTEP_SCRAPE_URL = "https://api.olostep.com/v1/scrapes"

DEFAULT_PREFERENCES = """Remote data science, AI writer, or technical writer roles in AI, machine learning, data science, or cloud.
Prefer roles focused on technical content, tutorials, developer education, research writing, and AI product storytelling."""

MAX_SEARCH_RESULTS = 10
MIN_DIRECT_SEARCH_RESULTS = 5
MIN_SCRAPED_SOURCES = 3
MAX_SOURCES_TO_SCRAPE = 5
MAX_SCRAPED_CHARS = 9000
MAX_SEARCH_QUERY_WORDS = 12
MAX_SEARCH_QUERY_CHARS = 100
FALLBACK_SEARCH_QUERY = "remote AI technical writer machine learning careers"

DIRECT_JOB_URL_HINTS = [
    "greenhouse.io",
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "lever.co",
    "jobs.ashbyhq.com",
    "workable.com",
    "smartrecruiters.com",
    "bamboohr.com/careers",
    "myworkdayjobs.com",
    "jobs.lever.co",
    "/careers/",
    "/jobs/",
    "/job/",
    "/positions/",
    "/openings/",
]

AGGREGATOR_URL_HINTS = [
    "indeed.com",
    "ziprecruiter.com",
    "lensa.com",
    "jobright.ai",
    "remotive.com/remote-jobs",
    "workingincontent.com",
    "dailyremote.com",
    "bebee.com",
    "talents.vaia.com",
    "glassdoor.com",
    "monster.com",
    "linkedin.com/jobs",
    "simplyhired.com",
    "careerbuilder.com",
    "dice.com",
    "flexjobs.com",
    "wellfound.com/jobs",
    "upwork.com",
    "freelancer.com",
    "builtin.com/jobs",
]

JOB_BOARD_TITLE_HINTS = [
    "best remote",
    "jobs in",
    "remote jobs",
    "job search",
    "job listings",
    "hiring now",
    "open jobs",
    "top jobs",
    "latest jobs",
]

SEARCH_PAGE_PATH_HINTS = [
    "/search/",
    "/jobs/search",
    "/content/search",
    "/remote/content/search",
    "?q=",
    "?query=",
]

CAREERS_SOURCE_URL_HINTS = [
    "greenhouse.io",
    "lever.co",
    "ashbyhq.com",
    "workable.com",
    "smartrecruiters.com",
    "myworkdayjobs.com",
    "/careers",
    "/jobs",
    "/openings",
    "/positions",
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


def looks_like_direct_job_url(url: str, title: str = "", description: str = "") -> bool:
    normalized = url.lower()
    source_text = f"{title} {description}".lower()

    if any(hint in normalized for hint in AGGREGATOR_URL_HINTS):
        return False

    if any(hint in source_text for hint in JOB_BOARD_TITLE_HINTS):
        return False

    if any(hint in normalized for hint in SEARCH_PAGE_PATH_HINTS):
        return False

    if normalized.rstrip("/").endswith(("/jobs", "/careers", "/openings", "/positions")):
        return False

    return any(hint in normalized for hint in DIRECT_JOB_URL_HINTS)


def looks_like_scrape_candidate_url(url: str, title: str = "", description: str = "") -> bool:
    normalized = url.lower()
    source_text = f"{title} {description}".lower()

    if any(hint in normalized for hint in AGGREGATOR_URL_HINTS):
        return False

    if any(hint in source_text for hint in JOB_BOARD_TITLE_HINTS):
        return False

    if any(hint in normalized for hint in SEARCH_PAGE_PATH_HINTS):
        return False

    return any(hint in normalized for hint in CAREERS_SOURCE_URL_HINTS)


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
            results.append(
                {
                    "index": index,
                    "title": link.get("title", "Untitled source"),
                    "url": link["url"],
                    "description": link.get("description", ""),
                }
            )
    return results


def merge_search_results(*result_sets: list[dict]) -> list[dict]:
    merged = []
    seen_urls = set()

    for results in result_sets:
        for result in results:
            normalized_url = result["url"].rstrip("/").lower()
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            merged.append(result)

    for index, result in enumerate(merged, start=1):
        result["index"] = index

    return merged


def build_fallback_query(primary_query: str, preferences: str) -> str:
    fallback_query = fallback_search_query(preferences)
    if fallback_query.lower() == primary_query.lower():
        fallback_query = FALLBACK_SEARCH_QUERY
    return sanitize_search_query(fallback_query, preferences)


def build_fallback_queries(primary_query: str, preferences: str) -> list[str]:
    candidates = [
        build_fallback_query(primary_query, preferences),
        "remote AI technical writer apply careers",
        "machine learning technical writer apply careers",
        "developer education AI writer careers",
        "greenhouse lever ashby AI technical writer",
        "greenhouse remote AI technical writer",
        "lever remote AI technical writer",
        "ashby remote AI technical writer",
        "workable remote AI technical writer",
        "smartrecruiters remote AI technical writer",
    ]
    queries = []
    seen = {primary_query.lower()}

    for candidate in candidates:
        query = sanitize_search_query(candidate, preferences)
        if query and query.lower() not in seen:
            seen.add(query.lower())
            queries.append(query)

    return queries


def filter_direct_job_sources(sources: list[dict]) -> tuple[list[dict], list[dict]]:
    direct_sources = []
    rejected_sources = []

    for source in sources:
        if looks_like_direct_job_url(source["url"], source.get("title", ""), source.get("description", "")):
            direct_sources.append(source)
        else:
            rejected_sources.append(
                {
                    **source,
                    "reason": "Likely aggregator/search page, not a direct job-listing URL",
                }
            )

    for index, source in enumerate(direct_sources, start=1):
        source["index"] = index

    return direct_sources, rejected_sources


def filter_scrape_candidate_sources(sources: list[dict]) -> list[dict]:
    candidates = []
    seen_urls = set()

    for source in sources:
        normalized_url = source["url"].rstrip("/").lower()
        if normalized_url in seen_urls:
            continue
        if looks_like_scrape_candidate_url(source["url"], source.get("title", ""), source.get("description", "")):
            seen_urls.add(normalized_url)
            candidates.append(
                {
                    **source,
                    "selection_reason": "Fallback company or ATS careers source",
                }
            )

    for index, source in enumerate(candidates, start=1):
        source["index"] = index

    return candidates


def scrape_url(source: dict) -> dict:
    response = requests.post(
        OLOSTEP_SCRAPE_URL,
        headers=olostep_headers(),
        json={"url_to_scrape": source["url"], "formats": ["markdown"]},
        timeout=120,
    )
    response.raise_for_status()

    markdown = response.json().get("result", {}).get("markdown_content") or ""
    return {
        **source,
        "scraped_chars": len(markdown),
        "markdown": markdown[:MAX_SCRAPED_CHARS],
    }


def select_sources_with_kimi(search_results: list[dict], preferences: str) -> tuple[list[dict], list[dict]]:
    prompt = f"""
Select up to {MAX_SOURCES_TO_SCRAPE} direct job listing sources to scrape from these filtered search results.

Candidate preferences:
{preferences}

Search results:
{json.dumps(search_results, indent=2)}

Choose only real, specific job posting pages that a candidate can apply to directly.
Reject broad job boards, aggregator pages, search result pages, category pages, low-detail directories, spam, and unrelated pages.
If there are no direct job listings, select nothing.

Return only JSON in this exact shape:
{{
  "selected": [
    {{"index": 1, "reason": "short reason"}}
  ],
  "skipped": [
    {{"index": 2, "reason": "short reason"}}
  ]
}}
"""
    selection = parse_json_response(ask_kimi(prompt))
    selected_sources = []

    for item in selection.get("selected", []):
        selected_index = item.get("index")
        match = next((source for source in search_results if source["index"] == selected_index), None)
        if match:
            selected_sources.append({**match, "selection_reason": item.get("reason", "Selected by Kimi")})

    return selected_sources[:MAX_SOURCES_TO_SCRAPE], selection.get("skipped", [])


def generate_search_query(cv_text: str, preferences: str) -> str:
    prompt = f"""
Create exactly one short focused web search query to find relevant job postings for this candidate.

Candidate CV excerpt:
{cv_text[:5000]}

Job preferences:
{preferences}

Keep it short: 6 to 10 words maximum.
Focus on the best role title plus 2 or 3 must-have skills.
Prefer semantic terms such as remote, AI, machine learning, technical writer, technical content, developer education, curriculum, data science, careers.
Do not use site: operators, OR operators, parentheses, quotes, company names, ATS names, or job-board names.
Do not include generic filler such as hiring immediately, apply now, full time, 2024, 2025, or 2026.

Return only the search query. No bullets, no explanation.
"""
    return sanitize_search_query(ask_kimi(prompt), preferences)


def sanitize_search_query(query: str, preferences: str) -> str:
    query = query.strip().strip('"').strip("'")
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"\b(AND|OR)\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\bsite:\S+", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"[()\"']", " ", query)
    query = re.sub(r"\b(2024|2025|2026|apply now|hiring immediately|full time)\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip()

    if not query or is_overloaded_search_query(query):
        query = fallback_search_query(preferences)

    words = query.split()

    if len(words) > MAX_SEARCH_QUERY_WORDS:
        query = " ".join(words[:MAX_SEARCH_QUERY_WORDS])

    if len(query) > MAX_SEARCH_QUERY_CHARS:
        query = query[:MAX_SEARCH_QUERY_CHARS].rsplit(" ", 1)[0]

    return query


def is_overloaded_search_query(query: str) -> bool:
    normalized = query.lower()
    return (
        " site:" in f" {normalized}"
        or " or " in f" {normalized} "
        or normalized.count(":") > 1
        or normalized.count("site:") > 0
    )


def fallback_search_query(preferences: str) -> str:
    preference_text = preferences.lower()

    role = "technical writer"
    if "developer education" in preference_text:
        role = "developer education"
    elif "technical content" in preference_text:
        role = "technical content writer"
    elif "data science" in preference_text and "writer" not in preference_text:
        role = "data science writer"

    domain = "AI"
    if "machine learning" in preference_text or "mlops" in preference_text:
        domain = "machine learning"
    elif "cloud" in preference_text:
        domain = "cloud"

    return f"remote {domain} {role} careers"


def fallback_selected_sources(direct_sources: list[dict]) -> list[dict]:
    return [
        {
            **source,
            "selection_reason": "Fallback selection from direct job-listing search results",
        }
        for source in direct_sources[:MAX_SOURCES_TO_SCRAPE]
    ]


def backfill_selected_sources(selected_sources: list[dict], candidate_sources: list[dict]) -> list[dict]:
    selected = list(selected_sources)
    selected_urls = {source["url"].rstrip("/").lower() for source in selected}

    for source in candidate_sources:
        if len(selected) >= MAX_SOURCES_TO_SCRAPE:
            break
        normalized_url = source["url"].rstrip("/").lower()
        if normalized_url in selected_urls:
            continue
        selected_urls.add(normalized_url)
        selected.append(
            {
                **source,
                "selection_reason": source.get("selection_reason", "Backfilled so enough sources are scraped"),
            }
        )

    return selected


def generate_unscraped_sources_report(direct_sources: list[dict], rejected_sources: list[dict]) -> str:
    rows = []
    for index, source in enumerate(direct_sources[:MAX_SOURCES_TO_SCRAPE], start=1):
        rows.append(
            f"| {index} | {source['title']} | [Open job]({source['url']}) | Needs manual review |"
        )

    rejected_lines = [
        f"- **{source['title']}:** {source.get('reason', 'Rejected source')}"
        for source in rejected_sources[:5]
    ]

    report = f"""
# JobFit AI Results

## Direct Job Links Found

The search found direct job-listing URLs, but none of the selected pages returned enough scrapeable content for ranking. Open these job links directly and rerun the app if the pages are accessible.

| Rank | Role / Source | URL | Status |
| --- | --- | --- | --- |
{chr(10).join(rows)}

## Skipped Or Weak Sources

{chr(10).join(rejected_lines) if rejected_lines else "- No aggregator or search sources were rejected."}
"""
    return normalize_report_markdown(report)


def generate_report(cv_text: str, preferences: str, scraped_sources: list[dict], rejected_sources: list[dict], skipped_sources: list[dict]) -> str:
    analysis_input = [
        {
            "title": item["title"],
            "url": item["url"],
            "selection_reason": item.get("selection_reason", ""),
            "scraped_chars": item["scraped_chars"],
            "markdown": item["markdown"],
        }
        for item in scraped_sources
    ]

    prompt = f"""
Rank these scraped direct job listings for the candidate.

Candidate CV:
{cv_text}

Job preferences:
{preferences}

Scraped sources:
{json.dumps(analysis_input, ensure_ascii=False, indent=2)}

Rejected search sources:
{json.dumps(rejected_sources[:10], ensure_ascii=False, indent=2)}

Kimi skipped sources:
{json.dumps(skipped_sources, ensure_ascii=False, indent=2)}

Write a clean, readable Markdown report.

Formatting rules:
- Use valid Markdown headings with #, ##, and ###.
- Put a blank line after every heading.
- Use a Markdown table for the ranked summary.
- Use bullet lists for reasons, concerns, and application angles.
- Keep each bullet short and specific.
- Do not write long paragraph blocks inside the ranking details.
- Do not recommend aggregator pages, search pages, or broad job-board category pages as jobs to apply for.
- Only rank direct job listing URLs. If a scraped source is not a direct job listing, move it to Skipped Or Weak Sources.
- Include clickable Markdown links for URLs: [Apply or view job](URL).

Return exactly this structure:

# JobFit AI Results

## Best Job To Apply For

**Role:** <job title>
**Company:** <company>
**Fit Score:** <score>/100
**URL:** [Apply or view job](<url>)

**Why this is the best fit:**

- <specific reason 1>
- <specific reason 2>
- <specific reason 3>

## Ranked Summary

| Rank | Role / Source | Company | Fit | Recommendation |
| --- | --- | --- | --- | --- |
| 1 | <role> | <company> | <score>/100 | Apply / Maybe / Skip |

## Detailed Notes

### 1. <Job title or source title>

**Company:** <company>
**Fit Score:** <score>/100
**Recommendation:** Apply / Maybe / Skip
**URL:** [Apply or view job](<url>)

**Why it matches:**

- <bullet>
- <bullet>

**Concerns:**

- <bullet>
- <bullet>

**Application angle:**

- <bullet>
- <bullet>

## Skipped Or Weak Sources

- **<source title>:** <why it is weak, generic, inaccessible, or not worth applying directly>
"""
    return normalize_report_markdown(ask_kimi(prompt))


def normalize_report_markdown(markdown_text: str) -> str:
    """Clean up common model Markdown mistakes before display/PDF export."""
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


def table_markdown(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "No rows."

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []

    for row in rows:
        values = []
        for column in columns:
            value = str(row.get(column, "")).replace("\n", " ").replace("|", "\\|")
            values.append(value[:180])
        body.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *body])


def markdown_inline_to_reportlab(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<u><font color="blue">\1</font></u> (\2)', text)
    return text


def markdown_to_pdf(markdown_text: str, output_path: str) -> str:
    styles = getSampleStyleSheet()
    styles["Title"].fontName = "Helvetica-Bold"
    styles["Heading1"].spaceBefore = 14
    styles["Heading1"].spaceAfter = 8
    styles["Heading2"].spaceBefore = 12
    styles["Heading2"].spaceAfter = 6
    styles["Heading3"].spaceBefore = 10
    styles["Heading3"].spaceAfter = 4
    styles["BodyText"].leading = 14

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="JobFit AI Results",
    )
    story = []
    bullet_items = []

    def flush_bullets():
        nonlocal bullet_items
        if bullet_items:
            story.append(ListFlowable(bullet_items, bulletType="bullet", leftIndent=18))
            story.append(Spacer(1, 6))
            bullet_items = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_bullets()
            story.append(Spacer(1, 4))
            continue

        if line.startswith("|") and line.endswith("|"):
            flush_bullets()
            cells = [markdown_inline_to_reportlab(cell.strip()) for cell in line.strip("|").split("|")]
            if set(cells) == {"---"} or all(set(cell.replace(" ", "")) <= {"-"} for cell in cells):
                continue
            table_data = [[Paragraph(cell, styles["BodyText"]) for cell in cells]]
            table = Table(table_data, hAlign="LEFT", repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 4))
            continue

        if line.startswith("- "):
            bullet_items.append(ListItem(Paragraph(markdown_inline_to_reportlab(line[2:]), styles["BodyText"])))
            continue

        flush_bullets()
        if line.startswith("# "):
            story.append(Paragraph(markdown_inline_to_reportlab(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(markdown_inline_to_reportlab(line[3:]), styles["Heading1"]))
        elif line.startswith("### "):
            story.append(Paragraph(markdown_inline_to_reportlab(line[4:]), styles["Heading2"]))
        else:
            story.append(Paragraph(markdown_inline_to_reportlab(line), styles["BodyText"]))

    flush_bullets()
    doc.build(story)
    return output_path


def run_jobfit(cv_file: str, preferences: str):
    logs = []

    def emit(message: str, report: str = "", pdf_path: str | None = None, progress_value: float | None = None):
        if progress_value is None:
            logs.append(message)
        else:
            logs.append(f"{int(progress_value * 100):>3}% - {message}")
        return "\n".join(logs), report, pdf_path

    if not cv_file:
        yield emit("Upload a CV PDF first.", progress_value=0)
        return

    if not preferences.strip():
        yield emit("Enter job preferences first.", progress_value=0)
        return

    try:
        yield emit("Checking API keys...", progress_value=0.05)
        require_env("MOONSHOT_API_KEY")
        require_env("OLOSTEP_API_KEY")

        yield emit("Reading CV PDF...", progress_value=0.12)
        cv_text, cv_logs = read_cv(cv_file)
        for log in cv_logs:
            yield emit(log)
        yield emit(f"CV text ready: {len(cv_text)} characters.", progress_value=0.2)

        yield emit("Asking Kimi to create one short direct-job search query...", progress_value=0.28)
        query = generate_search_query(cv_text, preferences)
        yield emit(f"Search query: {query}")

        yield emit(f"Searching {MAX_SEARCH_RESULTS} sources with Olostep...", progress_value=0.36)
        raw_sources = search_jobs(query, limit=MAX_SEARCH_RESULTS)
        direct_sources, rejected_sources = filter_direct_job_sources(raw_sources)
        yield emit(f"Raw sources: {len(raw_sources)}")
        yield emit(f"Direct job-listing sources: {len(direct_sources)}")
        yield emit(f"Rejected aggregator/search sources: {len(rejected_sources)}")

        fallback_queries = build_fallback_queries(query, preferences)
        for fallback_index, fallback_query in enumerate(fallback_queries, start=1):
            if len(direct_sources) >= MIN_DIRECT_SEARCH_RESULTS:
                break
            yield emit(f"Few direct listings found. Trying fallback query {fallback_index}: {fallback_query}", progress_value=0.42)
            fallback_raw_sources = search_jobs(fallback_query, limit=MAX_SEARCH_RESULTS)
            raw_sources = merge_search_results(raw_sources, fallback_raw_sources)
            direct_sources, rejected_sources = filter_direct_job_sources(raw_sources)
            yield emit(f"Combined raw sources: {len(raw_sources)}")
            yield emit(f"Combined direct job-listing sources: {len(direct_sources)}")
            yield emit(f"Combined rejected aggregator/search sources: {len(rejected_sources)}")

        if not direct_sources:
            direct_sources = filter_scrape_candidate_sources(raw_sources)
            if direct_sources:
                yield emit(f"No exact direct job pages found. Using {len(direct_sources)} company/ATS source candidate(s).")
            else:
                report = (
                    "# JobFit AI Results\n\n"
                    "No scrapeable company, ATS, or direct job-listing URLs were found. "
                    "Try broader preferences or rerun the search."
                )
                pdf_path = str(Path(tempfile.gettempdir()) / f"jobfit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                markdown_to_pdf(report, pdf_path)
                yield emit("No scrapeable job sources found. Report generated.", report, pdf_path, progress_value=1)
                return
        elif len(direct_sources) < MAX_SOURCES_TO_SCRAPE:
            supplemental_sources = filter_scrape_candidate_sources(raw_sources)
            direct_sources = merge_search_results(direct_sources, supplemental_sources)
            yield emit(f"Scrape candidate pool after backfill: {len(direct_sources)}")

        yield emit("Asking Kimi to select direct job listings to scrape...", progress_value=0.48)
        try:
            selected_sources, skipped_sources = select_sources_with_kimi(direct_sources, preferences)
        except Exception as exc:
            yield emit(f"Kimi source selection failed, using top direct links instead: {exc}")
            selected_sources, skipped_sources = [], []

        if not selected_sources:
            selected_sources = fallback_selected_sources(direct_sources)
            yield emit(f"Kimi selected 0 sources. Using top {len(selected_sources)} direct job link(s) instead.")
        elif len(selected_sources) < min(MAX_SOURCES_TO_SCRAPE, len(direct_sources)):
            selected_sources = backfill_selected_sources(selected_sources, direct_sources)
            yield emit(f"Backfilled selection to {len(selected_sources)} source(s) for scraping.")

        yield emit(f"Selected {len(selected_sources)} direct job listing(s) for scraping.")

        scraped_sources = []
        attempted_urls = set()
        for index, source in enumerate(selected_sources, start=1):
            scrape_progress = 0.52 + (0.24 * ((index - 1) / max(len(selected_sources), 1)))
            yield emit(f"Scraping {index}/{len(selected_sources)}: {source['title']}", progress_value=scrape_progress)
            attempted_urls.add(source["url"])
            try:
                scraped = scrape_url(source)
                if scraped["scraped_chars"] > 0:
                    scraped_sources.append(scraped)
                    yield emit(f"Scraped {scraped['scraped_chars']} characters from {source['url']}")
                else:
                    yield emit(f"Scrape returned empty content for {source['url']}")
            except Exception as exc:
                yield emit(f"Scrape failed for {source['url']}: {exc}")

        if len(scraped_sources) < MIN_SCRAPED_SOURCES:
            remaining_sources = [source for source in direct_sources if source["url"] not in attempted_urls]
            fallback_sources = fallback_selected_sources(remaining_sources)[: max(0, MIN_SCRAPED_SOURCES - len(scraped_sources))]

            if fallback_sources:
                yield emit(f"Trying {len(fallback_sources)} remaining direct job link(s)...", progress_value=0.76)

            for index, source in enumerate(fallback_sources, start=1):
                yield emit(f"Fallback scrape {index}/{len(fallback_sources)}: {source['title']}")
                try:
                    scraped = scrape_url(source)
                    if scraped["scraped_chars"] > 0:
                        scraped_sources.append(scraped)
                        yield emit(f"Scraped {scraped['scraped_chars']} characters from {source['url']}")
                except Exception as exc:
                    yield emit(f"Fallback scrape failed for {source['url']}: {exc}")

        if not scraped_sources:
            report = generate_unscraped_sources_report(direct_sources, rejected_sources)
            pdf_path = str(Path(tempfile.gettempdir()) / f"jobfit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            markdown_to_pdf(report, pdf_path)
            yield emit("Scraping failed, but direct job links were saved in the report.", report, pdf_path, progress_value=1)
            return

        yield emit("Asking Kimi to rank jobs and write the report...", progress_value=0.82)
        report = generate_report(cv_text, preferences, scraped_sources, rejected_sources, skipped_sources)

        yield emit("Creating PDF report...", progress_value=0.94)
        pdf_path = str(Path(tempfile.gettempdir()) / f"jobfit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        markdown_to_pdf(report, pdf_path)

        yield emit("Done. Report and PDF are ready.", report, pdf_path, progress_value=1)
    except Exception as exc:
        yield emit(f"Error: {exc}", progress_value=1)


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


def set_run_button_state(is_enabled: bool):
    return gr.update(interactive=is_enabled)


with gr.Blocks(title="JobFit AI") as demo:
    gr.Markdown(
        """
# JobFit AI

Upload a CV, describe the job preferences, and generate a ranked list of direct job listings to apply for.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            cv_input = gr.File(label="CV PDF", file_types=[".pdf"], type="filepath")
            preferences_input = gr.Textbox(
                label="Job preferences",
                value=DEFAULT_PREFERENCES,
                lines=8,
                placeholder="Describe role type, industry, remote/location preferences, seniority, and topics.",
            )
            run_button = gr.Button("Generate JobFit Report", variant="primary", elem_id="run-button")
        with gr.Column(scale=1):
            logs_output = gr.Textbox(
                label="Progress log",
                lines=8,
                max_lines=8,
                interactive=False,
                elem_id="progress-log",
            )
            pdf_output = gr.File(label="Download PDF report")

    gr.Markdown("## Final Report")
    report_output = gr.Markdown(label="Report")

    run_event = run_button.click(
        fn=lambda: set_run_button_state(False),
        outputs=run_button,
        show_progress="hidden",
    )
    run_event.then(
        fn=run_jobfit,
        inputs=[cv_input, preferences_input],
        outputs=[logs_output, report_output, pdf_output],
        show_progress="hidden",
    ).then(
        fn=lambda: set_run_button_state(True),
        outputs=run_button,
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue().launch(css=CSS, theme=THEME)
