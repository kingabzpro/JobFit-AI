# JobFit AI

JobFit AI is a Gradio app that reads a candidate CV, searches for relevant job sources, scrapes selected listings with Olostep, and uses Kimi 2.6 to produce a ranked job-fit report with a downloadable PDF.

The app is designed for AI, data science, technical writing, technical content, curriculum, and developer education roles, but the preferences box can be changed for other job searches.

## What It Does

- Uploads and reads a CV PDF with `pypdf`.
- Uses Kimi 2.6 through the OpenAI-compatible SDK to create a focused search query.
- Calls Olostep Search for job sources.
- Filters out broad job boards and aggregator/search pages.
- Tries multiple focused fallback searches when too few direct listings are found.
- Selects and backfills up to 5 sources to scrape.
- Scrapes pages with Olostep Scrape.
- Uses Kimi to rank the scraped jobs and write a Markdown report.
- Exports the final report as a PDF.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the required API keys.

PowerShell:

```powershell
$env:MOONSHOT_API_KEY="your_moonshot_key"
$env:OLOSTEP_API_KEY="your_olostep_key"
```

macOS/Linux:

```bash
export MOONSHOT_API_KEY="your_moonshot_key"
export OLOSTEP_API_KEY="your_olostep_key"
```

## Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

Upload a CV PDF, edit the job preferences if needed, then click **Generate JobFit Report**.

## Workflow

The app keeps the workflow intentionally simple:

1. Read the CV.
2. Ask Kimi for one concise search query.
3. Search Olostep for 10 sources.
4. If too few useful sources are found, run focused fallback searches.
5. Reject broad job boards, search pages, and category pages.
6. Ask Kimi which sources are best to scrape.
7. Backfill the scrape list so the app has enough sources to evaluate.
8. Scrape selected pages.
9. Ask Kimi to rank jobs and generate the report.
10. Create a PDF download.

Progress appears in the log box while the app runs. The generate button is disabled during a run to prevent duplicate requests.

## Search Behavior

The app avoids general job-board result pages such as LinkedIn search pages, Built In category pages, Indeed, ZipRecruiter, Glassdoor, and similar aggregators. It prefers:

- Direct ATS job pages such as Greenhouse, Lever, Ashby, Workable, SmartRecruiters, and Workday.
- Company careers pages when exact job pages are not available.
- URLs that look like specific job postings rather than category/search results.

Current defaults:

- Search results per query: `10`
- Target direct/candidate sources: `5`
- Sources scraped for ranking: up to `5`
- Minimum successful scraped sources before fallback scraping: `3`

## Notes

This project uses Kimi 2.6 through Moonshot's OpenAI-compatible endpoint:

```python
OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
)
```

Olostep is called directly with HTTP requests rather than the Python SDK.
