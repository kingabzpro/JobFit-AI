# JobFit AI

JobFit AI is a Gradio app that reads a candidate CV, searches for relevant job listings, scrapes selected pages with Olostep, and uses Kimi K2.6 to produce a ranked job-fit report with at least 5 job entries.

The app is designed for AI, data science, technical writing, technical content, curriculum, and developer education roles, but the preferences box can be changed for any job search.

<img width="1496" height="855" alt="image" src="https://github.com/user-attachments/assets/b17eb09b-bdea-4af0-a639-7364aed1867c" />


## What It Does

- Uploads and reads a CV PDF with `pypdf`.
- Uses Kimi K2.6 to create a focused search query.
- Calls Olostep Search for job sources.
- Filters out broad search/category pages, keeping specific job listings from any source (aggregators, ATS platforms, company sites).
- Skips unscrapeable domains (LinkedIn) before scraping.
- Scrapes pages with Olostep Scrape.
- Uses Kimi to rank the scraped jobs and write a Markdown report with at least 5 entries (unscraped listings are included as backup if needed).

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

1. Read the CV.
2. Ask Kimi for one concise search query.
3. Search Olostep for job sources.
4. Filter out broad search/category pages; keep specific job listings from any domain.
5. If fewer than 8 results, retry once with a modified query.
6. Ask Kimi to select at least 5 and up to 10 sources to scrape.
7. Skip unscrapeable domains (LinkedIn).
8. Scrape selected pages.
9. Collect unscraped job listings as backup data.
10. Ask Kimi to rank jobs and generate a report with at least 5 entries.

Progress appears in the log box while the app runs. The generate button is disabled during a run to prevent duplicate requests.

## Search Behavior

The app keeps results from any source — aggregators (Indeed, Glassdoor, ZipRecruiter), ATS platforms (Greenhouse, Lever, Ashby), and company careers pages — as long as the URL points to a **specific job listing**. Only broad search/category pages are filtered out.

Domains that block scrapers (currently LinkedIn) are accepted in search results but skipped during scraping.

Current defaults:

- Search results per query: `20`
- Maximum sources scraped: `10`
- Minimum report entries: `5`

## Notes

This project uses Kimi K2.6 through Moonshot's OpenAI-compatible endpoint:

```python
OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
)
```

Olostep is called directly with HTTP requests rather than the Python SDK.
