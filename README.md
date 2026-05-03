# JobFit AI

JobFit AI is a Gradio app that reads a candidate CV, runs a Kimi 2.6 agent through the OpenAI Agents SDK, searches the web with Olostep, and produces a ranked Markdown job-fit report.

The app is aimed at AI, data science, technical writing, technical content, curriculum, and developer education roles. The preferences box can be changed for other job searches.

<img width="1496" height="855" alt="image" src="https://github.com/user-attachments/assets/b17eb09b-bdea-4af0-a639-7364aed1867c" />

## What It Does

- Reads a CV PDF with `pypdf`.
- Runs a Kimi 2.6 agent through the OpenAI Agents SDK.
- Gives the agent two tools: `search_jobs` and `read_job_page`.
- Uses Olostep Search to find job listings.
- Uses Olostep Scrape to read selected job pages.
- Streams simple progress logs for tool calls, parameters, and tool output sizes.
- Generates a Markdown report with a best match, ranked jobs, job notes, and rejected jobs.

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

## Current Workflow

1. Read the uploaded CV.
2. Build one prompt from the CV and job preferences.
3. Run the `JobFit AI` agent with Kimi 2.6.
4. The agent searches for job listings with `search_jobs`.
5. The agent reads selected listings with `read_job_page`.
6. The agent writes the final Markdown report.
7. The app displays the report in Gradio.

Progress appears in the log box while the app runs. The generate button is disabled during a run to prevent duplicate requests.

## Report Format

The report asks the model to keep the output simple and practical:

- no em dashes
- no contractions
- short bullets
- clickable job links
- at least 5 ranked jobs when enough usable results exist
- rejected jobs with name, link, and reason

The main sections are:

- `Best Match`
- `Ranked Jobs`
- `Job Notes`
- `Rejected Jobs`

## Defaults

- Kimi model: `kimi-k2.6`
- Moonshot base URL: `https://api.moonshot.ai/v1`
- Agent max turns: `25`
- Search results per tool call: `10`
- Page reads requested in the prompt: up to `3`
- Scraped characters per page: `8000`

## Notes

The app uses Kimi 2.6 through Moonshot's OpenAI-compatible endpoint with the OpenAI Agents SDK:

```python
OpenAIChatCompletionsModel(
    model="kimi-k2.6",
    openai_client=AsyncOpenAI(
        api_key=os.environ["MOONSHOT_API_KEY"],
        base_url="https://api.moonshot.ai/v1",
    ),
)
```

Olostep is called directly with HTTP requests inside the agent tools.
