# Mini Spreadsheet Loader

Compact, well-commented project that ingests `.xlsx` and `.xlsb` files.

## Layout
- `mini_project/config.py` – lists input files and the output directory.
- `mini_project/reader.py` – handles Excel loading (switches to `pyxlsb` for `.xlsb`).
- `mini_project/writer.py` – saves DataFrames as CSVs.
- `mini_project/vector_store.py` – pushes rows into a Chroma collection.
- `mini_project/llm.py` – wraps OpenAI Responses API for question answering.
- `mini_project/report.py` – prints a quick summary.
- `mini_project/cli_ingest.py` – houses the ingest CLI command.
- `mini_project/cli_query.py` – houses the ask/chat CLI commands.
- `main.py` – minimal entry point that wires the commands together.
- `ui/` – lightweight React playground (HTML/CSS/JS) for crafting CLI queries.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
- `python main.py ingest` – processes spreadsheets, saves CSVs, and upserts to Chroma.
- `python main.py insight` – generates insight charts (`insights/*.png`) plus JSON summary for the UI.
- `python main.py serve` – launches the Flask API powering the UI (defaults to `0.0.0.0:5000`).
- `python main.py ask "question"` – single-shot query (prompts for missing collection/top-k).
- `python main.py chat` – interactive Q&A loop with menu.

Recommended flow:
1. `pip install -r requirements.txt`
2. `python main.py ingest`
3. `python main.py insight`
4. `python main.py serve` (and open the UI)
5. Use `python main.py ask` / `chat` for CLI-driven Q&A as needed.

Ingesting reads the files listed in `config.py`, normalizes header names by
removing slashes, commas, and other special characters (then converting any
remaining separators to underscores), writes CSV copies under `mini_output/`,
sanitizes any datetime columns to ISO strings, pushes each row into a Chroma
database under `mini_chroma/`, and prints rows/columns for every dataset.

To ask questions over the stored data, set `OPENAI_API_KEY`, then either ask a single question (you will be prompted to pick a collection if you omit it):

```bash
set OPENAI_API_KEY=sk-...
python main.py ask "Who has Azure experience?" --db-path mini_chroma
```

Leave `--collection` or `--top-k` empty to be prompted interactively.

Or start an interactive terminal session:

```bash
python main.py chat --db-path other_db_dir
```

Use the on-screen menu to choose **Ask question**, **Change top-k**, or **Quit** each round. You can
also supply `--collection some_name` / `--top-k 5` to skip the pickers. The `--db-path`
flag lets you point at any Chroma directory you want to query.

## Mini React Playground
- Start the Flask server (auto-opens the UI in your browser): `python main.py serve`.
- If the browser does not open automatically, visit `http://127.0.0.1:5000/`.
- Fill in the question, pick a database (or type a custom path), optionally set collection, and type the top-k value (text input).
- Click “Generate & Query” to both see the CLI command and run the query via `/api/query`; answers/errors show below the form.
- Switch to the **Insights** tab to view live charts (experience distribution, top bench skills, skill overlap) sourced from the `/api/insights` endpoint. Run `python main.py insight` to refresh the backing data if the CSVs change.

