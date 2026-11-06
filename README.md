# Ask Your Document

A local Flask app that answers questions about a PDF and shows the source passages with page citations. It uses the Gemini API for embeddings and answers.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your Gemini API key in `.env`:

```text
GEMINI_API_KEY=your-key-here
```

Keep the key private. `.env` is ignored by Git and is never sent to the browser. The app reads this file when it starts, so restart the server after changing it. Alternatively, set `$env:GEMINI_API_KEY` in the same PowerShell window before starting the app.

## Run

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Usage

1. Select one or several PDFs, or drag and drop them anywhere on the page (up to 20 MB each).
2. The indexed-document list shows each file and its passage count. Use its Remove button to delete an index.
3. Use document-based suggested questions or enter your own question (up to 2,000 characters). Press Enter or click Ask; answers stream as generated and cite the filename and page (for example, `report.pdf, p. 4`). Expand a source to read its passage.

## Optional settings

- `GEMINI_MODEL`: answer model (default `gemini-3.8-flash`)
- `GEMINI_EMBED_MODEL`: embedding model (default `gemini-embedding-001`)

## Limits

- Saved indexes are stored under `data/` and loaded at startup, so documents remain available after restarting without re-embedding.
- Removing a document also deletes its saved index files.
- Scanned PDFs are not supported (no OCR).
- This is a single-user local app, not an authenticated multi-user service. Do not expose it directly to the public internet.
- The Gemini API key is read from the server environment and is never sent to the browser.
- When you index a PDF, its extracted text is sent to Google's Gemini API to create embeddings. Questions and retrieved document passages are also sent to Gemini to generate answers. Avoid uploading confidential or sensitive documents unless you have reviewed Google's applicable terms and privacy practices.
- API requests may incur charges or use your provider quota. Keep your key private, monitor usage, and configure provider-side usage limits where available.
- The Gemini client is reused while the server is running. Answers stream to the page, use minimal model thinking, retrieve two passages, and are capped at 160 output tokens for concise replies. Total generation time still depends on the selected model, API service, and network.
- Suggested questions are derived from frequent terms in sampled document passages.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

## Troubleshooting

- If upload says the API key is missing, create `.env` beside `app.py`, set `GEMINI_API_KEY=your-key-here`, and restart the server. If indexing or answering fails, check that the configured Gemini models are available to your API key.
- If Gemini reports a rate limit, wait a minute and try again.
- If a PDF is rejected, confirm it is valid, unencrypted, text-based, and smaller than 20 MB. Encrypted PDFs and scanned PDFs without extractable text are not supported.

## Run tests

```powershell
python -m unittest -v
```
