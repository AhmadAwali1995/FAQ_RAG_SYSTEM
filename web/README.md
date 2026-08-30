# Chat UI (optional add-on)

A single-file browser chat interface for the FAQ RAG API. Plain HTML/CSS/JS —
no build step, no npm, no CDN, so it works offline and needs no toolchain.

```
web/
└── index.html    ← the entire UI
```

It is served by the API at **http://127.0.0.1:8000/chat** (`/` redirects there).

```bash
python -m faqrag.api      # then open http://127.0.0.1:8000/chat
```

## What it shows

- **Bilingual chat** — detects Arabic vs. English per message and flips direction
  (RTL/LTR) automatically, including in the input box as you type.
- **Citations** — every answer lists the FAQ ids it drew from; click a chip or
  expand *cited sources* to read the exact FAQ question and answer used.
- **Confidence** — a percentage bar from the API's `confidence` field.
- **Low-confidence answers look different.** When the API returns
  `confident: false`, the reply is styled as a warning and labelled
  *"No confident match"*, so a refusal can never be mistaken for a sourced
  answer. This is the single most important behaviour in the UI.
- **Debugging** (Settings ⚙) — *show all retrieved chunks* dims the ones that
  weren't cited; *show retrieval scores* adds a per-candidate table of vector,
  BM25, and rerank scores from `POST /retrieve`.
- **Top-K slider**, theme toggle, and clear-conversation.

Because a reranked query takes ~10s, the thinking indicator shows an elapsed
timer rather than an indeterminate spinner.

Conversation history is in-memory only and is not sent back to the API — each
question is answered independently, exactly as the CLI would answer it. Only
theme, top-k, and the debug toggles persist (in `localStorage`).

## Removing it

The UI is fully detachable. Nothing in the retrieval or generation pipeline
imports it.

**Turn it off, keep the files:**

```bash
FAQRAG_ENABLE_CHAT_UI=false     # in .env
```

**Delete it permanently:**

```bash
rm -rf web/ src/faqrag/web.py
```

That's it. The guarded import in `src/faqrag/api.py` catches the missing module,
and the API keeps serving `/health`, `/query`, and `/retrieve` unchanged. No
other file needs editing, and no test depends on it.

Optionally also drop `enable_chat_ui` and `web_dir` from
`src/faqrag/config.py` — harmless if left.

## Pointing it at a different backend

The page calls the origin that served it. To host it elsewhere, set the `API`
constant near the top of the `<script>` block to your API's base URL and enable
CORS on the FastAPI app.
