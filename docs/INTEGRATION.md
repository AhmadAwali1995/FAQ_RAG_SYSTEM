# Integration API

Two endpoints for connecting an external system (chat widget, WhatsApp bridge,
mobile app, CRM) to the FAQ assistant.

**Base URL:** `http://127.0.0.1:8000` · **Interactive docs:** `/docs`

---

## The call

```http
POST /v1/messages
{"text": "وش طرق الدفع عندكم؟"}
```

```json
200 OK
{
  "answer": "أبشر، عندنا عدة طرق دفع ميسرة تقدر تستخدمها: تابي، تمارا، مدى، فيزا، آبل باي، وسامسونج باي...",
  "cited_faq_ids": ["011"],
  "confident": true,
  "status": "done"
}
```

That is the whole integration. Send the text, read the answer off the response.
No `message_id`, no polling, no second call.

Add `?format=text` and the body is the bare answer as `text/plain`, stripped of
markdown and ready to hand to a text-to-speech engine:

```bash
curl -X POST "http://127.0.0.1:8000/v1/messages?format=text"      -H 'Content-Type: application/json' -d '{"text":"وش هي أكاديمية موفق؟"}'
```
```
أكاديمية موفق هي منظومة تعليمية ذكية، تقدم برامج تأهيل وتدريب متكاملة عبر منصة تعلم واحدة...
```

### One thing to get right

Generation takes **about 10-12 seconds**, and the request is held open for it.
**Set your client timeout above 60 seconds.** A default 10s or 30s timeout will
abandon the request just before the answer lands.

```python
httpx.post(url, json={"text": q}, timeout=120)   # not the 5s default
```

### If an answer is slow

You get `202` with a `message_id` instead of an error, and collect it from
`GET /v1/messages/{message_id}?wait=60`. So a slow answer degrades to a second
call; it never fails outright. In practice this only happens past 60 seconds.

### Opting out of waiting

Pass `wait=0` to always get the `message_id` form immediately. Use that when
something between you and this service — a browser, an API gateway, a webhook
platform — would cut a ten-second request short. That is the only reason to.

## 1. Receive text — `POST /v1/messages`

```http
POST /v1/messages
Content-Type: application/json

{
  "text": "وش طرق الدفع عندكم؟",
  "top_k": 5
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | The question, Arabic or English. 1–2000 chars. |
| `top_k` | int | no | FAQ chunks to retrieve (1–20). Defaults to config. |

**`202 Accepted`** — nothing has been generated yet:

```json
{
  "message_id": "msg_34de6ee9d32a4a63",
  "status": "processing",
  "created_at": "2026-08-30T12:42:25.713407Z",
  "poll_url": "/v1/messages/msg_34de6ee9d32a4a63"
}
```

Errors: `422` empty or oversized `text`.

---

## 2. Collect the answer — `GET /v1/messages/{message_id}`

Add `?wait=N` to block up to N seconds (max 60) for the answer instead of
polling. **Prefer this** — one call with `wait=30` beats fifteen polls.

```http
GET /v1/messages/msg_34de6ee9d32a4a63?wait=30
```

**`200 OK`** once done:

```json
{
  "message_id": "msg_34de6ee9d32a4a63",
  "status": "done",
  "text": "وش طرق الدفع عندكم؟",
  "created_at": "2026-08-30T12:42:25.713407Z",
  "updated_at": "2026-08-30T12:42:36.035Z",

  "answer": "أهلاً بك! عندنا طرق دفع ميسرة كثيرة، تقدر تختار اللي يناسبك: تابي، تمارا، مدى، فيزا، آبل باي، وسامسونج باي.",
  "cited_faq_ids": ["011"],
  "sources": [
    { "faq_id": "011", "category": "Payments & Support", "lang": "ar",
      "question": "ما هي طرق الدفع المتاحة في موفق؟",
      "answer": "يمكنك حجز فحصك الطبي والدفع عبر عدة طرق ميسرة: تابي، تمارا، مدى، فيزا، آبل باي، وسامسونج باي.",
      "score": 0.8706 }
  ],
  "confident": true,
  "confidence": 0.8706,
  "language": "ar",
  "latency_ms": 10322.4,
  "error": null
}
```

### Statuses

| `status` | Meaning | What to do |
|---|---|---|
| `queued` | Accepted, not started | Call again with `wait` |
| `processing` | Being answered | Call again with `wait` |
| `done` | Answer ready | Render `answer` |
| `failed` | Something broke | Read `error`; safe to resubmit |

**Branch on `status`, not on whether `answer` is present.** Every answer field is
`null` until `status` is `done`.

While pending, the response carries `Retry-After: 2`.

Errors: `404` unknown or expired `message_id`; `422` `wait` above 60.

### Two things the client must handle

**`confident: false`** — the FAQ did not cover the question. `answer` is a polite
"I don't have that", and `cited_faq_ids` is empty. Show it differently from a
sourced answer; do not present it as fact.

```js
if (!msg.confident || msg.cited_faq_ids.length === 0) {
  renderAsUnknown(msg.answer);          // no citations, visually distinct
} else {
  renderAnswer(msg.answer, msg.sources);
}
```

### Just the text, ready to speak — `?format=text`

For a caller that relays the reply or feeds it to text-to-speech:

```http
GET /v1/latest?wait=60&format=text
```

```
Content-Type: text/plain; charset=utf-8
X-Message-Status: done
X-Answer-Confident: true
X-Cited-FAQ-Ids: 025,026

أهلاً فيك! إذا كنت فرد: تقدر تحجز فحصك الطبي على طول من منصة موفق أو تطبيق الجوال...
```

The body is the answer and nothing else. Two things it does for you:

**Markdown is stripped.** The model sometimes formats with `**bold**`, bullets,
or numbered lists. On screen that is fine; spoken aloud a synthesiser either
pronounces the markers or stumbles over them. This representation removes them.

**Lines are terminated and joined.** Each line gets sentence punctuation before
being joined into one block, so a synthesiser pauses between list items instead
of reading them as one breathless sentence. The result contains no newlines.

The signals you would otherwise lose move into headers rather than disappearing:
`X-Message-Status`, and once answered, `X-Answer-Confident` and
`X-Cited-FAQ-Ids`.

**JSON stays the default, deliberately.** In text mode a client that ignores the
headers cannot tell a sourced answer from an "I don't have that", and it has no
citations to show. It still receives a sensible sentence, because the
low-confidence reply is itself plainly worded. Use `format=text` when the reply
is relayed or spoken; use JSON when something renders a UI. The raw, unflattened
answer is always available in the JSON `answer` field.

**`sources`** — render these as citations. Each carries the FAQ's own question
and answer text, so you can show the user exactly what the reply was based on.

---

## 3. Live push — `WS /v1/ws`  ⚡

Instead of collecting each answer, subscribe once and receive every message
**the moment it arrives**. This is what lets a chat screen show an inbound
question instantly rather than on the next poll.

```js
const ws = new WebSocket(location.origin.replace(/^http/, "ws") + "/v1/ws");
ws.onmessage = (e) => {
  const { event, data } = JSON.parse(e.data);
  if (event === "message.received") showIncomingQuestion(data);   // ~90ms after POST
  if (event === "message.answered") showAnswer(data);             // ~10s later
};
```

| `event` | When | `data` |
|---|---|---|
| `stream.open` | on connect | `{ listeners }` |
| `message.received` | a message arrives | `message_id`, `text`, `created_at` |
| `message.answered` | its answer is ready | the full message result (same shape as the GET) |
| `message.failed` | answering failed | `message_id`, `error` |
| `ping` | every 20s | `{}` — keeps idle proxies from closing the socket |

**Use the WebSocket, not the SSE stream, through a tunnel or CDN.** A
Server-Sent Events version exists at `GET /v1/events` with identical events, but
SSE does not survive every proxy — a Cloudflare quick tunnel buffers the
response indefinitely, so the server emits events correctly and the browser
receives nothing. A WebSocket is negotiated as a connection upgrade rather than
a long response body, so it is forwarded without buffering. Measured end to end
through a public tunnel: **89 ms** from `POST /v1/messages` to the message
appearing on screen.

A WebSocket does **not** reconnect on its own (an `EventSource` does). Reconnect
with exponential backoff, and note that events are **not replayed** — backfill
anything missed with the history endpoint below.

---

## 4. The latest answer — `GET /v1/latest`

Returns **one** message: the most recent question and its answer. No
`message_id` to track.

```bash
curl -X POST http://127.0.0.1:8000/v1/messages      -H 'Content-Type: application/json' -d '{"text":"وش طرق الدفع عندكم؟"}'

curl "http://127.0.0.1:8000/v1/latest?wait=60&format=text"
```

Supports `wait` and `format` exactly as `GET /v1/messages/{id}` does. Returns
`404` before any message has been received.

> **When not to use it.** "Latest" is inherently racy: if two questions are in
> flight at once, this returns whichever arrived last, which may not be yours.
> That is fine for one sequential conversation and wrong for concurrent ones —
> there, keep the `message_id` from the POST and use
> `GET /v1/messages/{message_id}`.

---

## 5. History — `GET /v1/history?limit=50`

The recent transcript, oldest first. This is **not** how you fetch an answer —
use `/v1/latest` for that. Its purpose is reconnection: the live socket does not
replay events, so call this once on (re)connect to fill the gap, then rely on
the socket.

```json
[ { "message_id": "msg_…", "status": "done", "text": "…", "answer": "…" } ]
```

---

## Worked example

```bash
# 1. hand over the text
ID=$(curl -s -X POST http://127.0.0.1:8000/v1/messages \
       -H 'Content-Type: application/json' \
       -d '{"text":"وش هي أكاديمية موفق؟"}' \
     | jq -r .message_id)

# 2. collect the answer (blocks until ready)
curl -s "http://127.0.0.1:8000/v1/messages/$ID?wait=30" | jq '{status, answer, cited_faq_ids}'
```

```js
async function ask(text) {
  const accepted = await fetch("/v1/messages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }).then(r => r.json());

  // Long-poll in a loop: `wait` bounds each call, the loop covers a slow answer.
  for (let attempt = 0; attempt < 5; attempt++) {
    const msg = await fetch(`/v1/messages/${accepted.message_id}?wait=30`).then(r => r.json());
    if (msg.status === "done") return msg;
    if (msg.status === "failed") throw new Error(msg.error);
  }
  throw new Error("timed out");
}
```

```python
import httpx

def ask(text: str) -> dict:
    """Send a question and return the completed message."""
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=90) as client:
        accepted = client.post(
            "/v1/messages", json={"text": text}
        ).json()
        message = client.get(
            f"/v1/messages/{accepted['message_id']}", params={"wait": 60}
        ).json()
        if message["status"] == "failed":
            raise RuntimeError(message["error"])
        return message
```

---

## Operational notes

**Concurrency.** Answers run on a bounded worker pool
(`FAQRAG_MESSAGE_WORKERS`, default 2). Extra messages queue rather than
spawning threads — the model serialises requests anyway, so a higher number
mostly converts latency into memory. Accepting a message is never blocked by
this; only answering it queues.

**Retention.** Messages are held **in memory** for
`FAQRAG_MESSAGE_TTL_SECONDS` (default 1 hour), capped at
`FAQRAG_MESSAGE_MAX_STORED` (default 1000, oldest completed evicted first). A
`message_id` collected after that returns `404`.

**Restarts drop in-flight messages.** This is a short-lived hand-off, not a
system of record — if the calling system needs durability, it should keep its
own copy of the conversation. Making it durable means replacing `MessageStore`
in `src/faqrag/messages.py` with a Redis or database implementation; nothing
else in the module depends on where state lives.

**No authentication.** These endpoints are open, and the server binds to
`127.0.0.1` by default. Before exposing them beyond localhost, put them behind a
gateway or add an API key check.

**Ordering.** Messages in one session are answered concurrently and may complete
out of order. If your UI needs strict ordering, send the next message only after
the previous one reaches `done`.
