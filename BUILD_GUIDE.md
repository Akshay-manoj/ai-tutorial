# LLM + RAG Study — Phase-by-Phase Build Guide (Node.js)

Build this yourself, one phase at a time. Each phase has a clear goal, folder layout, and a "done when" checkpoint.

**Stack:** Node.js 18+, `openai` SDK, `dotenv`, plain JSON vector store (no framework).

---

## Overview

| Phase | Project | Time | Prerequisite |
|-------|---------|------|--------------|
| 0 | Environment setup | ~30 min | — |
| 1 | Minimal chat client | ~half day | Phase 0 |
| 2 | From-scratch RAG | ~1–2 days | Phase 1 |
| 3 | RAG evaluation | ~1 day | Phase 2 |
| 4 | Tool-calling agent | ~1–2 days | Phase 2 |
| 5 | Local-only stack (optional) | ~1 day | Phase 2 |

---

## Phase 0 — Environment Setup

**Goal:** One working way to call an LLM.

**Pick one backend:**

- **Ollama** (local, free): install Ollama, pull `llama3.2` and `nomic-embed-text`
- **OpenAI** (cloud): get an API key

**Create a minimal project:**

```
llm-rag-study/
├── .env                 # API keys / model names
├── .env.example
├── package.json
├── src/
│   ├── config.js        # read env vars, pick OpenAI vs Ollama
│   └── client.js        # thin OpenAI SDK wrapper
└── sample_docs/         # (Phase 2)
```

**Initialize:**

```bash
mkdir llm-rag-study && cd llm-rag-study
npm init -y
npm install openai dotenv
```

**`.env.example`:**

```env
# OpenAI (cloud)
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# Ollama (local — used when OPENAI_API_KEY is unset)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

**`src/config.js` sketch:**

```js
import "dotenv/config";

const hasOpenAI = process.env.OPENAI_API_KEY && !process.env.OPENAI_API_KEY.startsWith("sk-...");
export const useOllama = !hasOpenAI;
export const chatModel = useOllama ? process.env.OLLAMA_CHAT_MODEL : process.env.OPENAI_CHAT_MODEL;
export const embedModel = useOllama ? process.env.OLLAMA_EMBED_MODEL : process.env.OPENAI_EMBED_MODEL;
export const baseURL = useOllama ? process.env.OLLAMA_BASE_URL : undefined;
```

**`src/client.js` sketch:**

```js
import OpenAI from "openai";
import { baseURL } from "./config.js";

export function getClient() {
  return new OpenAI({
    baseURL,
    apiKey: process.env.OPENAI_API_KEY ?? "ollama",
  });
}
```

Add to `package.json`:

```json
{ "type": "module" }
```

**Done when:** A one-liner script calls the model and prints a response:

```bash
node -e "import('./src/client.js').then(async ({getClient}) => { ... })"
```

Or a tiny `src/smoke-test.js` you run with `node src/smoke-test.js`.

---

## Phase 1 — Minimal Chat Client

**Goal:** Understand how chat APIs actually work.

**Build:** `src/01-chat-client/chat.js` — a CLI loop (use `readline`).

**Features to implement (in order):**

1. **Single-turn call** — one user message → one assistant reply
2. **System prompt** — add a `system` message that sets behavior
3. **Multi-turn history** — keep a `messages[]` array; push user + assistant each turn
4. **Streaming** — iterate `for await (const chunk of stream)` and print deltas
5. **Temperature flag** — parse `--temperature 0.7` from `process.argv`

**Example streaming call:**

```js
import { getClient } from "../client.js";
import { chatModel } from "../config.js";

const client = getClient();
const stream = await client.chat.completions.create({
  model: chatModel,
  messages,
  temperature: 0.7,
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

**Add npm script:**

```json
"scripts": {
  "chat": "node src/01-chat-client/chat.js"
}
```

**Concepts to study as you build:**

- Roles: `system`, `user`, `assistant`
- APIs are **stateless** — you resend full history every request
- Low temperature = more deterministic (good for facts later in RAG)

**Done when:** You can have a 5-turn conversation with streaming, and explain what each message role does.

**Stretch:** Sum `message.content.length` across turns to feel context growth.

---

## Phase 2 — From-Scratch RAG (Core)

**Goal:** Build the full retrieval pipeline by hand — no LangChain, no heavy frameworks.

**Target layout:**

```
src/
├── 02-rag/
│   ├── chunker.js       # split text into chunks
│   ├── vectorStore.js   # save/load embeddings + search
│   ├── ingest.js        # docs → chunks → embeddings → store
│   └── query.js         # question → retrieve → prompt → answer
sample_docs/             # 3–5 .md or .txt files you write
data/
└── vector_store.json    # generated
```

**No extra deps required** — use `fs/promises`, `path`, and a hand-rolled cosine similarity. Optional: `glob` for file discovery (`npm install glob`).

### Step 2a — Chunking (`chunker.js`)

```js
export function chunkText(text, { chunkSize = 500, overlap = 50 } = {}) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + chunkSize).trim());
    if (start + chunkSize >= text.length) break;
    start += chunkSize - overlap;
  }
  return chunks.filter(Boolean);
}
```

**Experiment:** try 200 vs 500 vs 1000 and notice what breaks later.

### Step 2b — Embeddings (`ingest.js`)

For each chunk:

1. Call embedding API (`text-embedding-3-small` or Ollama `nomic-embed-text`)
2. Store: `{ id, source, text, embedding: number[] }`

```js
const client = getClient();
const { data } = await client.embeddings.create({
  model: embedModel,
  input: texts, // batch an array of strings
});
const vectors = data.map((d) => d.embedding);
```

### Step 2c — Vector Store (`vectorStore.js`)

Start simple:

- `save()` → `fs.writeFile("data/vector_store.json", ...)`
- `load()` on startup if file exists
- `search(queryEmbedding, topK)` using cosine similarity:

```js
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

No Chroma/FAISS yet — raw JSON teaches you what vector DBs actually do.

### Step 2d — Ingest Pipeline (`ingest.js`)

```
for each .md/.txt in sample_docs/
  read file (fs.readFile)
  chunk it
  embed each chunk (batch if you want)
  add to store
await store.save()
```

**Run:**

```bash
node src/02-rag/ingest.js
node src/02-rag/ingest.js --chunk-size 500 --overlap 50
```

Parse flags with `process.argv` or add `minimist` (`npm install minimist`).

### Step 2e — Query Pipeline (`query.js`)

```
1. embed user question
2. store.search(queryVec, 3) → top chunks
3. build messages:
   system: "Answer ONLY from context. Cite [source: filename]."
   user:   "Context:\n...\n\nQuestion: ..."
4. stream chat completion
5. log retrieved chunks + scores before the answer
```

**Pipeline diagram:**

```mermaid
flowchart LR
  docs[Documents] --> chunk[Chunk]
  chunk --> embed[Embed]
  embed --> store[VectorStore]
  query[UserQuery] --> qEmbed[EmbedQuery]
  qEmbed --> retrieve[TopK]
  store --> retrieve
  retrieve --> prompt[BuildPrompt]
  prompt --> llm[LLM]
  llm --> answer[AnswerWithCitations]
```

**npm scripts:**

```json
"scripts": {
  "ingest": "node src/02-rag/ingest.js",
  "query": "node src/02-rag/query.js"
}
```

**Done when:** You ask "What is chunk overlap?" and get an answer grounded in your docs with source citations.

**Write 3 sample docs yourself** (best for learning):

- `rag_basics.md` — what RAG is, the loop, chunking
- `llm_chat_basics.md` — roles, temperature, streaming
- `vector_search.md` — cosine similarity, top-k, failure modes

---

## Phase 3 — RAG Evaluation Mini-Lab

**Goal:** Measure whether retrieval actually works — not just "it looks fine on one demo."

**Build:** `src/03-eval/eval.js` + `src/03-eval/test_questions.json`

**Create 10 Q&A pairs:**

```json
[
  {
    "question": "What is chunk overlap?",
    "expected_source": "vector_search.md"
  }
]
```

**Score retrieval hit rate:**

```js
import questions from "./test_questions.json" assert { type: "json" };
// or: JSON.parse(await fs.readFile(...))

let hits = 0;
for (const { question, expected_source } of questions) {
  const vec = await embedQuery(question);
  const results = store.search(vec, topK);
  const sources = results.map((r) => r.chunk.source);
  if (sources.includes(expected_source)) hits++;
}
console.log(`Hit rate: ${hits}/${questions.length}`);
```

**Target:** 8–10/10 with default settings.

**Then manually check answer faithfulness:**

- Run `node src/02-rag/query.js "..."` for each question
- Does the answer match the retrieved chunks?
- Does it hallucinate when context is missing?

**Experiments (note what breaks):**

| Change | Expected effect |
|--------|-----------------|
| `chunk_size=100` | Better precision, worse broad questions |
| `overlap=0` | Misses boundary-spanning content |
| `top_k=1` | Cheaper but more misses |
| High temperature | Less faithful to context |

**Done when:** You have a hit-rate number and a short note on what broke when you tuned knobs.

---

## Phase 4 — Tool-Calling Agent

**Goal:** LLM decides *when* to retrieve instead of always running RAG.

**Build:** `src/04-agent/agent.js`

**Tools to expose:**

1. `search_docs` — wraps your Phase 2 retriever (`{ query: string }`)
2. `calculator` — safe math (`{ expression: string }` — avoid raw `eval`; use a tiny parser or `mathjs`)

**Define tools for the OpenAI SDK:**

```js
const tools = [
  {
    type: "function",
    function: {
      name: "search_docs",
      description: "Search the knowledge base for relevant passages",
      parameters: {
        type: "object",
        properties: { query: { type: "string" } },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "calculator",
      description: "Evaluate a math expression",
      parameters: {
        type: "object",
        properties: { expression: { type: "string" } },
        required: ["expression"],
      },
    },
  },
];
```

**Agent loop:**

```js
const messages = [{ role: "system", content: "..." }];

while (true) {
  const response = await client.chat.completions.create({
    model: chatModel,
    messages,
    tools,
  });

  const msg = response.choices[0].message;
  messages.push(msg);

  if (msg.tool_calls?.length) {
    for (const call of msg.tool_calls) {
      const args = JSON.parse(call.function.arguments);
      const result = await runTool(call.function.name, args);
      messages.push({
        role: "tool",
        tool_call_id: call.id,
        content: JSON.stringify(result),
      });
    }
    continue;
  }

  console.log(msg.content);
  break;
}
```

**Failure modes to observe:**

- Infinite tool loops (add a max-iterations guard)
- Wrong tool chosen
- Bad tool arguments (malformed JSON)
- Answering from memory instead of calling search

**Done when:** Agent calls `search_docs` for doc questions and `calculator` for math, in one session.

---

## Phase 5 — Local-Only Stack (Optional)

**Goal:** Same as Phase 2, fully offline.

**Stack:**

- Ollama: `llama3.2` (chat) + `nomic-embed-text` (embeddings)
- Same JSON vector store (or swap to a real DB later)

**Ollama setup:**

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Leave `OPENAI_API_KEY` unset — `config.js` routes to `http://localhost:11434/v1`.

**Compare and note:**

- Latency on your machine
- Answer quality vs cloud
- When local is "good enough" vs when you need GPT-4 class models

**Done when:** Full ingest + query works with no API key and no network.

---

## Suggested Build Order (Weekend Plan)

| When | Phase |
|------|-------|
| Day 1 morning | Phase 0 + Phase 1 |
| Day 1 afternoon – Day 2 | Phase 2 (chunk → embed → store → query) |
| Day 3 | Phase 3 (eval + experiments) |
| Day 4–5 | Phase 4 (agent) |
| Optional | Phase 5 if you care about local/privacy |

---

## Final `package.json` Scripts (reference)

```json
{
  "type": "module",
  "scripts": {
    "chat": "node src/01-chat-client/chat.js",
    "ingest": "node src/02-rag/ingest.js",
    "query": "node src/02-rag/query.js",
    "eval": "node src/03-eval/eval.js",
    "agent": "node src/04-agent/agent.js"
  },
  "dependencies": {
    "dotenv": "^16.0.0",
    "openai": "^4.0.0"
  }
}
```

Optional dev deps: `minimist` (CLI flags), `glob` (file globbing), `mathjs` (safe calculator).

---

## What to Skip for Now

- Fine-tuning / training models
- LangChain.js / LlamaIndex (build raw first, then compare)
- Production concerns: auth, scaling, monitoring
- PDF parsing (add later with `pdf-parse` once markdown flow works)

---

## Self-Check Questions

After each phase, you should be able to explain without looking at code:

1. **Phase 1:** Why does the API need the full message history every call?
2. **Phase 2:** What happens if chunk size is too large? Too small?
3. **Phase 3:** What's the difference between retrieval hit rate and answer faithfulness?
4. **Phase 4:** When should the agent call a tool vs answer directly?
