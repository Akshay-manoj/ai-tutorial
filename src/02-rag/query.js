import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { getClient } from "../client.js";
import { backend, chatModel, embedModel, openaiApiKey } from "../config.js";
import { VectorStore } from "./vectorStore.js";

// ---------------------------------------------------------------------------
// CLI flag:  --top-k <n>
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
function getNumericArg(flag, defaultValue) {
  const idx = args.indexOf(flag);
  if (idx !== -1) {
    const val = Number(args[idx + 1]);
    return Number.isFinite(val) && val > 0 ? val : defaultValue;
  }
  return defaultValue;
}
const TOP_K = getNumericArg("--top-k", 3);

// Collect any non-flag tokens as the one-shot question
const questionArgs = args.filter((a, i) => !a.startsWith("--") && args[i - 1] !== "--top-k");
const oneShotQuestion = questionArgs.join(" ").trim();

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------
function hasConfiguredValue(value) {
  return Boolean(value) && value !== "sk-...";
}

if (backend !== "ollama" && !hasConfiguredValue(openaiApiKey)) {
  throw new Error(
    "Missing OPENAI_API_KEY in your .env. Set BACKEND=ollama to use local Ollama instead."
  );
}

// ---------------------------------------------------------------------------
// Load vector store
// ---------------------------------------------------------------------------
const store = new VectorStore();
await store.load();

if (store.size === 0) {
  console.error(
    "Vector store is empty. Run `npm run ingest` first to populate it."
  );
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Clients (single factory, type-dispatched)
// ---------------------------------------------------------------------------
const embedClient = getClient("embed");
const chatClient = getClient("chat");

console.log(`Backend: ${backend} | embed: ${embedModel} | chat: ${chatModel}\n`);

// ---------------------------------------------------------------------------
// Core helpers
// ---------------------------------------------------------------------------
async function embedQuery(text) {
  const { data } = await embedClient.embeddings.create({
    model: embedModel,
    input: [text],
  });
  return data[0].embedding;
}

async function answerQuestion(question) {
  output.write(`\nSearching for: "${question}"\n`);

  const queryVec = await embedQuery(question);
  const results = store.search(queryVec, TOP_K);


  // ── Build prompt ─────────────────────────────────────────────────────────
  const context = results
    .map(({ chunk }) => `[source: ${chunk.source}]\n${chunk.text}`)
    .join("\n\n---\n\n");

  const messages = [
    {
      role: "system",
      content:
        "You are a precise technical assistant. " +
        "Answer ONLY from the provided context — do not add information from your training data. " +
        "At the end of your answer, cite your sources using the format: [source: filename].",
    },
    {
      role: "user",
      content: `Context:\n${context}\n\nQuestion: ${question}`,
    },
  ];

  // ── Stream answer ─────────────────────────────────────────────────────────
  const stream = await chatClient.chat.completions.create({
    model: chatModel,
    messages,
    stream: true,
    temperature: 0,
  });

  output.write("answer> ");
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (delta) output.write(delta);
  }
  output.write("\n");
}

// ---------------------------------------------------------------------------
// One-shot mode (npm run query -- "question here")
// or interactive mode
// ---------------------------------------------------------------------------
if (oneShotQuestion) {
  await answerQuestion(oneShotQuestion);
} else {
  const rl = readline.createInterface({ input, output });
  output.write(
    `RAG query — backend=${backend} top_k=${TOP_K} — Type a question or "exit" to quit.\n\n`
  );

  try {
    while (true) {
      const question = (await rl.question("question> ")).trim();
      if (!question) continue;
      if (question.toLowerCase() === "exit") break;

      try {
        await answerQuestion(question);
      } catch (err) {
        console.error("\nRequest failed:", err.message);
      }
    }
  } finally {
    rl.close();
  }
}
