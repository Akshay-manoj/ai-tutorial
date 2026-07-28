import fs from "node:fs";
import path from "node:path";
import { getClient } from "../client.js";
import { backend, openaiApiKey, embedModel } from "../config.js";
import { chunkText } from "./chunker.js";
import { VectorStore } from "./vectorStore.js";

// ---------------------------------------------------------------------------
// CLI flags:  --chunk-size <n>   --overlap <n>
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
function getNumericArg(flag, defaultValue) {
  const idx = args.indexOf(flag);
  if (idx !== -1) {
    const val = Number(args[idx + 1]);
    if (!Number.isFinite(val) || val <= 0) {
      throw new Error(`Invalid value for ${flag}: ${args[idx + 1]}`);
    }
    return val;
  }
  return defaultValue;
}
const chunkSize = getNumericArg("--chunk-size", 500);
const overlap = getNumericArg("--overlap", 50);

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------
const SUPPORTED_EXTENSIONS = new Set([".md", ".txt"]);
const docsFolder = path.join(process.cwd(), "sample_docs");

function hasConfiguredValue(value) {
  return Boolean(value) && value !== "sk-...";
}

if (backend !== "ollama" && !hasConfiguredValue(openaiApiKey)) {
  throw new Error(
    "Missing OPENAI_API_KEY in your .env. Set BACKEND=ollama to use local Ollama embeddings instead."
  );
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------
function getDocumentFiles(folderPath) {
  return fs
    .readdirSync(folderPath)
    .filter((file) => SUPPORTED_EXTENSIONS.has(path.extname(file).toLowerCase()));
}

function readDocument(filename) {
  return fs.readFileSync(path.join(docsFolder, filename), "utf-8");
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------
const files = getDocumentFiles(docsFolder);

if (!files.length) {
  console.log("No supported documents found in sample_docs/.");
  process.exit(1);
}

const client = getClient("embed");
const store = new VectorStore();

console.log(`Backend: ${backend}`);
console.log(`Embed model: ${embedModel}`);
console.log(`Ingesting ${files.length} file(s)`);
console.log(`  chunkSize = ${chunkSize}, overlap = ${overlap}\n`);

for (const file of files) {
  console.log(`→ ${file}`);

  const content = readDocument(file);
  const chunks = chunkText(content, { chunkSize, overlap });

  if (!chunks.length) {
    console.log("  (empty — skipping)");
    continue;
  }

  // Embed all chunks for this file in one batched API call
  const { data } = await client.embeddings.create({
    model: embedModel,
    input: chunks,
  });

  const entries = data.map((d, i) => ({
    id: `${file}::${i}`,
    source: file,
    text: chunks[i],
    embedding: d.embedding,
  }));

  store.add(entries);
  console.log(`  ${entries.length} chunks embedded`);
}

await store.save();
console.log(`\nDone. ${store.size} total entries in the vector store.`);
