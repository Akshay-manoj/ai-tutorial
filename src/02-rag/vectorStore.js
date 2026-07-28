import fs from "node:fs/promises";
import path from "node:path";

const STORE_PATH = path.join(process.cwd(), "data", "vector_store.json");

/**
 * Compute cosine similarity between two equal-length vectors.
 * Returns a value in [-1, 1]; higher = more similar.
 */
function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * A simple flat JSON vector store.
 *
 * Each entry has the shape:
 *   { id: string, source: string, text: string, embedding: number[] }
 */
export class VectorStore {
  constructor() {
    /** @type {Array<{id:string, source:string, text:string, embedding:number[]}>} */
    this.entries = [];
  }

  /** Append new entries (from a single ingestion run). */
  add(entries) {
    this.entries.push(...entries);
  }

  /**
   * Return the top-K entries most similar to queryEmbedding.
   * @param {number[]} queryEmbedding
   * @param {number} topK
   * @returns {Array<{chunk: object, score: number}>}
   */
  search(queryEmbedding, topK = 3) {
    return this.entries
      .map((entry) => ({
        chunk: entry,
        score: cosineSimilarity(queryEmbedding, entry.embedding),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /** Persist all entries to data/vector_store.json. */
  async save() {
    await fs.mkdir(path.dirname(STORE_PATH), { recursive: true });
    await fs.writeFile(STORE_PATH, JSON.stringify(this.entries, null, 2), "utf-8");
    console.log(`Saved ${this.entries.length} entries → ${STORE_PATH}`);
  }

  /**
   * Load entries from disk (silently starts empty if file does not exist).
   * @returns {Promise<void>}
   */
  async load() {
    try {
      const raw = await fs.readFile(STORE_PATH, "utf-8");
      this.entries = JSON.parse(raw);
      console.log(`Loaded ${this.entries.length} entries from ${STORE_PATH}`);
    } catch (err) {
      if (err.code !== "ENOENT") throw err;
      this.entries = [];
    }
  }

  /** Number of stored entries. */
  get size() {
    return this.entries.length;
  }
}
