import OpenAI from "openai";
import {
  apiKeyForClient,
  baseURL,
  openaiApiKey,
  openaiBaseURL,
  ollamaBaseURL,
} from "./config.js";

/**
 * Returns an OpenAI-compatible client.
 *
 * @param {"chat" | "embed" | "ollama"} [type="chat"]
 *   - "chat"   → active chat backend (opencode or ollama, per BACKEND env var)
 *   - "embed"  → active embed backend (openai or ollama, per BACKEND env var)
 *   - "ollama" → always Ollama (local)
 */
export function getClient(type = "chat") {
  const backend = (process.env.BACKEND ?? "opencode").toLowerCase();

  if (type === "ollama" || (type !== "embed" && backend === "ollama")) {
    return new OpenAI({
      baseURL: ollamaBaseURL,
      // Ollama doesn't require a real API key; any non-empty string works.
      apiKey: "ollama",
    });
  }

  if (type === "embed") {
    if (backend === "ollama") {
      return new OpenAI({ baseURL: ollamaBaseURL, apiKey: "ollama" });
    }
    // Default: OpenAI embeddings
    return new OpenAI({ baseURL: openaiBaseURL, apiKey: openaiApiKey });
  }

  // Default chat: opencode (or any OpenAI-compatible cloud)
  return new OpenAI({ baseURL, apiKey: apiKeyForClient });
}
