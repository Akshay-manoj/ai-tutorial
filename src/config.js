import "dotenv/config";

// ---------------------------------------------------------------------------
// Backend selector
// ---------------------------------------------------------------------------
// Set BACKEND=ollama to use local Ollama for both chat and embeddings.
// Set BACKEND=opencode (default) to use OpenCode Zen for chat + OpenAI for embeddings.
export const backend = (process.env.BACKEND ?? "opencode").toLowerCase();

// ---------------------------------------------------------------------------
// OpenCode Zen — chat/model calls
// ---------------------------------------------------------------------------
export const apiKeyForClient = process.env.OPENCODE_API_KEY?.trim();
export const baseURL = process.env.OPENCODE_BASE_URL;
export const chatModel =
  backend === "ollama"
    ? (process.env.OLLAMA_MODEL ?? "llama3.2")
    : process.env.OPENCODE_CHAT_MODEL;

// ---------------------------------------------------------------------------
// OpenAI — embeddings (used when BACKEND != ollama)
// ---------------------------------------------------------------------------
export const openaiApiKey = process.env.OPENAI_API_KEY?.trim();
export const openaiEmbedModel =
  process.env.OPENAI_EMBED_MODEL ?? "text-embedding-3-small";
export const openaiBaseURL = process.env.OPENAI_BASE_URL;

// ---------------------------------------------------------------------------
// Ollama (local) — OpenAI-compatible endpoint
// ---------------------------------------------------------------------------
export const ollamaBaseURL =
  process.env.OLLAMA_BASE_URL ?? "http://localhost:11434/v1";
export const ollamaModel = process.env.OLLAMA_MODEL ?? "llama3.2";
export const ollamaEmbedModel =
  process.env.OLLAMA_EMBED_MODEL ?? "nomic-embed-text";
// ---------------------------------------------------------------------------
// Active embed model — resolved by backend
// ---------------------------------------------------------------------------
export const embedModel =
  backend === "ollama" ? ollamaEmbedModel : openaiEmbedModel;
