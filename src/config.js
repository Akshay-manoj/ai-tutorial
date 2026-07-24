import "dotenv/config";

const apiKey = process.env.OPENCODE_API_KEY?.trim();
const hasOpenCode = Boolean(apiKey) && apiKey !== "sk-...";

export const useOllama = !hasOpenCode;
export const backend = useOllama ? "ollama" : "opencode";

export const chatModel = useOllama
  ? process.env.OLLAMA_CHAT_MODEL
  : process.env.OPENCODE_CHAT_MODEL;

// Zen has no embedding endpoint — always use Ollama for embeds (Phase 2)
export const embedModel = process.env.OLLAMA_EMBED_MODEL;

export const baseURL = useOllama
  ? process.env.OLLAMA_BASE_URL
  : process.env.OPENCODE_BASE_URL;

export const apiKeyForClient = useOllama ? "ollama" : apiKey;
