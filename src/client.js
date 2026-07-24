import OpenAI from "openai";
import { apiKeyForClient, baseURL } from "./config.js";

export function getClient() {
  return new OpenAI({
    baseURL,
    apiKey: apiKeyForClient,
  });
}
