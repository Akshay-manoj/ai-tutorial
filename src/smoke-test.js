import { getClient } from "./client.js";
import { backend, chatModel } from "./config.js";

const client = getClient();

console.log(`Backend: ${backend}`);
console.log(`Model:   ${chatModel}`);

const response = await client.chat.completions.create({
  model: chatModel,
  messages: [{ role: "user", content: "Reply with exactly: smoke ok" }],
  temperature: 0,
});

console.log(response.choices[0]?.message?.content ?? "(no content)");
