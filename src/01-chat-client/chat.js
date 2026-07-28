import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { getClient } from "../client.js";
import { chatModel } from "../config.js";

const EXIT_COMMAND = "exit";
const SYSTEM_MESSAGE = { role: "system", content: "You are a helpful assistant." };

const client = getClient();
const messages = [SYSTEM_MESSAGE];
const rl = readline.createInterface({ input, output });

async function askUser() {
  return (await rl.question("you> ")).trim();
}

async function streamAssistantReply(history) {
  const stream = await client.chat.completions.create({
    model: chatModel,
    messages: history,
    stream: true,
  });

  output.write("assistant> ");

  let reply = "";
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (!delta) continue;

    reply += delta;
    output.write(delta);
  }

  output.write("\n\n");
  return reply;
}

console.log(`Chat started. Type '${EXIT_COMMAND}' to quit.\n`);

try {
  while (true) {
    const userText = await askUser();
    if (!userText) continue;
    if (userText.toLowerCase() === EXIT_COMMAND) break;

    messages.push({ role: "user", content: userText });

    try {
      const reply = await streamAssistantReply(messages);
      messages.push({ role: "assistant", content: reply });
    } catch (error) {
      messages.pop();
      console.error("\nRequest failed:", error.message);
    }
  }
} finally {
  rl.close();
}
