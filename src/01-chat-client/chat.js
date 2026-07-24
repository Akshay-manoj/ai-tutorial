import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { getClient } from "../client.js";
import { chatModel } from "../config.js";

const client = getClient();
const messages = [
    { role: "system", content: "You are a helpful assistant." },
];

const rl = readline.createInterface({ input, output });
console.log("Chat started. Type 'exit' to quit.\n");

while (true) {
    const userText = (await rl.question("you> ")).trim();
    if (!userText) continue;
    if (userText.toLowerCase() === "exit") break;

    messages.push({ role: "user", content: userText });

    const stream = await client.chat.completions.create({
        model: chatModel,
        messages,
        stream: true,
    });

    process.stdout.write("assistant> ");
    let reply = "";
    for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content ?? "";
        reply += delta;
        process.stdout.write(delta);
    }
    process.stdout.write("\n\n");

    messages.push({ role: "assistant", content: reply });
}

rl.close();
