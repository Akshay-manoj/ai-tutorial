import readline from "node:readline/promises"
import { stdin as input, stdout as output } from "node:process";
import { getClient } from "../client.js";
import { chatModel, embedModel } from "../config.js";
import { VectorStore } from "../02-rag/vectorStore.js";
import { callbackify } from "node:util";

const rl = readline.createInterface({ input, output });

const userQuestion = process.argv.slice(2).join(" ");
const messages = [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: userQuestion }
];

const client = getClient();
const embedClient = getClient("embed");
const store = new VectorStore();

const tools = [
    {
        type: "function",
        function: {
            name: "search_docs",
            description: "Search the knowledge base for relevant passages",
            parameters: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "calculator",
            description: "Evaluate a math expression",
            parameters: {
                type: "object",
                properties: { expression: { type: "string" } },
                required: ["expression"],
            },
        },
    },
];

async function search_docs(args) {
    console.log(`\n🔍 Agent is searching docs for: "${args.query}"`);

    const { data } = await embedClient.embeddings.create({
        model: embedModel,
        input: [args.query]
    })

    const vec = data[0].embedding;

    const results = store.search(vec, 3); // get top 3 chunks

    const context = results.map(r => r.text).join('\n');
    return context;
}

async function calculator(expression) {
    return eval(expression);
}

async function executeTool(name, args) {
    const tools = {
        search_docs,
        calculator,
    };
    return tools[name](args);
}

while (true) {
    const response = await client.chat.completions.create({
        model: chatModel,
        messages,
        tools
    });

    const msg = response.choices[0].message;
    messages.push(msg);

    if (msg.tool_calls?.length) {
        for (const toolCall of msg.tool_calls) {
            const args = JSON.parse(toolCall.function.arguments);
            const result = await executeTool(toolCall.function.name, args);
            messages.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: JSON.stringify(result),
            });
        }
        continue;
    }

    console.log(msg.content);
    break;
}