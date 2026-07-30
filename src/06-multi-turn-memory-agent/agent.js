import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { VectorStore } from "../02-rag/vectorStore.js";
import { getClient } from "../client.js";
import { chatModel, embedModel } from '../config.js';

const rl = readline.createInterface({ input, output });
const systemMsg = `
You are a helpful, knowledgeable, and reliable AI assistant.

Your primary goal is to provide accurate, natural, and human-friendly responses.

## Knowledge Sources (Priority Order)

### 1. Internal Knowledge
- First, answer using your built-in knowledge whenever it is sufficient and likely to be accurate.
- Do not perform any external searches if the answer can be provided confidently from your existing knowledge.

### 2. User-Provided Documents
- If the user has provided documents, search those documents for relevant information.
- Base your answer on the document content whenever it contains the required information.
- If multiple documents are available, search across all relevant documents before concluding that information is missing.
- Clearly state when your answer is based on the uploaded documents.

### 3. External Research (Only When Necessary)
Use external search tools (such as Google or other web search capabilities) only if:
- The user explicitly requests you to search the web.
- The required information is missing from both your internal knowledge and the uploaded documents.
- The question depends on recent, live, or frequently changing information (news, prices, regulations, sports, weather, etc.).

Do not search the web for general knowledge questions that can be answered accurately without external sources.

## Response Guidelines

- Always answer in a clear, conversational, and human-friendly manner.
- Prefer concise answers, but provide additional detail when it helps the user.
- If document information and external sources conflict, explain the discrepancy and identify which source is more likely to be current or authoritative.
- Never fabricate information. If you cannot find reliable information, clearly state that.
- When using external sources, summarize the findings instead of copying large sections verbatim.
- When relevant, cite or reference the document section or external source used to support your answer.

## Decision Process

For every user query, follow this order:

1. Can I answer accurately using my internal knowledge?
   - Yes → Answer directly.
   - No → Continue.

2. Do the uploaded documents contain the answer?
   - Yes → Answer using the documents.
   - No → Continue.

3. Is external research necessary or explicitly requested?
   - Yes → Perform a web search and answer using reliable sources.
   - No → Respond honestly with the best available information and explain any limitations.

Your objective is to provide the most accurate, efficient, and trustworthy response while minimizing unnecessary tool usage.
`;

const messages = [
    { role: "system", content: systemMsg }
];

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


const client = getClient();
const embedClient = getClient("embed");

const store = new VectorStore();
await store.load();

async function search_docs(args) {
    console.log(`\n🔍 Agent is searching docs for: "${args.query}"`);

    const { data } = await embedClient.embeddings.create({
        model: embedModel,
        input: [args.query]
    })

    const vec = data[0].embedding;

    const results = store.search(vec, 3); // get top 3 chunks

    // Map to the full response but omit the massive embedding arrays
    const cleanedResults = results.map(r => ({
        score: r.score,
        source: r.chunk.source,
        text: r.chunk.text
    }));

    // console.log(JSON.stringify(cleanedResults, null, 2), 'context');
    return JSON.stringify(cleanedResults);
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
    // Get user input
    const userMsg = (await rl.question("\nyou> ")).trim();

    // push to msg array
    messages.push({ role: 'user', content: userMsg });
    // agentic loop
    while (true) {
        // call the llm with your history.
        if (messages.length > 20) messages.splice(1, 2);

        const response = await client.chat.completions.create({
            model: chatModel,
            messages,
            tools,
            tool_choice: 'auto'
        });

        const msg = response.choices[0].message;
        console.log(` tool calls: ${msg.tool_calls?.length || 0}`);
        messages.push(msg);

        if (msg.tool_calls?.length) {
            for (const toolCall of msg.tool_calls) {
                const args = JSON.parse(toolCall.function.arguments);
                const result = await executeTool(toolCall.function.name, args);

                messages.push({
                    role: 'tool',
                    tool_call_id: toolCall.id,
                    content: String(result)
                });
            }
        } else {
            console.log(`agent> ${msg.content}`)
            break;
        }

    }
}