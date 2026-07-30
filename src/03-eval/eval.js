import questions from './test_questions.json' with { type: "json" };

import { getClient } from '../client.js'
import { embedModel } from '../config.js'
import { VectorStore } from "../02-rag/vectorStore.js";

const store = new VectorStore();
await store.load();

const embedClient = getClient("embed");

async function embedQuery(text) {
    const { data } = await embedClient.embeddings.create({
        model: embedModel,
        input: [text],
    });
    return data[0].embedding;
}

let hits = 0;
for (const { question, expected_source } of questions) {
    const vec = await embedQuery(question);

    const results = store.search(vec, 5);
    const source = results.map((e) => e.chunk.source);

    if (source.includes(expected_source)) hits++;
}

console.log(`Hit rate: ${hits}/${questions.length}`)