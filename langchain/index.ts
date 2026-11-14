import assert from "node:assert";
import { Document } from "@langchain/core/documents";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

const documents = [
  new Document({
    pageContent:
      "Dogs are great companions, known for their loyalty and friendliness.",
    metadata: { source: "mammal-pets-doc" },
  }),
  new Document({
    pageContent: "Cats are independent pets that often enjoy their own space.",
    metadata: { source: "mammal-pets-doc" },
  }),
];

const embeddings = new OpenAIEmbeddings({
  model: "openai/text-embedding-3-large",
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});
const vector1 = await embeddings.embedQuery(documents[0].pageContent);
const vector2 = await embeddings.embedQuery(documents[1].pageContent);

assert(vector1.length === vector2.length);

const vectorStore = new MemoryVectorStore(embeddings);
