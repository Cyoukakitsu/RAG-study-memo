import { readFileSync } from "node:fs";
import { Document } from "@langchain/core/documents";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

// Below is a minimal helper for demonstration purposes.
async function loadPdfPages(filePath: string): Promise<Document[]> {
  const parser = new PDFParse({
    data: new Uint8Array(readFileSync(filePath)),
  });
  try {
    const { pages } = await parser.getText();
    return pages.map(
      (page) =>
        new Document({
          pageContent: page.text,
          metadata: { source: filePath, page: page.num - 1 },
        })
    );
  } finally {
    await parser.destroy();
  }
}

const filePath = "./data/nke-10k-2023.pdf";
const docs = await loadPdfPages(filePath);
console.log(docs.length);

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);

console.log(allSplits.length);

const embeddings = new OpenAIEmbeddings({
  model: "nvidia/nemotron-3.5-embed-8b:free",
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});
const vectorStore = new MemoryVectorStore(embeddings);

await vectorStore.addDocuments(allSplits);
console.log(`Added ${vectorStore.memoryVectors.length} chunks to vector store`);

const results1 = await vectorStore.similaritySearch(
  "When was Nike incorporated?"
);

console.log(results1[0]);

const llm = new ChatOpenAI({
  model: "nvidia/nemotron-3-super-120b-a12b:free",
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});

const question = "When was Nike incorporated?";
const context = results1.map((d) => d.pageContent).join("\n\n");
const answer = await llm.invoke(
  `请只根据下面的资料，用中文回答问题。\n\n资料：\n${context}\n\n问题：${question}`
);

console.log(answer.content);
