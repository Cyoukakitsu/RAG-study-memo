# RAG Study Memo

Study notes and practice code for RAG (Retrieval-Augmented Generation), organized by course chapter.

## Contents

| Folder | Topic |
|---|---|
| `01_rag_basic/` | RAG basics: document loading, chunking, embedding, retrieval, Q&A |

More chapters (`02_...`, `03_...`) will be added over time.

## Quick Start (using 01 as an example)

```bash
cd 01_rag_basic
python3 -m venv basicragenv          # create a virtual environment
source basicragenv/bin/activate      # activate it
pip install -r requirements.txt      # install dependencies
```

Create a `.env` file in `01_rag_basic/` and add your own API keys (it is listed in `.gitignore`, so it won't be uploaded):

```
GROQ_API_KEY=your_key
JINA_API_KEY=your_key
```

Then open `rag.ipynb` in VS Code, select the `basicragenv` kernel in the top-right corner, and run the cells one by one.

## Tech Stack

Python 3.12 · LangChain · Groq (LLM) · Jina (embeddings) · FAISS (vector store) · Streamlit · Jupyter
