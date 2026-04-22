# 🏥 MediChat: AI-Powered Medical Chatbot
MediChat is a **context-aware medical chatbot** built using **Flask, LangChain, and Retrieval-Augmented Generation (RAG)**. It leverages a vector database and large language models to provide more accurate and relevant responses to user queries, while maintaining **conversation memory** for a more natural chat experience.

---

## 🚀 Features

- **RAG Pipeline** - Retrieves top-3 semantically similar documents from Pinecone before generating answers
- **Conversational Memory** - Each user session maintains independent chat history across multiple turns
- **History-Aware Retrieval** - Rephrases follow-up questions into standalone queries using prior context
- **HuggingFace LLM** - Powered by `openai/gpt-oss-120b` via HuggingFace Inference Endpoint

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Framework | Flask |
| LLM | HuggingFace `openai/gpt-oss-120b` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2`|
| Vector DB | Pinecone |
| RAG Orchestration | LangChain |
| Frontend | HTML/CSS |

---
## ⚙️ Installation and Requirements

1. In your terminal, clone the repository:
```bash
https://github.com/Enqing07/medical-chatbot.git
cd medical-chatbot
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Set up environment variables. Create a `.env` file in the root directory:
```bash
PINECONE_API_KEY = "your_pinecone_api_key"
HUGGINGFACEHUB_API_TOKEN = "your_huggingface_api_token"
```
3. Run the app:
```bash
python app.py
```
