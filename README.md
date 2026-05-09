# MediChat: AI-Powered Medical Chatbot🏥
MediChat is a **context-aware medical chatbot** built using **Flask, LangChain, and Retrieval-Augmented Generation (RAG)**. It leverages a vector database and large language models to provide more accurate and relevant responses to user queries, while maintaining **conversation memory** for a more natural chat experience.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://medical-chatbot-1-zqya.onrender.com/)

---

## Features

- **RAG Pipeline** - Retrieves top-3 semantically similar documents from Pinecone before generating answers
- **Semantic Embeddings** - Uses HuggingFace `sentence-transformers/all-MiniLM-L6-v2` for efficient vector representation and similarity search
- **Conversational Memory** - Each user session maintains independent chat history across multiple turns
- **History-Aware Retrieval** - Rephrases follow-up questions into standalone queries using prior context
- **HuggingFace LLM** - Powered by `openai/gpt-oss-120b` via HuggingFace Inference Endpoint

---

## Tech Stack
![Flask](https://img.shields.io/badge/Framework-Flask-black?logo=flask)
![HuggingFace](https://img.shields.io/badge/LLM-HuggingFace-orange?logo=huggingface)
![SentenceTransformers](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-blue)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-0066FF)
![LangChain](https://img.shields.io/badge/RAG-LangChain-green)
![HTML5](https://img.shields.io/badge/Frontend-HTML%2FCSS-E34F26?logo=html5)

---
## Installation and Requirements

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
