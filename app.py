from flask import Flask, render_template, request, session
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_core.runnables.history import RunnableWithMessageHistory    
from langchain_community.chat_message_histories import ChatMessageHistory  
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from src.prompts import *
import os
import uuid

app = Flask(__name__)

app.secret_key = os.urandom(24)  # Required for session (tracking users)

# Load environment variables
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


print("Step 1: Downloading embeddings...")
embeddings = download_hugging_face_embeddings()
print("Step 2: Embeddings loaded. Connecting to Pinecone...")

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
print("Step 3: Pinecone connected. Setting up LLM...")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    max_new_tokens=256,
    do_sample=False,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)
chatModel = ChatHuggingFace(llm=llm)
print("Step 4: LLM ready. App startup complete.")


# Prompt to rephrase the question using chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, "
     "rephrase it into a standalone question. "
     "Do NOT answer it. If no rephrasing is needed, return it as-is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Retriever with previous conversation context
history_aware_retriever = create_history_aware_retriever(
    chatModel, retriever, contextualize_q_prompt
)

# Main prompt, includes chat_history
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),           # your existing system prompt
    MessagesPlaceholder("chat_history"), # injected conversation turns
    ("human", "{input}"),
])


question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# In-memory store mapping session_id -> ChatMessageHistory
chat_histories: dict[str, ChatMessageHistory] = {}  

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_histories: # new user create new memory
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

@app.route("/")
def index():
    # Assign a unique session ID to each user
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    session_id = session.get("session_id", str(uuid.uuid4()))
    print(f"[{session_id}] User: {msg}")

    response = conversational_rag_chain.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}},  # ties history to user
    )

    answer = response["answer"]
    print(f"[{session_id}] Bot: {answer}")
    return str(answer)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=False)