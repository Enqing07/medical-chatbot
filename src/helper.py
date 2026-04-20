from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf_files(filepath):
    loader = DirectoryLoader(
        filepath,
        glob='**/*.pdf',
        show_progress=True,
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: list[Document]) -> list[Document]:
    minimal_docs: list[Document] = []
    for doc in docs:
        src = doc.metadata.get('source')
        minimal_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata={'source': src}
            )
        )
    return minimal_docs

def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunk = text_splitter.split_documents(minimal_docs)
    return text_chunk

def download_hugging_face_embeddings():
    """
    Load and return SentenceTransformer embedding model.
    """
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return model
