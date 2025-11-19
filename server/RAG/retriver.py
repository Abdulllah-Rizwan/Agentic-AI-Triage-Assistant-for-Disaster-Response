import os
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing import List

from . import constants
from . import embedder
from . import loader

# Load environment variables
load_dotenv()


class CorpusRetriever:
    """
    Main RAG class - handles vector store creation and retrieval.
    
    Usage:
        retriever = CorpusRetriever()
        results = retriever.search("What is diabetes?")
    """
    
    def __init__(self, rebuild: bool = False):
        """
        Initialize the retriever.
        
        Args:
            rebuild: If True, deletes and rebuilds vector store
        """
        vs_path = str(constants.VECTOR_STORE_DIR)
        
        # Rebuild if requested
        if rebuild and os.path.exists(vs_path):
            print(f"🔄 Rebuilding: Deleting {vs_path}...")
            shutil.rmtree(vs_path)
        
        # Get embedder
        gemini_embedder = embedder.get_gemini_embedder()
        
        # Load or build vector store
        if os.path.exists(vs_path):
            print(f"📂 Loading existing vector store from {vs_path}...")
            self.vector_store = Chroma(
                persist_directory=vs_path,
                embedding_function=gemini_embedder
            )
            print("✓ Vector store loaded")
        else:
            print("🔨 Building new vector store...")
            self.vector_store = self._build_new_store(
                embedder=gemini_embedder,
                persist_path=vs_path
            )
            print("✓ Vector store created and saved")
    
    def _build_new_store(self, embedder: Embeddings, persist_path: str) -> VectorStore:
        """
        Builds a new vector store from scratch.
        
        Steps:
        1. Load documents (PDF + YAML)
        2. Chunk documents
        3. Embed and store
        """
        # 1. Load
        print(f"\n📚 Loading from {constants.CORPUS_DIR}...")
        all_docs = loader.load_corpus(str(constants.CORPUS_DIR))
        
        if not all_docs:
            raise ValueError("No documents loaded. Check corpus_dir.")
        
        # 2. Chunk
        print(f"\n✂️ Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=constants.CHUNK_SIZE,
            chunk_overlap=constants.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(all_docs)
        print(f"   • Original documents: {len(all_docs)}")
        print(f"   • Total chunks: {len(chunks)}")
        
        # 3. Embed & Store
        print(f"\n🔢 Embedding chunks and building Chroma vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            persist_directory=persist_path
        )
        
        return vector_store
    
    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Search the vector store.
        
        Args:
            query: Search query
            k: Number of results (default from constants)
            
        Returns:
            List of relevant documents
        """
        k = k or constants.DEFAULT_K_RESULTS
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        return retriever.invoke(query)
    
    def get_langchain_retriever(self, k_results: int = None):
        """
        Returns a LangChain retriever object.
        
        Args:
            k_results: Number of results to retrieve
            
        Returns:
            LangChain retriever
        """
        k = k_results or constants.DEFAULT_K_RESULTS
        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )


if __name__ == "__main__":
    # Test the retriever
    print("=" * 70)
    print("TESTING RAG RETRIEVER")
    print("=" * 70)
    
    try:
        # Build/load vector store
        retriever = CorpusRetriever(rebuild=False)
        
        # Test search
        test_query = "What are the symptoms?"
        print(f"\n🔍 Testing search: '{test_query}'")
        
        results = retriever.search(test_query, k=3)
        
        print(f"\n✓ Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            if '/' in source:
                source = source.split('/')[-1]
            if '\\' in source:
                source = source.split('\\')[-1]
            
            page = doc.metadata.get('page', 'N/A')
            print(f"\n{i}. {source} (Page {page})")
            print(f"   Content: {doc.page_content[:150]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")