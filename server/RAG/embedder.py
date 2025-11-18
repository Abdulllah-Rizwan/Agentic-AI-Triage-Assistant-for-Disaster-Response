import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_gemini_embedder() -> Embeddings:
    """
    Initializes and returns the Google Generative AI (Gemini) embedding model.
    
    Returns:
        GoogleGenerativeAIEmbeddings instance
        
    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Get your key from https://makersuite.google.com/app/apikey"
        )

    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
        task_type="retrieval_document"
    )
    
    return embedder


if __name__ == "__main__":
    # Test the embedder
    try:
        print("Testing embedder...")
        embedder = get_gemini_embedder()
        
        test_text = "What is a large language model?"
        vector = embedder.embed_query(test_text)
        
        print(f"✓ Embedder working!")
        print(f"  Query: '{test_text}'")
        print(f"  Vector dimension: {len(vector)}")
        print(f"  First 5 values: {vector[:5]}")
        
    except Exception as e:
        print(f"✗ Error: {e}")