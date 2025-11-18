import yaml
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_corpus(corpus_dir: str) -> List[Document]:
    """
    Loads all PDF documents and merges metadata from matching YAML files.
    
    For each PDF:
    1. Looks for a matching .yaml file (same name)
    2. Loads YAML metadata if it exists
    3. Loads the PDF
    4. Merges YAML metadata into each page
    
    Args:
        corpus_dir: Path to directory with PDFs and YAMLs
        
    Returns:
        List of Document objects with merged metadata
    """
    corpus_path = Path(corpus_dir)
    all_documents = []
    
    if not corpus_path.exists():
        raise ValueError(f"Corpus directory does not exist: {corpus_dir}")
    
    # Find all PDFs
    pdf_files = list(corpus_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠ Warning: No PDF files found in {corpus_dir}")
        return all_documents
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    print("=" * 60)
    
    for pdf_path in pdf_files:
        print(f"\n📄 Processing: {pdf_path.name}")
        
        # 1. Check for matching YAML
        yaml_path = pdf_path.with_suffix('.yaml')
        custom_metadata = {}
        
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    custom_metadata = yaml.safe_load(f) or {}
                print(f"   ✓ Loaded YAML metadata: {list(custom_metadata.keys())}")
            except Exception as e:
                print(f"   ✗ Error reading YAML: {e}")
                custom_metadata = {}
        else:
            print(f"   ℹ No YAML file found (optional)")
        
        # 2. Load PDF
        try:
            loader = PyPDFLoader(str(pdf_path))
            pdf_pages = loader.load()
            print(f"   ✓ Loaded {len(pdf_pages)} page(s)")
            
            # 3. Merge YAML metadata into each page
            for page_doc in pdf_pages:
                page_doc.metadata.update(custom_metadata)
                page_doc.metadata['source'] = str(pdf_path)
            
            all_documents.extend(pdf_pages)
            
        except Exception as e:
            print(f"   ✗ Error loading PDF: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"✓ Total documents loaded: {len(all_documents)}")
    print("=" * 60)
    
    return all_documents


if __name__ == "__main__":
    # Test the loader
    from . import constants
    
    try:
        print("Testing loader...")
        print(f"Corpus directory: {constants.CORPUS_DIR}\n")
        
        docs = load_corpus(str(constants.CORPUS_DIR))
        
        if docs:
            print("\n--- First Document Sample ---")
            print(f"Content: {docs[0].page_content[:200]}...")
            print(f"Metadata: {docs[0].metadata}")
        
    except Exception as e:
        print(f"Error: {e}")