from typing import List
from langchain_core.documents import Document


def reorder_documents(docs: List[Document]) -> List[Document]:
    """
    Re-orders documents to combat "lost in the middle" problem.
    
    Places documents in order: [0, 2, 4, ..., 5, 3, 1]
    This ensures important info isn't buried in the middle.
    
    Args:
        docs: List of retrieved documents
        
    Returns:
        Re-ordered list of documents
    """
    if not docs:
        return []

    reordered = []
    
    # Add even-indexed docs (0, 2, 4, ...)
    for i, doc in enumerate(docs):
        if i % 2 == 0:
            reordered.append(doc)
    
    # Add odd-indexed docs in reverse (..., 5, 3, 1)
    for i in range(len(docs) - 1, -1, -1):
        if i % 2 != 0:
            reordered.append(docs[i])
    
    return reordered


def format_docs_with_citations(docs: List[Document]) -> str:
    """
    Formats documents with clear citations for the LLM.
    Forces the LLM to reference sources properly.
    
    Args:
        docs: List of documents
        
    Returns:
        Formatted string with citations
    """
    if not docs:
        return "No relevant documents found."

    context_parts = []
    
    for i, doc in enumerate(docs, 1):
        # Extract source filename
        source = doc.metadata.get("source", "Unknown")
        if "/" in source:
            source = source.split("/")[-1]
        if "\\" in source:
            source = source.split("\\")[-1]
        
        # Get page number (add 1 for human-readable format)
        page = doc.metadata.get("page")
        page_str = f"Page {page + 1}" if isinstance(page, int) else "Page N/A"
        
        # Build citation
        citation = f"[{source}, {page_str}]"
        
        # Format document
        context_parts.append(
            f"--- Document {i} ---\n"
            f"Citation: {citation}\n"
            f"Content: {doc.page_content}\n"
        )
    
    return "\n".join(context_parts)


def extract_metadata_field(docs: List[Document], field: str) -> List[str]:
    """
    Extracts a specific metadata field from all documents.
    
    Useful for getting categories, tags, etc.
    
    Args:
        docs: List of documents
        field: Metadata field name
        
    Returns:
        List of unique values for that field
    """
    values = set()
    for doc in docs:
        value = doc.metadata.get(field)
        if value:
            if isinstance(value, list):
                values.update(value)
            else:
                values.add(value)
    return list(values)


if __name__ == "__main__":
    # Test
    from langchain_core.documents import Document
    
    test_docs = [
        Document(
            page_content="Test content 1",
            metadata={"source": "/path/to/doc.pdf", "page": 0}
        ),
        Document(
            page_content="Test content 2",
            metadata={"source": "/path/to/doc.pdf", "page": 1}
        )
    ]
    
    formatted = format_docs_with_citations(test_docs)
    print("Formatted output:")
    print(formatted)