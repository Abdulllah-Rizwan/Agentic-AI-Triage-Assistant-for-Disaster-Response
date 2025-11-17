├── rag/
│   ├── __init__.py
│   ├── retriever.py             ← Main class everyone imports
│   ├── loader.py                ← Loads Golden Corpus (PDF + YAML metadata)
│   ├── embedder.py              ← Gemini embeddings (fast & free)
│   ├── postprocessor.py          ← Forces citation format + blocks hallucination
│   └── constants.py