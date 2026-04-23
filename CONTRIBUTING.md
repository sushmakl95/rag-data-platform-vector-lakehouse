# Contributing

## Dev setup
```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements-dev.txt
```

## Before pushing
```bash
make ci
```

## Adding a new vector store
1. Implement the `VectorStore` protocol in `src/rag/vector_stores/`.
2. Add unit tests in `tests/unit/` covering `add()`, `query()`, `delete()`.
3. Register in the factory in `src/rag/vector_stores/__init__.py`.
