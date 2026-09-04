from app.ingestion import VectorStore

store = VectorStore(collection_name="rag_docs")
print(f"Count: {store.get_count()}")

if store.get_count() > 0:
    results = store.search("What is RAG?", n_results=3)
    print(results)
else:
    print("Collection is empty.")
