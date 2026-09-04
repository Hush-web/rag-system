from app.ingestion import VectorStore

store = VectorStore(collection_name="rag_docs")
print(f"Number of documents: {store.get_count()}")
