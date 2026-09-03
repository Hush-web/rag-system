import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

class VectorStore:
    """Manage embeddings and vector storage using ChromaDB."""

    def __init__(self, collection_name: str = "documents", persist_directory: str = "./data/chromadb"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Use the default embedding function (all-MiniLM-L6-v2)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Check if collection exists, create if not
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' loaded")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Collection '{collection_name}' created")

    def add_chunks(self, chunks: list) -> int:
        """Add chunks to the vector database."""
        if not chunks:
            return 0
            
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            doc_id = f"{chunk['source']}_{chunk['chunk_id']}"
            ids.append(doc_id)
            documents.append(chunk['text'])
            metadatas.append({
                "source": chunk['source'],
                "chunk_id": chunk['chunk_id'],
                "token_count": chunk['token_count']
            })
        
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            return 0

    def search(self, query: str, n_results: int = 5) -> list:
        """Search for similar chunks."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.collection.count()
        except:
            return 0

    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        except:
            pass
        