import os
import re
from typing import List, Dict, Any
from pathlib import Path
import pdfplumber
from loguru import logger
import tiktoken
import chromadb
from fastembed import TextEmbedding
import numpy as np

class DocumentChunker:
    """Load and chunk documents from various formats."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def load_text_file(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_pdf(self, path: str) -> str:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def load_document(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == '.txt':
            return self.load_text_file(path)
        elif ext == '.pdf':
            return self.load_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def chunk_text(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            if not para.strip():
                continue
            para_tokens = len(self.encoding.encode(para))
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                overlap_text = current_chunk[-1] if self.overlap > 0 else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_tokens = len(self.encoding.encode(overlap_text)) if overlap_text else 0

            current_chunk.append(para)
            current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def process_directory(self, directory: str) -> List[Dict[str, Any]]:
        results = []
        for file_path in Path(directory).glob("*"):
            if file_path.suffix.lower() in ['.txt', '.pdf']:
                try:
                    logger.info(f"Processing: {file_path.name}")
                    text = self.load_document(str(file_path))
                    chunks = self.chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        results.append({
                            "source": file_path.name,
                            "chunk_id": i,
                            "text": chunk,
                            "token_count": len(self.encoding.encode(chunk))
                        })
                    logger.info(f"  → {len(chunks)} chunks created")
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
        return results


class FastEmbeddingFunction:
    """Custom embedding function using fastembed."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name=model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB expects parameter name 'input'."""
        embeddings = list(self.model.embed(input))
        return [emb.tolist() for emb in embeddings]


class VectorStore:
    """Manage embeddings and vector storage using ChromaDB with fastembed."""

    def __init__(self, collection_name: str = "documents", persist_directory: str = "./data/chromadb"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        self.embedding_function = FastEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        
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

    def search(self, query: str, n_results: int = 5) -> dict:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {}

    def get_count(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0


def setup_sample_documents():
    os.makedirs("docs", exist_ok=True)
    
    sample_docs = [
        ("rag_overview.txt", """
RAG stands for Retrieval-Augmented Generation. It is a technique that combines retrieval of relevant documents with generative AI to answer questions.

The RAG architecture works in four steps:
1. Document Ingestion: Documents are loaded, cleaned, and split into chunks.
2. Embedding: Each chunk is converted to a vector using embedding models.
3. Storage: Vectors are stored in a vector database for efficient search.
4. Retrieval & Generation: When a question is asked, similar chunks are retrieved and used as context for the LLM.

RAG reduces hallucinations, improves accuracy, and allows AI to access external knowledge beyond its training data.
"""),
        ("vector_databases.txt", """
Vector databases are specialized databases designed to store and search high-dimensional vectors.

Popular vector databases include:
- ChromaDB: Lightweight, open-source, good for prototyping
- Pinecone: Cloud-based, scalable, enterprise-grade
- Weaviate: Open-source with GraphQL API
- Qdrant: High-performance, open-source
- Milvus: Distributed, designed for large-scale applications

Vector databases enable efficient similarity search, which is the foundation of RAG systems.
"""),
        ("embeddings.txt", """
Embeddings are numerical representations of text that capture semantic meaning.

Words, sentences, and documents are converted to vectors of floating-point numbers (e.g., 384 dimensions for all-MiniLM-L6-v2).

Similar meanings have similar vectors, enabling operations like:
- Similarity search: Find documents related to a query
- Semantic clustering: Group documents by theme
- Recommendation: Find similar items

Popular embedding models include:
- OpenAI embeddings: ADA-002 (1536 dimensions)
- Sentence Transformers: all-MiniLM-L6-v2 (384 dimensions)
- Cohere embeddings: multilingual
- BGE: Chinese and English support
""")
    ]
    
    for filename, content in sample_docs:
        filepath = os.path.join("docs", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        logger.info(f"Created: {filepath}")


if __name__ == "__main__":
    setup_sample_documents()
    
    chunker = DocumentChunker(chunk_size=500, overlap=50)
    chunks = chunker.process_directory("docs")
    
    print(f"\nTotal chunks created: {len(chunks)}")
    for r in chunks[:3]:
        print(f"\n--- Chunk {r['chunk_id']} from {r['source']} ---")
        print(r['text'][:200] + "...")
    
    print("\n" + "="*50)
    print("Creating Vector Store...")
    
    vector_store = VectorStore(collection_name="rag_docs")
    count = vector_store.add_chunks(chunks)
    print(f"Total vectors in store: {vector_store.get_count()}")
    
    print("\n" + "="*50)
    print("Testing Search...")
    
    test_queries = [
        "What is RAG?",
        "What are vector databases?",
        "How do embeddings work?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.search(query, n_results=2)
        
        if results and results.get('documents') and results['documents'][0]:
            print(f"Found {len(results['documents'][0])} results:")
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"  {i+1}. From: {metadata['source']}")
                print(f"     Preview: {doc[:100]}...")
        else:
            print("  No results found.")