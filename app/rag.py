import os
import pickle
from dotenv import load_dotenv
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from ingestion import VectorStore, DocumentChunker

load_dotenv()

class RAGSystem:
    """End-to-end RAG system with hybrid retrieval."""

    def __init__(self):
        self.vector_store = VectorStore(collection_name="rag_docs")
        self.llm = ChatGroq(
            model="groq/compound-mini",  # working model with high limits
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=300
        )
        self.chunker = DocumentChunker()
        self.load_bm25_index()

    def load_bm25_index(self):
        try:
            with open('data/bm25_index.pkl', 'rb') as f:
                self.bm25, self.chunks, self.chunk_map = pickle.load(f)
            logger.info("BM25 index loaded.")
        except FileNotFoundError:
            self.bm25 = None
            self.chunks = []
            self.chunk_map = {}
            logger.warning("BM25 index not found. Run ingestion first.")

    def bm25_search(self, query: str, k: int = 10) -> list:
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = [self.chunks[i] for i in top_indices]
        return results

    def hybrid_search(self, query: str, k: int = 10) -> list:
        # Vector search
        vector_results = self.vector_store.search(query, n_results=k)
        vector_chunks = []
        seen_keys = set()

        for meta in vector_results.get('metadatas', [[]])[0]:
            key = f"{meta['source']}_{meta['chunk_id']}"
            if key in self.chunk_map and key not in seen_keys:
                seen_keys.add(key)
                vector_chunks.append(self.chunk_map[key])

        # BM25 search
        bm25_chunks = self.bm25_search(query, k=k)

        # Merge, deduplicate
        all_chunks = []
        seen_keys = set()

        for chunk in vector_chunks:
            key = f"{chunk['source']}_{chunk['chunk_id']}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_chunks.append(chunk)

        for chunk in bm25_chunks:
            key = f"{chunk['source']}_{chunk['chunk_id']}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_chunks.append(chunk)

        return all_chunks

    def answer(self, query: str) -> dict:
        retrieved = self.hybrid_search(query, k=10)

        if not retrieved:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "query": query
            }

        top_chunks = retrieved[:5]

        context = ""
        sources = []
        for i, chunk in enumerate(top_chunks):
            context += f"[{i+1}] {chunk['text']}\n\n"
            sources.append({
                "source": chunk['source'],
                "chunk_id": chunk['chunk_id']
            })

        system_prompt = """You are a helpful assistant. Answer the user's question based on the provided context.

        Rules:
        - Use ONLY the context provided.
        - If the answer is not in the context, say "I don't have enough information."
        - Cite sources using [1], [2], etc. where applicable.
        - Be concise and clear.

        Context:
        {context}

        Question: {query}

        Answer:"""

        messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on provided context."),
            HumanMessage(content=system_prompt.format(context=context, query=query))
        ]

        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": sources,
            "query": query
        }


if __name__ == "__main__":
    rag = RAGSystem()

    print("\n" + "="*60)
    print("RAG System Ready! Ask me anything about your documents.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*60 + "\n")

    while True:
        query = input("Your question: ")

        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not query.strip():
            print("Please enter a question.\n")
            continue

        print("\n" + "="*60)
        print(f"Query: {query}")
        print("="*60)

        result = rag.answer(query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        print("\n" + "-"*60 + "\n")