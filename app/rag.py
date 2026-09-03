import os
from dotenv import load_dotenv
from loguru import logger
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from ingestion import VectorStore, DocumentChunker

# Load environment variables
load_dotenv()

class RAGSystem:
    """End-to-end RAG system: retrieval + generation."""

    def __init__(self):
        self.vector_store = VectorStore(collection_name="rag_docs")
        self.llm = ChatGroq(
            model="qwen/qwen3.6-27b",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.chunker = DocumentChunker()

    def answer(self, query: str, n_results: int = 3) -> dict:
        """Answer a question using RAG."""
        # 1. Retrieve relevant chunks
        results = self.vector_store.search(query, n_results=n_results)
        
        if not results or not results.get('documents'):
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "query": query
            }

        # 2. Build context from retrieved chunks
        context = ""
        sources = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context += f"[{i+1}] {doc}\n\n"
            sources.append({
                "source": metadata['source'],
                "chunk_id": metadata['chunk_id']
            })

        # 3. Generate answer using Groq
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