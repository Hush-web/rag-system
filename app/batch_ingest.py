import os
from pathlib import Path
from ingestion import DocumentChunker, VectorStore
from loguru import logger

# Create the chunker and vector store
chunker = DocumentChunker()
vector_store = VectorStore()

# Get all files in docs folder
files = list(Path("docs").glob("*.*"))
supported = [f for f in files if f.suffix.lower() in ['.txt', '.pdf']]

logger.info(f"Found {len(supported)} documents")

# Process in batches of 10
batch_size = 10
all_chunks = []

for i in range(0, len(supported), batch_size):
    batch = supported[i:i + batch_size]
    batch_num = i // batch_size + 1
    logger.info(f"Processing batch {batch_num}...")

    chunks = []
    for file in batch:
        try:
            text = chunker.load_document(str(file))
            file_chunks = chunker.chunk_text(text)
            for j, chunk in enumerate(file_chunks):
                chunks.append({
                    "source": file.name,
                    "chunk_id": j,
                    "text": chunk,
                    "token_count": len(chunker.encoding.encode(chunk))
                })
            logger.info(f"  → {file.name}: {len(file_chunks)} chunks")
        except Exception as e:
            logger.error(f"  ✗ {file.name}: {e}")

    # Save these chunks to the vector store
    if chunks:
        vector_store.add_chunks(chunks)
        all_chunks.extend(chunks)
        logger.info(f"  Added {len(chunks)} chunks")

# Build BM25 index from all chunks
logger.info("Building BM25 index...")
chunker.build_bm25_index(all_chunks)
logger.info("BM25 index built and saved.")

logger.info("All done!")