import os
from pathlib import Path
from ingestion import DocumentChunker, VectorStore

# Create the chunker and vector store
chunker = DocumentChunker()
vector_store = VectorStore()

# Get all files in docs folder
files = list(Path("docs").glob("*.*"))

# Process in batches of 10
batch_size = 10

for i in range(0, len(files), batch_size):
    batch = files[i:i + batch_size]  # Take 10 files
    print(f"Processing batch {i//batch_size + 1}...")
    
    chunks = []
    for file in batch:
        text = chunker.load_document(str(file))
        file_chunks = chunker.chunk_text(text)
        for j, chunk in enumerate(file_chunks):
            chunks.append({
                "source": file.name,
                "chunk_id": j,
                "text": chunk,
                "token_count": len(chunker.encoding.encode(chunk))
            })
    
    # Save these chunks to the vector store
    if chunks:
        vector_store.add_chunks(chunks)
        print(f"  Added {len(chunks)} chunks")

print("All done!")