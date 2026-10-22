def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def chunk_documents(documents, chunk_size=300, overlap=50):
    corpus_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            corpus_chunks.append({
                "chunk_id": f"{doc['file_name']}_chunk_{i}",
                "domain": doc["domain"],
                "file_name": doc["file_name"],
                "content": chunk,
                "path": doc["path"]
            })
    return corpus_chunks
