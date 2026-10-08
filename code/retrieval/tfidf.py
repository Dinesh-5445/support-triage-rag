import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                 )
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

from retrieval.base import BaseRetriever
from utils import tokenize_and_filter

def clean_markdown(text):
    text = re.sub(r'^---[\s\S]*?---\n', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[#\*_`~]', '', text)
    text = re.sub(r'\n+', ' . ', text)
    return text.strip()

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def load_corpus(data_dir):
    corpus_chunks = []
    if not data_dir.exists():
        return corpus_chunks
        
    for domain_dir in data_dir.iterdir():
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name.lower()
        for file_path in domain_dir.rglob("*.md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                clean_content = clean_markdown(content)
                chunks = chunk_text(clean_content)
                
                for chunk in chunks:
                    corpus_chunks.append({
                        "domain": domain,
                        "file_name": file_path.name,
                        "content": chunk,
                        "path": str(file_path)
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    return corpus_chunks

class TFIDFRetriever(BaseRetriever):
    def __init__(self, data_dir):
        if TfidfVectorizer is None:
            raise ImportError("scikit-learn is required for TF-IDF. Run pip install scikit-learn")
            
        self.corpus_chunks = load_corpus(data_dir)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        
        texts = [chunk["content"] for chunk in self.corpus_chunks]
        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            self.tfidf_matrix = None

    def retrieve(self, query, expected_domain, top_k=3):
        if not self.vectorizer or self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[::-1][:top_k * 3]
        
        retrieved = []
        expected_domain_lower = expected_domain.lower() if expected_domain and expected_domain != "Unknown" else None
        
        for idx in top_indices:
            chunk = self.corpus_chunks[idx]
            score = similarities[idx]
            
            if expected_domain_lower and expected_domain_lower != "none":
                if chunk["domain"] != expected_domain_lower:
                    continue
                    
            retrieved.append({
                "content": chunk["content"],
                "score": float(score),
                "domain": chunk["domain"]
            })
            
            if len(retrieved) >= top_k:
                break
                
        return retrieved

def validate_chunk(query, chunk):
    chunk_score = chunk["score"]
    query_tokens = set(tokenize_and_filter(query))
    chunk_tokens = set(tokenize_and_filter(chunk["content"]))
    overlap = len(query_tokens.intersection(chunk_tokens))
    
    if chunk_score < 0.25:
        return False, f"TF-IDF score ({chunk_score:.2f}) < threshold (0.25)", chunk_score, overlap
        
    if overlap < 2:
        return False, f"keyword overlap ({overlap}) < threshold (2)", chunk_score, overlap
            
    return True, "Valid retrieval", chunk_score, overlap
