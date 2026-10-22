import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import tokenize_and_filter
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

def rescue_idx(sentences, target):
    for i, s in enumerate(sentences):
        if target in s: return i
    return 0

def construct_response_from_chunk(query, chunk_content, request_type, vectorizer):
    sentences = re.split(r'(?<=[.!?])\s+', chunk_content)
    
    query_tokens = set(tokenize_and_filter(query))
    query_vec = vectorizer.transform([query])
    
    scored_sentences = []
    
    actionable_verbs = {"reset", "update", "add", "delete", "contact", "submit", "login", "create", "remove", "change", "download", "install", "follow", "click", "navigate", "go", "use", "select", "enter", "provide"}
    
    for sentence in sentences:
        sentence_clean = sentence.strip()
        sentence_clean = re.sub(r':\s*\.$', ':', sentence_clean)
        words = sentence_clean.split()
        
        if len(words) < 6:
            continue
        if sentence_clean.endswith(":"):
            continue
            
        sentence_lower = sentence_clean.lower()
        has_action = any(verb in sentence_lower for verb in actionable_verbs)
        if not has_action:
            continue
            
        sentence_tokens = set(tokenize_and_filter(sentence_clean))
        keyword_overlap = len(query_tokens.intersection(sentence_tokens))
        
        sent_vec = vectorizer.transform([sentence_clean])
        tfidf_similarity = cosine_similarity(query_vec, sent_vec)[0][0]
        
        length_penalty = 0
        if len(sentence_clean) > 250:
            length_penalty = 0.5
            
        if "?" in sentence_clean:
            length_penalty += 1.0
            
        score = (keyword_overlap * 2) + (tfidf_similarity * 3) - length_penalty
        
        scored_sentences.append({
            "text": sentence_clean,
            "overlap": keyword_overlap,
            "tfidf": float(tfidf_similarity),
            "penalty": length_penalty,
            "score": float(score)
        })
        
    scored_sentences.sort(key=lambda x: x["score"], reverse=True)
    
    if not scored_sentences or scored_sentences[0]["score"] <= 0.5:
        return "", "no sentence achieved strong keyword/semantic overlap or actionable criteria", scored_sentences
        
    top_sentences = [scored_sentences[0]["text"]]
    
    if len(scored_sentences) > 1 and scored_sentences[1]["score"] >= scored_sentences[0]["score"] * 0.7 and scored_sentences[1]["score"] > 0:
        idx0 = rescue_idx(sentences, scored_sentences[0]["text"])
        idx1 = rescue_idx(sentences, scored_sentences[1]["text"])
        
        if idx0 < idx1:
            top_sentences.append(scored_sentences[1]["text"])
        else:
            top_sentences.insert(0, scored_sentences[1]["text"])
            
    final_response = " ".join(top_sentences)
    final_response = re.sub(r'\s+', ' ', final_response).strip()
    
    return final_response, "Answer synthesized from retrieved corpus via weighted sentence extraction.", scored_sentences
