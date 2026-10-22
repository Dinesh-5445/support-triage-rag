import re
from pathlib import Path

def clean_markdown(text):
    text = re.sub(r'^---[\s\S]*?---\n', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[#\*_`~]', '', text)
    text = re.sub(r'\n+', ' . ', text)
    return text.strip()

def load_documents(data_dir: Path):
    documents = []
    if not data_dir.exists():
        return documents
        
    for domain_dir in data_dir.iterdir():
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name.lower()
        for file_path in domain_dir.rglob("*.md"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                clean_content = clean_markdown(content)
                documents.append({
                    "domain": domain,
                    "file_name": file_path.name,
                    "content": clean_content,
                    "path": str(file_path)
                })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
    return documents
