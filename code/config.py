import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = BASE_DIR / "data"

class Settings:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    USE_LLM = bool(OPENROUTER_API_KEY)

OUTPUT_CSV = BASE_DIR / "results" / "evaluation_results.csv"
LOG_FILE = BASE_DIR / "results" / "log.txt"

STOPWORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
             "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
             "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
             "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
             "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
             "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
             "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
             "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
             "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
             "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
             "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
             "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "please", "help"}

def get_settings():
    return Settings
