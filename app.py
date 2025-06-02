

from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import logging
import time
from dotenv import load_dotenv
import os
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Gemini API
GEMINI_MODEL = "gemini-2.0-flash"
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    logging.error("GEMINI_API_KEY not found")
else:
    genai.configure(api_key=api_key)

# BBC configuration
BBC_URL = 'https://www.bbc.com/news'
BBC_SELECTOR = 'h2[data-testid="card-headline"]'

# Cache for scraped headlines and embeddings
BBC_HEADLINES_CACHE = []
BBC_EMBEDDINGS_CACHE = None

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# NLTK stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tagged = pos_tag(tokens)
    tokens = [token for token, pos in tagged if pos in ('NNP', 'NNPS') or (token.isalnum() and token not in stop_words)]
    return ' '.join(tokens)

def scrape_news():
    global BBC_HEADLINES_CACHE, BBC_EMBEDDINGS_CACHE
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(BBC_URL, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.select(BBC_SELECTOR)
        titles = [article.get_text().strip() for article in articles if article.get_text().strip()]

        if not titles:
            logging.warning("No articles found for BBC")
            return []

        BBC_HEADLINES_CACHE = titles[:10]
        BBC_EMBEDDINGS_CACHE = embedding_model.encode(BBC_HEADLINES_CACHE, convert_to_numpy=True)
        logging.info(f"Scraped {len(BBC_HEADLINES_CACHE)} BBC articles and cached embeddings.")
        return BBC_HEADLINES_CACHE
    except Exception as e:
        logging.error(f"Error scraping BBC: {e}")
        return []

def compute_bert_similarity(input_headline):
    global BBC_HEADLINES_CACHE, BBC_EMBEDDINGS_CACHE
    if not BBC_HEADLINES_CACHE or BBC_EMBEDDINGS_CACHE is None:
        return 0.0
    input_embedding = embedding_model.encode([input_headline], convert_to_numpy=True)
    similarities = cosine_similarity(input_embedding, BBC_EMBEDDINGS_CACHE)
    max_sim = np.max(similarities)
    return max_sim

def predict_news_gemini(text):
    try:
        if not api_key:
            logging.error("GEMINI_API_KEY not found")
            return "Unknown"

        similarity = compute_bert_similarity(text)
        logging.info(f"BERT similarity for '{text}': {similarity:.4f}")

        if similarity > 0.99:
            similarity_pct = similarity * 100
            logging.info(f"High similarity ({similarity_pct:.2f}%) to BBC headlines — classifying as Real.")
            # Return Real with similarity percentage
            return f"Real (Similarity: {similarity_pct:.2f}%)"

        example_headlines = "\n".join([f"- '{h}' → Real" for h in BBC_HEADLINES_CACHE[:5]]) if BBC_HEADLINES_CACHE else ""

        prompt = f"""
You are an expert in detecting fake news. Analyze the news headline for credibility, focusing on the plausibility of named entities (e.g., countries, organizations) in the given context, such as geopolitical or humanitarian situations. Use the provided BBC headlines as examples of credible, factual reporting. Classify it as 'Real' if it aligns with BBC-like reporting, or 'Fake' if the entities or context are implausible or exaggerated.

Return exactly one word: 'Real' or 'Fake'. No other text.

BBC Examples:
{example_headlines}
- "Gaza aid trucks rushed by desperate and hungry crowds, WFP says" → Real
- "Pakistan aid trucks rushed by desperate and hungry crowds, WFP says" → Fake (less plausible without a reported crisis)
- "EU 'strongly' regrets US plan to double steel tariffs" → Real
- "India 'strongly' regrets US plan to double steel tariffs" → Fake
- "Scientists discover new species in Pacific Ocean" → Real
- "Aliens invade Earth tomorrow" → Fake

Headline: {text}
"""

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        prediction = response.text.strip()
        logging.info(f"Gemini prediction for '{text}': {prediction}")

        if prediction not in ['Real', 'Fake']:
            logging.warning(f"Invalid Gemini response '{prediction}' for headline '{text}'")
            return "Unknown"

        return prediction
    except Exception as e:
        logging.error(f"Error predicting news: {e}")
        return "Unknown"

@app.route('/')
def index():
    return render_template('index.html', results=None)

@app.route('/predict', methods=['POST'])
def predict():
    results = []
    if request.form.get('scrape_bbc'):
        articles = scrape_news()
        time.sleep(1)
        if articles:
            for article in articles:
                results.append({'headline': article})
        else:
            results.append({'headline': 'No articles found for BBC'})
    else:
        headline = request.form.get('headline', '').strip()
        if headline:
            prediction = predict_news_gemini(headline)
            results.append({'headline': headline, 'prediction': prediction})
        else:
            results.append({'headline': 'No headline provided', 'prediction': 'N/A'})
    
    return render_template('index.html', results=results)

if __name__ == '__main__':
    # Preload BBC news cache and embeddings on app startup
    scrape_news()
    app.run(debug=True)
