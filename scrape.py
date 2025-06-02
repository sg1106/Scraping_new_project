

import requests
from bs4 import BeautifulSoup
import logging
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to scrape news articles from BBC
def scrape_news(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Scrape all headlines
        articles = soup.find_all('h2', {'data-testid': 'card-headline'})
        titles = [article.get_text().strip() for article in articles if article.get_text().strip()]
        
        if not titles:
            logging.warning("No articles found with the given selector. Check the website's HTML structure.")
            return []
        logging.info(f"Scraped {len(titles)} articles from {url}")
        return titles
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        return []

# Function to call Gemini LLM for fake news detection
def predict_news_gemini(text):
    try:
        # Placeholder for Gemini API call (replace with actual Gemini API client)
        api_key = "YOUR_GEMINI_API_KEY"  # Replace with your Gemini API key
        endpoint = "https://api.gemini.ai/v1/completions"  # Replace with actual Gemini API endpoint
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prompt for Gemini to classify news as Real or Fake
        prompt = f"""
        You are an expert in detecting fake news. Analyze the following news headline and determine if it is likely to be real or fake. 
        Provide a one-word response: 'Real' or 'Fake'. Do not include any explanation or additional text.
        Headline: {text}
        """
        
        payload = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.5
        }
        
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract the prediction (adjust based on actual Gemini API response structure)
        prediction = result.get('choices', [{}])[0].get('text', 'Unknown').strip()
        if prediction not in ['Real', 'Fake']:
            logging.warning(f"Unexpected response from Gemini: {prediction}")
            return "Unknown"
        return prediction
    except Exception as e:
        logging.error(f"Error calling Gemini API for text '{text}': {e}")
        return "Unknown"

# Main function
def main():
    url = "https://www.bbc.com/news"
    articles = scrape_news(url)
    time.sleep(1)  # Respectful delay to avoid overloading the server
    
    if articles:
        logging.info("Scraped articles and predictions:")
        for article in articles:
            prediction = predict_news_gemini(article)
            print(f"Article: {article}")
            print(f"Prediction: {prediction}")
            print("-" * 50)
    else:
        logging.warning("No articles scraped. Check the URL or scraping logic.")

if __name__ == "__main__":
    main()