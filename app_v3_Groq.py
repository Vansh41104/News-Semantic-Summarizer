import os
import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from textblob import TextBlob
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
import re
from newspaper import Article
import logging
from urllib.parse import urlparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Groq client with error handling
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not found in environment variables. Some features will be limited.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# Load spaCy model with fallback
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Loaded spaCy model: en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Attempting to download...")
    try:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully downloaded and loaded spaCy model")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {str(e)}")
        # Fallback to simple processing if spaCy fails
        nlp = None

def is_valid_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_main_content(url):
    """Extract main content using multiple methods for better compatibility"""
    # First attempt: Use newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # If article text is reasonably long, use it
        if article.text and len(article.text) > 500:
            return {
                "text": article.text,
                "title": article.title,
                "authors": article.authors,
                "publish_date": article.publish_date,
                "source": "newspaper3k"
            }
    except Exception as e:
        logger.info(f"newspaper3k extraction failed: {str(e)}")
    
    # Second attempt: Custom extraction with BeautifulSoup
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove boilerplate elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'form']):
            element.decompose()
            
        # Extract title
        title = soup.title.text if soup.title else ""
        
        # Try to find main content with common news site patterns
        main_content = None
        
        # Look for article or main content containers
        content_candidates = [
            soup.find('article'),
            soup.find('main'),
            soup.find(id=re.compile('article|content|main|story', re.I)),
            soup.find(class_=re.compile('article|content|main|story', re.I)),
            soup.find(role='main')
        ]
        
        # Use the first non-None candidate
        for candidate in content_candidates:
            if candidate and len(' '.join(candidate.stripped_strings)) > 500:
                main_content = candidate
                break
        
        # If no main content found, use body with specific paragraph filtering
        if not main_content:
            paragraphs = soup.find_all('p')
            content_paragraphs = [p.text for p in paragraphs if len(p.text) > 50]
            if content_paragraphs:
                return {
                    "text": ' '.join(content_paragraphs),
                    "title": title,
                    "authors": [],
                    "publish_date": None,
                    "source": "paragraph_filter"
                }
        else:
            return {
                "text": ' '.join(main_content.stripped_strings),
                "title": title,
                "authors": [],
                "publish_date": None,
                "source": "main_content_container"
            }
    except Exception as e:
        logger.warning(f"Custom extraction failed: {str(e)}")
    
    # Last resort: Just extract all text from body
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, "html.parser")
        
        for element in soup(['script', 'style']):
            element.decompose()
            
        return {
            "text": ' '.join(soup.body.stripped_strings) if soup.body else "",
            "title": soup.title.text if soup.title else "",
            "authors": [],
            "publish_date": None,
            "source": "fallback"
        }
    except Exception as e:
        logger.error(f"All extraction methods failed: {str(e)}")
        return {"error": f"Content extraction error: {str(e)}"}

def semantic_analyzer(url):
    """Improved semantic analysis with content filtering and robust error handling"""
    if not is_valid_url(url):
        return {"error": "Invalid URL format. Please enter a complete URL including http:// or https://"}
    
    try:
        # Extract content using enhanced method
        content = extract_main_content(url)
        
        if "error" in content:
            return content
            
        text = content["text"]
        
        if not text or len(text) < 200:
            return {"error": "Insufficient content extracted from the URL. This may not be a news article."}
        
        # Process with spaCy if available
        if nlp:
            # Limit processing to first 100k characters for performance
            doc = nlp(text[:100000])
            
            relevant_entities = [
                (ent.text, ent.label_) 
                for ent in doc.ents
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'LOC', 'FAC', 'WORK_OF_ART', 'LAW']
            ]
            
            # Get more relevant keywords using proper frequency filtering
            keywords = Counter([
                token.lemma_.lower() 
                for token in doc
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] 
                and not token.is_stop
                and len(token.text) > 2
            ]).most_common(15)
            
            # Get word frequency for the most common non-stopwords
            word_freq = Counter([
                token.text.lower() 
                for token in doc 
                if token.is_alpha and not token.is_stop and len(token.text) > 2
            ]).most_common(10)
            
        else:
            # Fallback without spaCy (limited functionality)
            relevant_entities = []
            keywords = []
            word_freq = Counter()
            for word in text.split():
                if len(word) > 3:  # Simple filtering
                    word_freq[word.lower()] += 1
            word_freq = word_freq.most_common(10)
        
        # Calculate sentiment
        try:
            sentiment = TextBlob(text).sentiment.polarity
        except:
            sentiment = 0
            
        # Calculate reading level using Flesch-Kincaid
        try:
            words = len(text.split())
            sentences = len(re.split(r'[.!?]+', text))
            if sentences == 0:
                sentences = 1
            avg_words_per_sentence = words / sentences
            reading_level = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * (sum(1 for c in text if c in 'aeiouAEIOU') / words))
        except:
            reading_level = None
        
        return {
            "title": content["title"],
            "entities": relevant_entities[:15],
            "keywords": keywords,
            "sentiment": sentiment,
            "word_freq": word_freq,
            "reading_level": reading_level,
            "source_type": content["source"],
            "content_length": len(text)
        }
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {"error": f"Analysis error: {str(e)}"}

def generate_summary(analysis):
    """Generate summary using Groq's API with improved error handling and fallbacks"""
    if "error" in analysis:
        return f"⚠️ Error: {analysis['error']}"
    
    if not client:
        # Fallback summary without API
        return create_fallback_summary(analysis)
    
    system_prompt = """You are a senior news analyst. Generate a concise 3-paragraph summary that:
1. Highlights main topics and context based on the provided analysis
2. Analyzes sentiment implications and overall tone
3. Identifies key entities and relationships

Use a journalistic tone and keep it under 200 words. Focus on factual analysis rather than opinion.
Respond ONLY with the final summary text without any formatting, markers, or additional commentary."""
    
    # Create a more comprehensive prompt with all available data
    user_prompt = f"""Analysis Results for: {analysis.get('title', 'Untitled Article')}
- Overall Sentiment: {analysis.get('sentiment', 0):.2f} (-1 negative to +1 positive)
- Content Length: {analysis.get('content_length', 0)} characters
- Key Entities: {', '.join([f"{e[0]} ({e[1]})" for e in analysis.get('entities', [])[:10]])}
- Top Keywords: {', '.join([f"{k[0]} ({k[1]})" for k in analysis.get('keywords', [])[:10]])}
- Frequent Terms: {', '.join([f"{k} ({v})" for k,v in analysis.get('word_freq', [])])}
- Reading Level: {analysis.get('reading_level', 'Unknown')}"""

    # Add retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="deepseek-r1-distill-llama-70b",
                temperature=0.7,
                max_tokens=400,
                top_p=0.9,
            )
            
            raw_summary = chat_completion.choices[0].message.content
            
            # Clean the summary
            cleaned_summary = raw_summary.strip()
            for marker in ["<think>", "</think>", "</s>", "```", "Summary:", "SUMMARY:"]:
                if marker in cleaned_summary:
                    parts = cleaned_summary.split(marker)
                    cleaned_summary = ''.join(parts)
            
            cleaned_summary = cleaned_summary.strip()
            
            # Ensure valid output
            if not cleaned_summary or len(cleaned_summary) < 50:
                if attempt < max_retries - 1:
                    logger.warning("Empty or short summary received. Retrying...")
                    time.sleep(1)  # Add a small delay before retry
                    continue
                return create_fallback_summary(analysis)
                
            return cleaned_summary
            
        except Exception as e:
            logger.error(f"Generation error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Exponential backoff
            else:
                return create_fallback_summary(analysis)

def create_fallback_summary(analysis):
    """Create a basic summary without using the LLM API"""
    try:
        title = analysis.get('title', 'Untitled Article')
        sentiment_desc = "positive" if analysis.get('sentiment', 0) > 0.1 else "negative" if analysis.get('sentiment', 0) < -0.1 else "neutral"
        
        entities = [e[0] for e in analysis.get('entities', [])[:5]]
        entities_text = ", ".join(entities) if entities else "No major entities identified"
        
        keywords = [k[0] for k in analysis.get('keywords', [])[:5]]
        keywords_text = ", ".join(keywords) if keywords else "No significant keywords identified"
        
        summary = f"""This article titled "{title}" presents {sentiment_desc} content focusing on {keywords_text}.

The key entities mentioned include {entities_text}. The overall tone is {sentiment_desc} with a sentiment score of {analysis.get('sentiment', 0):.2f}.

This analysis was performed using automated text processing tools and reflects the main topics and entities detected in the article."""
        
        return summary
    except Exception as e:
        logger.error(f"Fallback summary generation failed: {str(e)}")
        return "Unable to generate summary due to processing errors."

def process_url(url):
    """Complete processing pipeline with timing and error handling"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing URL: {url}")
        analysis = semantic_analyzer(url)
        
        if "error" in analysis:
            return f"⚠️ {analysis['error']}"
        
        summary = generate_summary(analysis)
        
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        return summary
    except Exception as e:
        logger.error(f"Processing pipeline error: {str(e)}")
        return f"⚠️ An unexpected error occurred: {str(e)}"

# Improved Gradio interface
def create_interface():
    with gr.Blocks(title="Advanced News Analyzer") as demo:
        gr.Markdown("# 📰 News Webpage Semantic Analysis & Summary")
        gr.Markdown("""Enter a URL to extract and analyze news content. The system will:
1. Extract the main content from various news sites
2. Perform entity and keyword analysis
3. Generate a concise summary""")
        
        with gr.Row():
            url_input = gr.Textbox(
                label="News URL", 
                placeholder="Enter a complete news URL (e.g., https://www.example.com/article)"
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Row():
            output = gr.Textbox(label="Analysis Summary", show_copy_button=True, lines=10)
            
        analyze_btn.click(fn=process_url, inputs=url_input, outputs=output)
        
        gr.Markdown("""
        ### Notes:
        - Processing time varies depending on the article length and complexity
        - For best results, use direct links to specific news articles
        - Supports a wide range of news sites with adaptive content extraction
        """)
        
    return demo

# Main execution
if __name__ == "__main__":
    # Add dependency checks
    missing_deps = []
    try:
        import newspaper
    except ImportError:
        missing_deps.append("newspaper3k")
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}. Installing...")
        os.system(f"pip install {' '.join(missing_deps)}")
        logger.info("Dependencies installed. Restarting may be required for full functionality.")
    
    # Launch the interface
    port = int(os.getenv("PORT", 7860))
    demo = create_interface()
    demo.launch(
        server_port=port,
        share=False,  # Disable sharing for production
        show_error=True
    )