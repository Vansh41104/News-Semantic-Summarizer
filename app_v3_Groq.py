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
from urllib.parse import urlparse
import logging
import time
import sys

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
    """Extract main content using BeautifulSoup with enhanced extraction"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract title
        title = soup.title.text.strip() if soup.title else ""
        
        # Remove boilerplate elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'form', 'noscript']):
            element.decompose()
            
        # Try to find main content with common news site patterns
        main_content = None
        content_candidates = [
            soup.find('article'),
            soup.find('main'),
            soup.find(id=re.compile('article|content|main|story', re.I)),
            soup.find(class_=re.compile('article|content|main|story', re.I)),
            soup.find(role='main')
        ]
        
        for candidate in content_candidates:
            if candidate and len(' '.join(candidate.stripped_strings)) > 500:
                main_content = candidate
                break
        
        if not main_content:
            paragraphs = soup.find_all('p')
            content_paragraphs = [p.text for p in paragraphs if len(p.text.strip()) > 50]
            
            if content_paragraphs and sum(len(p) for p in content_paragraphs) > 500:
                return {
                    "text": ' '.join(content_paragraphs),
                    "title": title,
                    "source": "paragraph_filter"
                }
            
            divs = soup.find_all('div')
            div_with_most_text = None
            max_text_length = 0
            
            for div in divs:
                text = ' '.join(div.stripped_strings)
                if len(text) > max_text_length:
                    max_text_length = len(text)
                    div_with_most_text = div
            
            if div_with_most_text and max_text_length > 500:
                return {
                    "text": ' '.join(div_with_most_text.stripped_strings),
                    "title": title,
                    "source": "largest_div"
                }
        else:
            return {
                "text": ' '.join(main_content.stripped_strings),
                "title": title,
                "source": "main_content_container"
            }
        
        body_text = ' '.join(soup.body.stripped_strings) if soup.body else ""
        return {
            "text": body_text,
            "title": title,
            "source": "body_fallback"
        }
    except Exception as e:
        logger.error(f"Content extraction failed: {str(e)}")
        return {"error": f"Content extraction error: {str(e)}"}

def semantic_analyzer(url):
    """Improved semantic analysis with content filtering and robust error handling"""
    if not is_valid_url(url):
        return {"error": "Invalid URL format. Please enter a complete URL including http:// or https://"}
    
    try:
        content = extract_main_content(url)
        
        if "error" in content:
            return content
            
        text = content["text"]
        
        if not text or len(text) < 200:
            return {"error": "Insufficient content extracted from the URL. This may not be a news article."}
        
        if nlp:
            doc = nlp(text[:100000])
            relevant_entities = [
                (ent.text, ent.label_) 
                for ent in doc.ents
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'LOC', 'FAC', 'WORK_OF_ART', 'LAW']
            ]
            
            keywords = Counter([
                token.lemma_.lower() 
                for token in doc
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] 
                and not token.is_stop
                and len(token.text) > 2
            ]).most_common(15)
            
            word_freq = Counter([
                token.text.lower() 
                for token in doc 
                if token.is_alpha and not token.is_stop and len(token.text) > 2
            ]).most_common(10)
        else:
            relevant_entities = []
            keywords = []
            word_freq = Counter()
            for word in text.split():
                if len(word) > 3:
                    word_freq[word.lower()] += 1
            word_freq = word_freq.most_common(10)
        
        try:
            sentiment = TextBlob(text).sentiment.polarity
        except:
            sentiment = 0
            
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

def generate_summary(analysis):
    """Generate summary using Groq's API with improved error handling and fallbacks"""
    if "error" in analysis:
        return f"⚠️ Error: {analysis['error']}"
    
    if not client:
        return create_fallback_summary(analysis)
    
    system_prompt = """You are a senior news analyst. Generate a concise 3-paragraph summary that:
1. Highlights main topics and context based on the provided analysis
2. Analyzes sentiment implications and overall tone
3. Identifies key entities and relationships

Use a journalistic tone and keep it under 200 words. Focus on factual analysis rather than opinion.
Respond ONLY with the final summary text without any formatting, markers, or additional commentary."""
    
    user_prompt = f"""Analysis Results for: {analysis.get('title', 'Untitled Article')}
- Overall Sentiment: {analysis.get('sentiment', 0):.2f} (-1 negative to +1 positive)
- Content Length: {analysis.get('content_length', 0)} characters
- Key Entities: {', '.join([f"{e[0]} ({e[1]})" for e in analysis.get('entities', [])[:10]])}
- Top Keywords: {', '.join([f"{k[0]} ({k[1]})" for k in analysis.get('keywords', [])[:10]])}
- Frequent Terms: {', '.join([f"{k} ({v})" for k,v in analysis.get('word_freq', [])])}
- Reading Level: {analysis.get('reading_level', 'Unknown')}"""

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
            cleaned_summary = raw_summary.strip()
            for marker in ["<think>", "</think>", "</s>", "```", "Summary:", "SUMMARY:"]:
                if marker in cleaned_summary:
                    parts = cleaned_summary.split(marker)
                    cleaned_summary = ''.join(parts)
            cleaned_summary = cleaned_summary.strip()
            
            if not cleaned_summary or len(cleaned_summary) < 50:
                if attempt < max_retries - 1:
                    logger.warning("Empty or short summary received. Retrying...")
                    time.sleep(1)
                    continue
                return create_fallback_summary(analysis)
                
            return cleaned_summary
            
        except Exception as e:
            logger.error(f"Generation error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return create_fallback_summary(analysis)

def process_url(url):
    """Complete processing pipeline with timing and error handling"""
    start_time = time.time()
    try:
        logger.info(f"Processing URL: {url}")
        analysis = semantic_analyzer(url)
        
        if "error" in analysis:
            return f"⚠️ {analysis['error']}", None, None, None, None, None
        
        summary = generate_summary(analysis)
        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        
        entities_html = ""
        if analysis.get("entities"):
            for entity, entity_type in analysis.get("entities")[:10]:
                entities_html += f"<span class='entity-pill'>{entity} <small>{entity_type}</small></span>"
        
        keywords_html = ""
        if analysis.get("keywords"):
            for keyword, count in analysis.get("keywords")[:10]:
                keywords_html += f"<span class='keyword-pill'>{keyword} <small>{count}</small></span>"
        
        sentiment = analysis.get("sentiment", 0)
        sentiment_class = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
        sentiment_color = "#047857" if sentiment > 0.1 else "#b91c1c" if sentiment < -0.1 else "#475569"
        
        # Updated sentiment HTML with inline styles for better visibility
        sentiment_html = f"""
        <div class="sentiment-meter {sentiment_class}" style="background-color: rgba(0, 0, 0, 0.05); border: 1px solid {sentiment_color};">
            <div class="sentiment-label" style="color: {sentiment_color}; font-weight: bold; font-size: 1.1rem;">{sentiment_class.capitalize()}</div>
            <div class="sentiment-value" style="color: {sentiment_color}; font-weight: bold; font-size: 1.8rem;">{sentiment:.2f}</div>
        </div>
        """
        
        reading_level = analysis.get("reading_level")
        if reading_level:
            if reading_level > 90:
                reading_desc = "Very Easy"
            elif reading_level > 80:
                reading_desc = "Easy"
            elif reading_level > 70:
                reading_desc = "Fairly Easy"
            elif reading_level > 60:
                reading_desc = "Standard"
            elif reading_level > 50:
                reading_desc = "Fairly Difficult"
            elif reading_level > 30:
                reading_desc = "Difficult"
            else:
                reading_desc = "Very Difficult"
            
            # Updated reading level HTML with inline styles for better visibility
            reading_html = f"""
            <div class="reading-level" style="background-color: rgba(0, 0, 0, 0.05); border: 1px solid #4338ca;">
                <div class="reading-score" style="color: #4338ca; font-weight: bold; font-size: 1.8rem;">{reading_level:.1f}</div>
                <div class="reading-desc" style="color: #4338ca; font-weight: bold; font-size: 1.1rem;">{reading_desc}</div>
            </div>
            """
        else:
            reading_html = "<div class='reading-level' style='color: #4338ca; font-weight: bold;'>Unknown</div>"
        
        title_html = f"<h3>{analysis.get('title', 'Untitled Article')}</h3>"
        return summary, title_html, entities_html, keywords_html, sentiment_html, reading_html
    except Exception as e:
        logger.error(f"Processing pipeline error: {str(e)}")
        return f"⚠️ An unexpected error occurred: {str(e)}", None, None, None, None, None

# Create the Gradio interface with custom CSS and layout
def create_interface():
    custom_css = """
    :root {
        --primary: #2563eb;
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary: #64748b;
        --accent: #f97316;
        --background: #f8fafc;
        --card: #ffffff;
        --card-foreground: #ffffff;
        --border: #e2e8f0;
        --input: #e2e8f0;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --ring: #94a3b8;
    }

    body {
        background-color: var(--background);
        color: var(--card-foreground);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .app-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }
    
    .app-logo {
        font-size: 1.5rem;
        font-weight: 700;
        margin-right: 100px;
        color: var(--primary);
    }
    
    .app-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--card-foreground);
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 1rem;
        color: var(--secondary);
        margin: 0.5rem 0 1.5rem 0;
        line-height: 1.5;
    }
    
    .feature-list {
        margin: 1rem 0;
        padding-left: 1.5rem;
    }
    
    .feature-list li {
        margin-bottom: 0.5rem;
        position: relative;
        padding-left: 0.5rem;
    }
    
    .feature-list li::before {
        content: "•";
        color: var(--primary);
        font-weight: bold;
        position: absolute;
        left: -1rem;
    }
    
    .url-input {
        margin-bottom: 1rem;
    }
    
    .url-input input {
        width: 100%;
        padding: 0.75rem 1rem;
        border: 1px solid var(--input);
        border-radius: 0.375rem;
        font-size: 1rem;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    
    .url-input input:focus {
        outline: none;
        border-color: var(--primary-light);
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    .analyze-button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 0.375rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .analyze-button:hover {
        background-color: var(--primary-dark);
    }
    
    .analyze-button:focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.4);
    }
    
    .result-container {
        margin-top: 2rem;
    }
    
    .result-card {
        background-color: var(--card);
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .result-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #000000;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid rgba(0, 0, 0, 0.1);
        border-radius: 50%;
        border-top-color: var(--primary);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .summary-content {
        line-height: 1.6;
        font-size: 1rem;
        white-space: pre-line;
    }
    
    .metadata-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .metadata-card {
        background-color: var(--card);
        border-radius: 0.375rem;
        padding: 1rem;
        border: 1px solid var(--border);
    }
    
    .metadata-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #101025;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .entity-pill, .keyword-pill {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.1);
        color: var(--primary);
        border-radius: 9999px;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        font-size: 0.875rem;
    }
    
    .entity-pill small, .keyword-pill small {
        opacity: 0.7;
        font-size: 0.75rem;
        margin-left: 0.25rem;
    }
    
    .sentiment-meter {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        padding: 1rem;
        border-radius: 0.375rem;
    }
    
    .sentiment-meter.positive {
        background-color: rgba(16, 185, 129, 0.1);
        color: #047857;
    }
    
    .sentiment-meter.negative {
        background-color: rgba(239, 68, 68, 0.1);
        color: #b91c1c;
    }
    
    .sentiment-meter.neutral {
        background-color: rgba(100, 116, 139, 0.1);
        color: #475569;
    }
    
    .sentiment-label {
        font-weight: 700;
        margin-bottom: 0.25rem;
        text-shadow: 0px 0px 1px rgba(255, 255, 255, 0.8);
    }
    
    .sentiment-value {
        font-size: 1.5rem;
        font-weight: 800;
        text-shadow: 0px 0px 1px rgba(255, 255, 255, 0.8);
    }
    
    .reading-level {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        padding: 1rem;
        border-radius: 0.375rem;
        background-color: rgba(99, 102, 241, 0.1);
        color: #4338ca;
    }
    
    .reading-score {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        text-shadow: 0px 0px 1px rgba(255, 255, 255, 0.8);
    }
    
    .reading-desc {
        font-weight: 700;
        text-shadow: 0px 0px 1px rgba(255, 255, 255, 0.8);
    }
    
    .info-section {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
    }
    
    .info-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .info-list li {
        padding: 0.5rem 0;
        display: flex;
        align-items: center;
    }
    
    .info-list li::before {
        content: "•";
        color: var(--primary);
        margin-right: 0.5rem;
    }
    
    .footer {
        margin-top: 2rem;
        text-align: center;
        font-size: 0.875rem;
        color: var(--secondary);
    }
    
    .footer a {
        color: var(--primary);
        text-decoration: none;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    .error-message {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--error);
        padding: 1rem;
        border-radius: 0.375rem;
        color: #b91c1c;
        margin: 1rem 0;
    }
    
    @media (max-width: 768px) {
        .metadata-grid {
            grid-template-columns: 1fr;
        }
        
        .app-title {
            font-size: 1.5rem;
        }
    }
    """

    with gr.Blocks(css=custom_css, title="News Semantic Analyzer") as demo:
        with gr.Column():
            with gr.Row(elem_classes="app-header"):
                gr.HTML("<div class='app-logo'>📰</div>")
                gr.HTML("<h1 class='app-title'>News Semantic Analyzer</h1>")
            
            gr.HTML("<p class='app-subtitle'>Analyze news articles with advanced NLP techniques to extract key insights, identify entities, and generate concise summaries.</p>")
            
            with gr.Row():
                with gr.Column(scale=4):
                    url_input = gr.Textbox(
                        label="News URL", 
                        placeholder="Enter a complete news URL (e.g., https://www.example.com/article)",
                        elem_classes="url-input"
                    )
                with gr.Column(scale=1):
                    analyze_btn = gr.Button("Analyze Article", variant="primary", elem_classes="analyze-button")
            
            with gr.Row(visible=False) as loading_indicator:
                gr.HTML('<div class="loading-animation"><div class="spinner"></div></div>')
            
            with gr.Column(visible=False) as results_container:
                title_html = gr.HTML(elem_classes="article-title")
                with gr.Column(elem_classes="result-card"):
                    gr.HTML("<div class='result-title'>Executive Summary</div>")
                    summary_output = gr.Textbox(
                        label="", 
                        show_copy_button=True, 
                        lines=6,
                        elem_classes="summary-content"
                    )
                with gr.Row(elem_classes="metadata-grid"):
                    with gr.Column(elem_classes="metadata-card"):
                        gr.HTML("<div class='metadata-title'>Key Entities</div>")
                        entities_html = gr.HTML()
                    with gr.Column(elem_classes="metadata-card"):
                        gr.HTML("<div class='metadata-title'>Top Keywords</div>")
                        keywords_html = gr.HTML()
                with gr.Row(elem_classes="metadata-grid"):
                    with gr.Column(elem_classes="metadata-card"):
                        gr.HTML("<div class='metadata-title'>Sentiment Analysis</div>")
                        sentiment_html = gr.HTML()
                    with gr.Column(elem_classes="metadata-card"):
                        gr.HTML("<div class='metadata-title'>Reading Level</div>")
                        reading_level_html = gr.HTML()
            
            with gr.Row(elem_classes="info-section"):
                gr.HTML("""
                <div class="info-title">About this tool</div>
                <ul class="info-list">
                    <li>This tool extracts and analyzes news content using advanced NLP techniques</li>
                    <li>Processing time varies depending on the article length and complexity</li>
                    <li>For best results, use direct links to specific news articles</li>
                    <li>Supports a wide range of news sites with adaptive content extraction</li>
                    <li>All analysis is performed using Python libraries including spaCy, TextBlob, and BeautifulSoup</li>
                </ul>
                
                <div class="footer">
                    Powered by <a href="https://gradio.app" target="_blank">Gradio</a> and <a href="https://spacy.io" target="_blank">spaCy</a>
                </div>
                """)
        
        # Event handlers
        def on_analyze_click(url):
            if not url or url.strip() == "":
                return {
                    loading_indicator: gr.update(visible=False),
                    results_container: gr.update(visible=False),
                    summary_output: "⚠️ Please enter a valid URL",
                    title_html: "",
                    entities_html: "",
                    keywords_html: "",
                    sentiment_html: "",
                    reading_level_html: ""
                }
            return {
                loading_indicator: gr.update(visible=True),
                results_container: gr.update(visible=False)
            }
            
        def after_processing(summary, title, entities, keywords, sentiment, reading_level):
            if summary and summary.startswith("⚠️"):
                return {
                    loading_indicator: gr.update(visible=False),
                    results_container: gr.update(visible=True),
                    summary_output: summary,
                    title_html: "<div class='error-message'>Analysis could not be completed</div>",
                    entities_html: "",
                    keywords_html: "",
                    sentiment_html: "",
                    reading_level_html: ""
                }
            else:
                return {
                    loading_indicator: gr.update(visible=False),
                    results_container: gr.update(visible=True),
                    summary_output: summary,
                    title_html: title,
                    entities_html: entities,
                    keywords_html: keywords,
                    sentiment_html: sentiment,
                    reading_level_html: reading_level
                }
        
        analyze_btn.click(
            fn=on_analyze_click,
            inputs=[url_input],
            outputs=[loading_indicator, results_container]
        ).then(
            fn=process_url,
            inputs=[url_input],
            outputs=[summary_output, title_html, entities_html, keywords_html, sentiment_html, reading_level_html]
        ).then(
            fn=after_processing,
            inputs=[summary_output, title_html, entities_html, keywords_html, sentiment_html, reading_level_html],
            outputs=[loading_indicator, results_container, summary_output, title_html, entities_html, keywords_html, sentiment_html, reading_level_html]
        )
        
    return demo

demo = create_interface()

# Launch the app
demo.launch(share=False)

if __name__ == "__main__":
    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8001, help="Port to run the Gradio app on")
        parser.add_argument("--share", action="store_true", help="Create a shareable link")
        args = parser.parse_args()
        demo.launch(server_name="0.0.0.0", server_port=8001, share=False)
    except KeyboardInterrupt:
        print("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)
