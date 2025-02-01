import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from textblob import TextBlob
from transformers import pipeline
import gradio as gr

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize optimized text-generation pipeline
summary_pipeline = pipeline(
    "text-generation",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device=0,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=256,
    repetition_penalty=1.1
)

def semantic_analyzer(url):
    """Improved semantic analysis with content filtering"""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}
    
    try:
        soup = BeautifulSoup(response.content, "html.parser")
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        text = ' '.join(soup.stripped_strings)
    except Exception as e:
        return {"error": f"Content parsing error: {str(e)}"}
    
    doc = nlp(text[:100000])  # Limit processing to first 100k characters
    
    # Improved entity filtering
    relevant_entities = [
        (ent.text, ent.label_) 
        for ent in doc.ents
        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT']
    ]
    
    # Keyword extraction with POS filtering
    keywords = [
        token.lemma_.lower() 
        for token in doc
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop
    ][:20]
    
    return {
        "entities": relevant_entities[:15],
        "keywords": list(set(keywords))[:10],  # Deduplicate
        "sentiment": TextBlob(text).sentiment.polarity,
        "word_freq": Counter([token.text.lower() for token in doc if token.is_alpha])
    }

def generate_summary(analysis):
    """Optimized summary generation with better prompt engineering"""
    if "error" in analysis:
        return f"⚠️ Error: {analysis['error']}"
    
    prompt = f"""[INST] Generate a concise, insightful news summary based on these analysis results:
    
- Overall Sentiment: {analysis['sentiment']:.2f} (range: -1 to 1)
- Key Entities: {', '.join([f"{e[0]} ({e[1]})" for e in analysis['entities']])}
- Top Keywords: {', '.join(analysis['keywords'])}
- Frequent Terms: {', '.join([f"{k} ({v})" for k,v in analysis['word_freq'].most_common(5)])}

Write a 3-paragraph summary that:
1. Highlights main topics and context
2. Analyzes sentiment implications
3. Identifies key entities and their relationships
Use journalistic tone, avoid markdown, and keep under 200 words.
[/INST] Summary:"""
    
    try:
        response = summary_pipeline(
            prompt,
            return_full_text=False,
            clean_up_tokenization_spaces=True
        )
        summary = response[0]['generated_text'].strip()
        # Post-processing
        summary = summary.split("</s>")[0].replace("\n", " ").strip()
        return summary
        
    except Exception as e:
        return f"⚠️ Generation Error: {str(e)}"

def process_url(url):
    """Streamlined processing pipeline"""
    analysis = semantic_analyzer(url)
    return generate_summary(analysis)

# Gradio interface with plaintext output
iface = gr.Interface(
    fn=process_url,
    inputs=gr.Textbox(label="News URL", placeholder="Enter valid news website URL..."),
    outputs=gr.Textbox(label="Analysis Summary", show_copy_button=True),
    title="Advanced News Analyzer",
    description="Semantic analysis and AI-powered summary of news articles",
    allow_flagging="never",
    examples=[
        ["https://indianexpress.com/article/opinion/columns/union-budget-2025-nilesh-shah-9245264/"]
    ]
)

if __name__ == "__main__":
    iface.launch(server_port=7860,share=True)