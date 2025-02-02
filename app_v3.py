import os
import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from textblob import TextBlob
import gradio as gr
from groq import Groq
from dotenv import load_dotenv

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def semantic_analyzer(url):
    """Improved semantic analysis with content filtering"""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}
    
    try:
        soup = BeautifulSoup(response.content, "html.parser")
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        text = ' '.join(soup.stripped_strings)
    except Exception as e:
        return {"error": f"Content parsing error: {str(e)}"}
    
    doc = nlp(text[:100000])  # Limit processing to first 100k characters
    
    relevant_entities = [
        (ent.text, ent.label_) 
        for ent in doc.ents
        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT']
    ]
    
    keywords = [
        token.lemma_.lower() 
        for token in doc
        if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop
    ][:20]
    
    return {
        "entities": relevant_entities[:15],
        "keywords": list(set(keywords))[:10],
        "sentiment": TextBlob(text).sentiment.polarity,
        "word_freq": Counter([token.text.lower() for token in doc if token.is_alpha])
    }

def generate_summary(analysis):
    """Generate summary using Groq's API with robust output cleaning"""
    if "error" in analysis:
        return f"⚠️ Error: {analysis['error']}"
    
    system_prompt = """You are a senior news analyst. Generate a concise 3-paragraph summary that:
1. Highlights main topics and context
2. Analyzes sentiment implications
3. Identifies key entities and relationships
Use a journalistic tone and keep it under 200 words.

Respond ONLY with the final summary text without any formatting, markers, or additional commentary."""
    
    user_prompt = f"""Analysis Results:
- Overall Sentiment: {analysis['sentiment']:.2f}
- Key Entities: {', '.join([f"{e[0]} ({e[1]})" for e in analysis['entities']])}
- Top Keywords: {', '.join(analysis['keywords'])}
- Frequent Terms: {', '.join([f"{k} ({v})" for k,v in analysis['word_freq'].most_common(5)])}"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="deepseek-r1-distill-llama-70b",  # Updated model name
            temperature=0.7,
            max_tokens=400,
            top_p=0.9,
        )
        
        raw_summary = chat_completion.choices[0].message.content
        
        # Robust cleaning pipeline
        cleaned_summary = raw_summary.strip()
        for marker in ["<think>", "</think>", "</s>", "```"]:
            cleaned_summary = cleaned_summary.split(marker)[0]
        
        # Ensure valid output
        if not cleaned_summary:
            return "Summary generation failed - please try again with different content"
            
        return cleaned_summary
        
    except Exception as e:
        return f"⚠️ Generation Error: {str(e)}"

def process_url(url):
    """Processing pipeline"""
    analysis = semantic_analyzer(url)
    return generate_summary(analysis)

# Gradio interface
iface = gr.Interface(
    fn=process_url,
    inputs=gr.Textbox(label="News URL", placeholder="Enter valid news website URL..."),
    outputs=gr.Textbox(label="Analysis Summary", show_copy_button=True),
    title="News Webpage Semantic Analysis & Enhanced Summary",
    description=(
        "Enter a URL to perform semantic analysis on its content. "
        "A  summary is generated using NLP techniques and a Hugging Face text-generation model."
    ),
    flagging_mode="never"
    
)

import os
if __name__ == "__main__":
    iface.launch(
    server_port=int(os.getenv("PORT", 7860)),
    share=False  # Disable sharing for production
)