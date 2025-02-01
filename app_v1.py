import requests
from bs4 import BeautifulSoup
import spacy
from collections import Counter
from textblob import TextBlob
from transformers import pipeline
import gradio as gr
import json

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize Hugging Face text-generation pipeline.
# Use 'max_new_tokens' to ensure generated tokens do not conflict with the input length.
hf_pipeline = pipeline(
    "text-generation",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device=0,  # Use GPU (set to -1 for CPU)
    temperature=1,
    top_k=25,
    max_new_tokens=200
)

def semantic_analyzer(url):
    """Perform semantic analysis on webpage content."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    
    soup = BeautifulSoup(response.content, "html.parser")
    text = ' '.join(soup.get_text(strip=True, separator=' ').split())
    
    doc = nlp(text)
    blob = TextBlob(text)
    
    words, lemmas, keywords = [], [], []
    for token in doc:
        if token.is_alpha and not token.is_stop:
            lower_word = token.text.lower()
            words.append(lower_word)
            lemmas.append(token.lemma_.lower())
            # Use nouns as keywords
            if token.pos_ == "NOUN":
                keywords.append(token.text)
    
    # Extract entities as tuples (text, label)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # Frequency counts for words and lemmas
    word_freq = Counter(words)
    lemma_freq = Counter(lemmas)
    
    return {
        "entities": entities,
        "keywords": keywords,
        "sentiment": blob.sentiment.polarity,
        "word_freq": word_freq,
        "lemma_freq": lemma_freq
    }

def generate_summary(analysis):
    """
    Generate an enhanced collaborative summary using LangChain's new chain syntax.
    The summary will be formatted as JSON and include only the summary text.
    """
    # Updated prompt template with delimiters and a "Summary:" header
    prompt_template = """
You are an expert summarization assistant. Based on the following webpage analysis data, generate a unique and insightful summary of approximately 100 words. Do not repeat the analysis details or instructions; only provide the summary text.
---
Overall Sentiment: {sentiment}.
Important Keywords: {keywords}.
Notable Entities: {entities}.
Key Word Frequencies: {word_freq}.
Key Lemma Frequencies: {lemma_freq}.
---
Summary:"""

    # Prepare the analysis data as strings.
    sentiment = f"{analysis['sentiment']:.2f}"
    keywords = ", ".join(analysis["keywords"][:10])
    entities = ", ".join([f"{text} ({label})" for text, label in analysis["entities"]])
    word_freq = ", ".join([f"{word}: {count}" for word, count in analysis["word_freq"].most_common(10)])
    lemma_freq = ", ".join([f"{lemma}: {count}" for lemma, count in analysis["lemma_freq"].most_common(10)])
    
    # Import updated LangChain components from langchain_huggingface
    from langchain.prompts import PromptTemplate
    from langchain_huggingface import HuggingFacePipeline

    # Create the prompt template.
    template = PromptTemplate(
        input_variables=["sentiment", "keywords", "entities", "word_freq", "lemma_freq"],
        template=prompt_template
    )
    
    # Wrap the Hugging Face pipeline with the updated HuggingFacePipeline.
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    # Create the chain using the new syntax (prompt | llm)
    chain = template | llm

    # Invoke the chain with the provided values.
    summary_result = chain.invoke({
        "sentiment": sentiment,
        "keywords": keywords,
        "entities": entities,
        "word_freq": word_freq,
        "lemma_freq": lemma_freq
    })

    # Check if the output is a dict and extract text if so.
    if isinstance(summary_result, dict):
        summary_text = summary_result.get("text", "")
    else:
        summary_text = summary_result
    summary_text = summary_text.strip()
    
    return json.dumps({"summary": summary_text}, indent=4)

def process_url(url):
    """Process the URL: perform semantic analysis and generate a JSON summary."""
    analysis = semantic_analyzer(url)
    if "error" in analysis:
        return json.dumps({"error": analysis["error"]}, indent=4)
    
    return generate_summary(analysis)

# Create the Gradio Interface with JSON summary output.
iface = gr.Interface(
    fn=process_url,
    inputs=gr.Textbox(
        label="Enter News Webpage URL",
        placeholder="Paste News URL here..."
    ),
    outputs=gr.Textbox(
        label="Summary Output",
        lines=10,
    ),
    title="News Webpage Semantic Analysis & Enhanced Summary",
    description=(
        "Enter a URL to perform semantic analysis on its content. "
        "A  summary is generated using NLP techniques and a Hugging Face text-generation model."
    )
)

if __name__ == "__main__":
    iface.launch()
