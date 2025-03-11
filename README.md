# News Webpage Semantic Analysis & Enhanced Summary

## Overview
This project is a **news webpage semantic analysis tool** that extracts and analyzes key insights from online news articles. It utilizes **NLP (spaCy, TextBlob)** for entity recognition, sentiment analysis, and keyword extraction, while **Groq's AI API** generates a refined summary. The project is wrapped in an easy-to-use **Gradio** web interface.

## Features
- **Web Scraping**: Extracts text content from online news articles while filtering out unnecessary elements (scripts, styles, headers, etc.).
- **Named Entity Recognition (NER)**: Identifies key entities like **organizations, people, locations, and events**.
- **Keyword Extraction**: Highlights relevant keywords based on **lemmatization and part-of-speech tagging**.
- **Sentiment Analysis**: Evaluates the **polarity** of the article's content using **TextBlob**.
- **Word Frequency Analysis**: Computes the most frequent words appearing in the text.
- **AI-Powered Summary Generation**: Uses **Groq's AI API** to generate a **concise, journalistic-style summary**.
- **Gradio UI**: Provides a simple interface for users to input URLs and receive a structured analysis.

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Clone the Repository
```bash
git clone https://github.com/Vansh41104/News_Semantic_Summarizer.git
cd news-semantic-analyzer
```

### Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Set Up API Keys
Create a `.env` file in the project directory and add your **Groq API key**:
```env
GROQ_API_KEY=your_api_key_here
```

## Usage
### Run the Application
```bash
python app.py
```
This will start a local **Gradio** server at `http://127.0.0.1:7860/`, where you can input a news article URL and get the analysis.

## Code Structure
```
📂 news-semantic-analyzer
├── app.py            # Main application script
├── requirements.txt  # Dependencies
├── .env.example      # Environment variable template
├── README.md         # Project documentation
```

## API & Models Used
- **spaCy** (`en_core_web_sm`) - Named Entity Recognition, Tokenization, and NLP processing
- **TextBlob** - Sentiment analysis and text processing
- **BeautifulSoup** - Web scraping for extracting article text
- **Groq API (deepseek-r1-distill-llama-70b)** - AI-generated text summary
- **Gradio** - Web interface for user interaction

## Future Improvements
- **Multilingual Support**
- **Improved Entity Linking** (Cross-referencing with knowledge bases)
- **Topic Categorization** (Classifying articles into predefined topics)

## License
This project is licensed under the MIT License. Feel free to modify and enhance it!

## Deploy
This project is deployed on Render, because of which it takes around a minute to start. ```https://news-semantic-summarizer.onrender.com/```

## Author
Developed by **Vansh Bhatnagar**.

---
Enjoy using **News Webpage Semantic Analysis & Enhanced Summary** 🚀

