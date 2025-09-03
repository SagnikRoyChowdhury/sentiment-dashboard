# 📊 Sentiment Pulse Dashboard

## Project Overview
**Sentiment Pulse Dashboard** is an interactive web application that fetches recent news headlines and analyzes their sentiment in real time.  
It provides clear visualizations of positive, negative, and neutral sentiment, helping users understand the overall "mood" around a topic, brand, or event.

**Key Features:**
- Fetches headlines from **Google News RSS** or **NewsAPI.org**.
- Two sentiment analysis options:
  - **VADER:** Fast and lightweight
  - **RoBERTa (CardiffNLP):** More accurate for social media / news text
- Interactive **pie/donut charts** and metrics
- Displays **recent mentions** with sentiment badges
- Allows **CSV download** of analyzed data
- Provides **quick insights** and representative positive/negative headlines

**Screenshot:**  
<img width="1909" height="925" alt="image" src="https://github.com/user-attachments/assets/1f3510be-af01-4456-abbe-7db057e59c72" />

---

## Data Sources
- **Google News RSS:** Fetches real-time news headlines by query  
- **NewsAPI.org:** Alternative source with API key support (`NEWSAPI_KEY` environment variable)

**Rationale:**  
Both sources provide current, relevant news coverage that captures public sentiment, making them ideal for testing real-world sentiment analysis.

---

## Sentiment Analysis Model
- **VADER:** 
  - Lexicon-based sentiment analyzer from NLTK  
  - Lightweight, fast, ideal for general headlines  
- **CardiffNLP RoBERTa (twitter-roberta-base-sentiment-latest):**
  - Transformer-based 3-class sentiment model (negative, neutral, positive)  
  - More accurate for nuanced social media and news text  
- Users can select the model in the sidebar depending on speed vs. accuracy needs.

---

## Local Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-dashboard.git
cd sentiment-dashboard
