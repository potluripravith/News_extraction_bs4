# News Sentiment Analysis with Hindi Audio Report

This application extracts and analyzes news articles related to a given company, performs sentiment analysis, and generates an audio report in Hindi. The system provides valuable insights by comparing coverage across multiple news sources.

## Features

- **News Extraction**: Scrapes and analyzes multiple news articles from IndianExpress
- **Sentiment Analysis**: Classifies article sentiment as positive, negative, or neutral
- **Topic Extraction**: Identifies key topics from each article
- **Comparative Analysis**: Compares sentiment and topics across different news sources
- **Hindi Translation**: Translates analysis results to Hindi
- **Text-to-Speech**: Generates Hindi audio report of the analysis
- **User-friendly Interface**: Easy-to-use Streamlit web interface

## Architecture

The application follows a client-server architecture:

- **Frontend**: Streamlit web application for user interaction
- **Backend**: FastAPI service for data processing and analysis
- **Communication**: REST API between frontend and backend
- **Data Processing**: NLP pipeline for analysis and audio generation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news-sentiment-analysis.git
   cd news-sentiment-analysis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('averaged_perceptron_tagger')
   ```

5. Download required SpaCy model:
   ```bash
   python -m spacy download en_core_web_lg
   ```

## Usage

1. Start the FastAPI backend:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

2. In a new terminal, start the Streamlit frontend:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

4. Enter a company name and click "Analyze"

## API Documentation

The backend provides the following endpoints:

- `POST /analyze`: Analyzes news articles for a specified company
  - Expects JSON body: `{"company_name": "Tesla", "num_articles": 10}`
  - Returns analysis results with Hindi audio report

- `GET /health`: Health check endpoint to verify API status

## Implementation Details

### Models Used

- **Summarization**: Hugging Face seq2seq model for text summarization
- **Sentiment Analysis**: Transformer-based sentiment classification model
- **Topic Extraction**: TF-IDF vectorization combined with key phrase extraction
- **Text-to-Speech**: Google Text-to-Speech (gTTS) library for Hindi audio generation

### Code Structure

- `app.py`: Streamlit frontend application
- `api.py`: FastAPI backend service
- `utils.py`: Core functionality for article processing and analysis
- `requirements.txt`: List of required Python packages

## Limitations and Assumptions

- Currently only extracts articles from IndianExpress website
- Relies on web scraping which may break if the website structure changes
- Hindi translation quality depends on third-party translation service
- Limited to processing text content (images and videos are ignored)
- Requires internet connection to access news sources

## Deployment

The application is deployed on Hugging Face Spaces:
- [Link to your Hugging Face Spaces deployment]

## Future Improvements

- Support for additional news sources
- More sophisticated topic modeling
- User authentication and saved reports
- Improved sentiment accuracy with fine-tuned models
- Additional language support

## License

[Specify your license here]

## Contributors

[Your name and contact information]
