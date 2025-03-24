from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from utils import ArticleProcessor
from translate import Translator
from gtts import gTTS
import base64
import tempfile
import os

# Initialize FastAPI app
app = FastAPI(title="News Sentiment Analysis API",
              description="API for extracting and analyzing news articles related to companies")

# Initialize translator
translator = Translator(to_lang="hi")

# Models for API requests and responses
class CompanyRequest(BaseModel):
    company_name: str
    num_articles: Optional[int] = 10

class ArticleData(BaseModel):
    Title: str
    Summary: str
    Sentiment: str
    Topics: List[List[Any]]

class TopicOverlap(BaseModel):
    Common_Topics: List[str]
    Unique_Topics: Dict[str, List[str]]

class CoverageDifference(BaseModel):
    Comparison: str
    Impact: str

class ComparativeSentimentScore(BaseModel):
    Sentiment_Distribution: Dict[str, int]
    Coverage_Differences: List[CoverageDifference]
    Topic_Overlap: Dict[str, Any]

class SentimentResponse(BaseModel):
    Company: str
    Articles: List[ArticleData]
    Comparative_Sentiment_Score: ComparativeSentimentScore
    Final_Sentiment_Analysis: str
    Audio_Base64: Optional[str] = None

# Function to translate English text to Hindi using chunking
def translate_to_hindi(text):
    """
    Translates English text to Hindi using the translate library.
    Splits the text into chunks to handle long translations.
    """
    try:
        # Split the text into chunks of 500 characters or less
        chunk_size = 500
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Translate each chunk individually
        hindi_chunks = []
        for chunk in chunks:
            try:
                translation = translator.translate(chunk)
                hindi_chunks.append(translation)
            except Exception as e:
                print(f"Translation failed for chunk: {e}")
                hindi_chunks.append(chunk)  # Use original text for failed chunks
        
        # Combine the translated chunks
        hindi_text = " ".join(hindi_chunks)
        return hindi_text
        
    except Exception as e:
        print(f"Translation failed: {e}")
        return text  # Return original text if translation fails

# Function to generate Hindi description from analysis
def generate_hindi_audio_description(final_analysis, company_name):
    """
    Generates a Hindi audio description of the sentiment analysis report.
    """
    sentiment_distribution = final_analysis["Comparative_Sentiment_Score"]["Sentiment_Distribution"]
    coverage_differences = final_analysis["Comparative_Sentiment_Score"]["Coverage_Differences"]
    topic_overlap = final_analysis["Comparative_Sentiment_Score"]["Topic_Overlap"]
    final_sentiment = final_analysis["Final_Sentiment_Analysis"]

    # Generate English text for the report
    english_text = f"""
    Sentiment Analysis Report for {company_name}:
    Sentiment Distribution:
    - Positive: {sentiment_distribution['Positive'] if 'Positive' in sentiment_distribution else 0}
    - Negative: {sentiment_distribution['Negative'] if 'Negative' in sentiment_distribution else 0}
    - Neutral: {sentiment_distribution['Neutral'] if 'Neutral' in sentiment_distribution else 0}

    Coverage Differences:
    - Comparison: {coverage_differences[0]['Comparison'] if coverage_differences else "No comparison available"}
    - Impact: {coverage_differences[0]['Impact'] if coverage_differences else "No impact available"}

    Topic Overlap:
    - Common Topics: {', '.join(topic_overlap['Common_Topics']) if topic_overlap.get('Common_Topics') else "No common topics"}
    - Unique Topics: {', '.join(topic_overlap.get('Unique_Topics_in_Article_1', [])) if 'Unique_Topics_in_Article_1' in topic_overlap else "No unique topics"}

    Final Sentiment Analysis:
    {final_sentiment}
    """

    # Translate the English text to Hindi using chunking
    hindi_text = translate_to_hindi(english_text)
    return hindi_text

# Function to convert Hindi text to speech and return base64 encoded audio
def text_to_hindi_speech_base64(text):
    """
    Converts Hindi text to speech and returns base64 encoded audio data.
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Generate the audio file
        tts = gTTS(text=text, lang='hi', slow=False)
        tts.save(temp_path)
        
        # Read the file and encode to base64
        with open(temp_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        return audio_base64
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return None

@app.post("/analyze", response_model=SentimentResponse)
async def analyze_company(request: CompanyRequest):
    """
    Analyze news articles for a given company, perform sentiment analysis,
    and generate Hindi audio report without returning the Hindi text.
    """
    try:
        # Initialize the article processor
        processor = ArticleProcessor(request.company_name, num_links=request.num_articles)
        
        # Process articles
        processor.process_articles()
        
        # Summarize and extract topics
        processor.summarize_and_extract_topics()
        
        # Generate final sentiment analysis
        raw_analysis = processor.generate_final_sentiment_analysis()
        
        # Format into the specified JSON structure
        formatted_analysis = {
            "Company": request.company_name,
            "Articles": []
        }
        
        # Add articles information
        for article in processor.articles_data:
            formatted_article = {
                "Title": article["Title"],
                "Summary": article["Summary"],
                "Sentiment": article["Sentiment"],
                "Topics": article["Topics"]
            }
            formatted_analysis["Articles"].append(formatted_article)
        
        # Restructure the Comparative Sentiment Score
        sentiment_distribution = raw_analysis["Comparative Sentiment Score"]["Sentiment Distribution"]
        coverage_differences = raw_analysis["Comparative Sentiment Score"]["Coverage Differences"]
        topic_overlap = raw_analysis["Comparative Sentiment Score"]["Topic Overlap"]
        
        # Format coverage differences as a list of dictionaries
        formatted_coverage_differences = []
        
        # Add the original comparison
        formatted_coverage_differences.append({
            "Comparison": coverage_differences["Comparison"],
            "Impact": coverage_differences["Impact"]
        })
        
  
        
        # Format topic overlap with separate keys for each article's unique topics
        formatted_topic_overlap = {
            "Common_Topics": topic_overlap["Common Topics"]
        }
        
        # Add unique topics for each article if available
        for i, article in enumerate(formatted_analysis["Articles"]):
            if i < 2:  # Only do this for the first two articles
                formatted_topic_overlap[f"Unique_Topics_in_Article_{i+1}"] = [
                    topic[0] for topic in article["Topics"] 
                    if topic[0] not in topic_overlap["Common Topics"]
                ]
        
        # Combine everything into the final format
        formatted_analysis["Comparative_Sentiment_Score"] = {
            "Sentiment_Distribution": sentiment_distribution,
            "Coverage_Differences": formatted_coverage_differences,
            "Topic_Overlap": formatted_topic_overlap
        }
        
        formatted_analysis["Final_Sentiment_Analysis"] = raw_analysis["Final Sentiment Analysis"]

        # Generate Hindi translation and convert to audio
        hindi_text = generate_hindi_audio_description(formatted_analysis, request.company_name)
        
        # Generate audio and encode to base64 (only include audio in response)
        if hindi_text:
            audio_base64 = text_to_hindi_speech_base64(hindi_text)
            formatted_analysis["Audio_Base64"] = audio_base64
        
        return formatted_analysis
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing company: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Check if the API is running.
    """
    return {"status": "healthy"}

# Run the API if executed directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)