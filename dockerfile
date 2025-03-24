FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab');nltk.download('averaged_perceptron_tagger_eng')"

# Copy the rest of the application
COPY . .

# Expose the ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Create a script to run both the API and Streamlit app


# Set environment variable for the API URL in the Streamlit app
ENV API_BASE_URL=http://localhost:8000

# Command to run when the container starts
CMD ["/start.sh"]
