import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
from collections import defaultdict
import requests
from bs4 import BeautifulSoup
import csv
import spacy
import json
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger')
# Load summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Map model labels to human-readable sentiment categories
label_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Load the spaCy model
nlp = spacy.load('en_core_web_lg')

# Function for summarization
def summarize_text(text, max_length=50, min_length=30, model_max_length=1024):
    """
    Summarizes the text using BART model by dynamically adjusting chunk_size.
    """
    # Check if the input text is empty
    if not text.strip():
        return ""  # Return empty string for empty input

    # Calculate dynamic chunk_size based on input text length
    text_length = len(text.split())
    if text_length <= model_max_length:
        chunk_size = text_length  # Use the entire text as a single chunk
    else:
        chunk_size = model_max_length - 100  # Leave room for overlap or safety

    # Split the raw text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    summaries = []
    for chunk in chunks:


        inputs = summarizer.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=model_max_length)
        tokenized_chunk = summarizer.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
        chunk_length = len(tokenized_chunk.split())
        
        if chunk_length <= max_length:
            summaries.append(tokenized_chunk)  # Use the chunk as-is
            continue
        # Skip summarization for very short chunks
        
        # Summarize the chunk with a smaller max_length for individual chunks
        summary = summarizer(
            tokenized_chunk,
            max_length=max_length,  # Use a smaller max_length for individual chunks
            min_length=min_length,
            do_sample=False
        )
        summaries.append(summary[0]['summary_text'])
    
    # Combine the summaries into a single summary
    combined_summary = " ".join(summaries)
    
    return combined_summary

# Normal preprocessing function
def preprocess_text(text):
    """
    Preprocesses the text by converting to lowercase, removing punctuation,
    and lemmatizing while removing stopwords.
    """
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Preprocessing function to remove company names and extract noun phrases
def normalize_text(text):
    """
    Normalizes text using spaCy's pipeline:
    - Lowercase
    - Remove stopwords
    - Remove punctuation
    - Remove numbers
    - Remove short words (less than 6 characters)
    - Remove company names (ORG entities)
    """
    doc = nlp(text)
    normalized_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) >= 7:
            if token.ent_type_ != "ORG":  # Remove organization names
                normalized_tokens.append(token.lemma_.lower())
    return " ".join(normalized_tokens)

def extract_noun_phrases(text):
    """
    Extract noun phrases using chunking.
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
    cp = RegexpParser(grammar)
    tree = cp.parse(pos_tags)
    noun_phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        np = " ".join(word for word, tag in subtree.leaves())
        noun_phrases.append(np)
    return " ".join(noun_phrases)


def extract_key_terms_improved(text, top_n=3, score_threshold = 0.2):
    """
    Extract key terms for a single article with improved filtering and relevance.
    - Excludes terms with a score less than 2.
    - Excludes terms containing the word "company".
    """
    # Preprocess text
    processed_text = normalize_text(text)
    filtered_text = extract_noun_phrases(processed_text)

    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english', max_df=1.0, min_df=1, ngram_range=(1, 2))
    X = vectorizer.fit_transform([filtered_text])
    feature_names = vectorizer.get_feature_names_out()
    term_scores = X.toarray()[0]

    # Collect term information
    term_info = []
    for idx, score in enumerate(term_scores):
        if score > 0:  # Only include terms with a score of 2 or higher
            term = feature_names[idx]
            word_count = len(term.split())
            boosted_score = score * (1.7 if word_count == 2 else 1.0)
            term_info.append((term, boosted_score, word_count))

    # Sort terms by boosted score
    term_info.sort(key=lambda x: x[1], reverse=True)

    # Filter terms
    filtered_terms = []
    component_words = set()
    for term, score, word_count in term_info:
        # Skip terms containing the word "company"
        if "company" in term.lower():
            continue

        # Normalize term order for 2-word phrases
        if word_count == 2:
            term = " ".join(sorted(term.split()))

        # Skip redundant or irrelevant terms
        term_words = set(term.split())
        if term_words.issubset(component_words) and word_count == 1:
            continue  # Skip single-word terms already covered by a phrase
        if word_count == 1 and term in component_words:
            continue  # Skip single-word terms already in the list
        if word_count == 1 and len(term) <= 2:
            continue  # Skip very short single-word terms
        if word_count == 2 and term in filtered_terms:
            continue  # Skip duplicate phrases

        # Add the term to the list
        if  score >= score_threshold:  # Ensure score is between 0.20 and 0.25
            filtered_terms.append((term, score))  # Store term and its score
            component_words.update(term_words)
            if len(filtered_terms) >= top_n:
                break
    

    # Ensure at least one topic is returned
    if not filtered_terms and term_info:
        filtered_terms.append((term_info[0][0], term_info[0][1]))

    return filtered_terms[:top_n]

class ContentAnalyzer:
    @staticmethod
    def is_similar(content1, content2, threshold=0.50):
        """
        Checks if two pieces of content are similar using TF-IDF and cosine similarity.
        """
        content1 = preprocess_text(content1)
        content2 = preprocess_text(content2)
        
        # Vectorize the content using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([content1, content2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        print(f"Similarity Score: {similarity:.2f}")
        return similarity > threshold

class ArticleProcessor:
    def __init__(self, company_name, num_links=10):
        self.company_name = company_name
        self.num_links = num_links
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.analyzer = ContentAnalyzer()
        self.articles_data = []
        self.processed_links = set()
    def analyze_sentiment_distribution(self):
        """
        Calculate the sentiment distribution (Positive, Negative, Neutral) for the collected articles.
        """
        sentiment_distribution = defaultdict(int)
        for article in self.articles_data:
            sentiment = article['Sentiment']
            sentiment_distribution[sentiment] += 1
        return sentiment_distribution

    def analyze_topic_overlap(self):
        """
        Extract and compare topics from the articles.
        - Returns combined common topics across all articles.
        - Returns combined unique topics that are not present in any other article.
        """
        all_topics = [set([topic[0] for topic in article['Topics']]) for article in self.articles_data]
        
        # Find common topics across all articles
        common_topics = set.intersection(*all_topics) if all_topics else set()
        
        # Find unique topics that are not present in any other article
        unique_topics = set()
        for i, topics in enumerate(all_topics):
            other_topics = set.union(*[t for j, t in enumerate(all_topics) if j != i])
            unique_topics.update(topics - other_topics)
        
        return {
            "Common Topics": list(common_topics),
            "Unique Topics": list(unique_topics)
        }

    def compare_coverage_differences(self):
        """
        Identify key differences in the content of the articles and their impact.
        - Groups articles by sentiment (positive/negative/neutral).
        - Summarizes their topics and impacts, referring to articles by their numbers.
        """
        positive_articles = [(i + 1, article) for i, article in enumerate(self.articles_data) if article['Sentiment'] == "Positive"]
        negative_articles = [(i + 1, article) for i, article in enumerate(self.articles_data) if article['Sentiment'] == "Negative"]
        neutral_articles = [(i + 1, article) for i, article in enumerate(self.articles_data) if article['Sentiment'] == "Neutral"]

        # Edge Case 1: No Positive or Negative Articles (Only Neutral)
        if not positive_articles and not negative_articles:
            return {
                "Comparison": "As there is only neutral talk about the company, there is nothing to compare.",
                "Impact": "No articles are impacting the growth of the company; it stays stable."
            }

        # Edge Case 2: Only Positive Articles
        if not negative_articles:
            positive_summary = []
            for article_num, article in positive_articles:
                topics = [topic[0] for topic in article['Topics']]
                positive_summary.append(f"Article {article_num} highlights {self.company_name}'s strong {', '.join(topics)}")
            
            comparison = ", ".join(positive_summary) + ". There is no negative talk about the company."
            impact = f"The positive articles boost confidence in {self.company_name}'s market growth. There is no raise of concern about the company."
            
            return {
                "Comparison": comparison,
                "Impact": impact
            }

        # Edge Case 3: Only Negative Articles
        if not positive_articles:
            negative_summary = []
            for article_num, article in negative_articles:
                topics = [topic[0] for topic in article['Topics']]
                negative_summary.append(f"Article {article_num} discusses {', '.join(topics)} issues")
            
            comparison = f"There is no positive talk about {self.company_name}, while " + ", ".join(negative_summary) + "."
            impact = f"The negative articles raise concerns about future hurdles. As there is no positive talk, it may impact the company's growth."
            
            return {
                "Comparison": comparison,
                "Impact": impact
            }

        # Default Case: Both Positive and Negative Articles
        positive_summary = []
        for article_num, article in positive_articles:
            topics = [topic[0] for topic in article['Topics']]
            positive_summary.append(f"Article {article_num} highlights {self.company_name}'s strong {', '.join(topics)}")
        
        negative_summary = []
        for article_num, article in negative_articles:
            topics = [topic[0] for topic in article['Topics']]
            negative_summary.append(f"Article {article_num} discusses {', '.join(topics)} issues")
        
        comparison = ", ".join(positive_summary) + ", while " + ", ".join(negative_summary) + "."
        impact = f"The {', '.join([f'{article_num}th article' for article_num, _ in positive_articles])}  boost confidence in {self.company_name}'s market growth, while the {', '.join([f'{article_num}th article' for article_num, _ in negative_articles])}  raise concerns about future hurdles."
        
        return {
            "Comparison": comparison,
            "Impact": impact
        }
    def generate_final_sentiment_analysis(self):
        """
        Generate the final sentiment analysis based on the aggregated data.
        - Combines sentiment distribution, coverage differences, and topic overlap.
        - Provides a clear, combined analysis.
        - Returns data in the format needed for JSON export.
        """
        sentiment_distribution = self.analyze_sentiment_distribution()
        coverage_differences = self.compare_coverage_differences()
        topic_overlap = self.analyze_topic_overlap()
        
        final_sentiment = "Neutral"
        if sentiment_distribution['Positive'] > sentiment_distribution['Negative']:
            final_sentiment = "Positive"
        elif sentiment_distribution['Negative'] > sentiment_distribution['Positive']:
            final_sentiment = "Negative"
        
        # Customize final sentiment analysis based on sentiment
        if final_sentiment == "Positive":
            stock_outcome = "Potential stock growth expected."
        elif final_sentiment == "Negative":
            stock_outcome = "Potential stock falls expected."
        else:
            stock_outcome = "Stock is expected to stay about the same."
        
        final_analysis = {
            "Comparative Sentiment Score": {
                "Sentiment Distribution": dict(sentiment_distribution),
                "Coverage Differences": coverage_differences,
                "Topic Overlap": topic_overlap
            },
            "Final Sentiment Analysis": f"The latest news coverage is mostly {final_sentiment.lower()}. {stock_outcome}"
        }
        return final_analysis

    def fetch_article_details(self, article_url):
        """
        Fetches the title and content of an article from its URL.
        """
        response = requests.get(article_url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title = soup.find('h1', class_='native_story_title')
        title = title.get_text(strip=True) if title else "No Title Found"

        # Extract content
        content = ""
        group_elements = soup.find_all(class_='story_details')
        for element in group_elements:
            paragraphs = element.find_all('p')
            if paragraphs:
                content += '\n'.join([p.get_text(strip=True) for p in paragraphs]) + '\n\n'

        return title, content

    def process_articles(self):
        """
        Main method that processes articles and checks for similarity immediately.
        """
        # Start with a search to get links
        search_url = f'https://www.bing.com/news/search?q={self.company_name.replace(" ", "+")} company +site:indianexpress.com'
        
        # We'll make multiple search attempts if needed
        search_attempts = 0
        max_search_attempts = 3
        page_param = 0
        
        while len(self.articles_data) < self.num_links and search_attempts < max_search_attempts:
            # Fetch search results
            if search_attempts > 0:
                # For subsequent searches, try to get different results by using a page parameter
                current_url = f"{search_url}&first={page_param}"
                page_param += 10
            else:
                current_url = search_url
                
            print(f"Searching for articles: {current_url}")
            response = requests.get(current_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get links from search results
            links_found = False
            for result in soup.find_all('a', class_='title'):
                if len(self.articles_data) >= self.num_links:
                    break
                    
                if result and 'href' in result.attrs:
                    link = result['href']
                    
                    # Check if it's an Indian Express link and not already processed
                    if 'indianexpress.com' in link and link not in self.processed_links:
                        self.processed_links.add(link)
                        links_found = True
                        
                        print(f"Checking link: {link}")
                        try:
                            # Fetch article details
                            title, content = self.fetch_article_details(link)
                            
                            # Skip if content is empty
                            if not content.strip():
                                print(f"Empty content found. Skipping: {link}")
                                continue
                                
                            # Check for similarity with existing articles BEFORE adding
                            is_duplicate = False
                            for existing_article in self.articles_data:
                                if self.analyzer.is_similar(content, existing_article['Content']):
                                    print(f"Duplicate content found. Skipping: {link}")
                                    is_duplicate = True
                                    break
                                    
                            # Only add non-duplicate articles
                            if not is_duplicate:
                                self.articles_data.append({
                                    'Title': title,
                                    'Content': content,
                                    'Link': link
                                })
                                print(f"Added unique article: {title}")
                                print(f"Progress: {len(self.articles_data)}/{self.num_links}")
                                
                        except Exception as e:
                            print(f"Error processing link {link}: {str(e)}")
            
            # Increment search attempts and check if we found any new links
            search_attempts += 1
            if not links_found:
                print("No new links found in search results.")
                
        print(f"\nCollected {len(self.articles_data)} unique articles out of requested {self.num_links}.")

    def summarize_and_extract_topics(self):
        """
        Performs summarization, topic extraction, and sentiment analysis on all collected articles.
        """
        for i, article in enumerate(self.articles_data):
            # Summarize the article
            article['Summary'] = summarize_text(article['Content'])
            
            # Extract topics for the article using NMF
            article['Topics'] = extract_key_terms_improved(article['Content'], top_n=3, score_threshold = 0.2)
            
            # Split the content into chunks for sentiment analysis
            max_chunk_length = 500  # Set a safe limit for token count
            content_chunks = [article['Content'][i:i + max_chunk_length] for i in range(0, len(article['Content']), max_chunk_length)]
            
            # Perform sentiment analysis on each chunk and aggregate results
            sentiment_results = []
            for chunk in content_chunks:
                try:
                    sentiment_result = sentiment_analyzer(chunk)[0]
                    sentiment_results.append(sentiment_result)
                except Exception as e:
                    print(f"Error during sentiment analysis for article {i}, chunk: {e}")
            
            # Aggregate sentiment results (e.g., take the majority sentiment)
            if sentiment_results:
                sentiment_label = max(set([label_mapping[result['label']] for result in sentiment_results]), key=[label_mapping[result['label']] for result in sentiment_results].count)
                sentiment_confidence = sum([result['score'] for result in sentiment_results]) / len(sentiment_results)
            else:
                sentiment_label = "Unknown"
                sentiment_confidence = 0.0
            
            article['Sentiment'] = sentiment_label
            article['Sentiment Confidence'] = sentiment_confidence
            
            print(f"Processed article: {article['Title']}")
            print(f"  Topics: {article['Topics']}")
            print(f"  Sentiment: {article['Sentiment']}, Confidence: {article['Sentiment Confidence']:.2f}")


if __name__ == '__main__':
    company_name = input("Enter the company name: ")
    processor = ArticleProcessor(company_name, num_links=10)
    processor.process_articles()
    
    # Perform summarization, topic extraction, and sentiment analysis
    processor.summarize_and_extract_topics()
    
  