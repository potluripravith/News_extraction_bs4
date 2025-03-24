import streamlit as st
import base64
import requests
import json
import os
import io

# Streamlit App
def main():
    st.title("Company News Sentiment Analysis with Hindi Audio Report")
    st.write("Enter a company name to fetch news articles, analyze sentiment, and generate a Hindi audio report.")

    # API endpoint
    API_ENDPOINT = "http://localhost:8000/analyze"  # Change this if your API is hosted elsewhere
    
    # Check API health
    try:
        health_response = requests.get("http://localhost:8000/health")
        if health_response.status_code == 200:
            st.success("API connection successful!")
        else:
            st.error("API is not responding correctly. Please check if the API server is running.")
            return
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please make sure the API server is running at http://localhost:8000")
        return

    # Input for company name
    company_name = st.text_input("Enter the company name (e.g., Tesla):")
    num_articles = st.slider("Number of articles to analyze", min_value=1, max_value=20, value=10)

    if st.button("Analyze"):
        if company_name:
            with st.spinner(f"Fetching and analyzing news articles for {company_name}..."):
                try:
                    # Call the API
                    response = requests.post(
                        API_ENDPOINT,
                        json={"company_name": company_name, "num_articles": num_articles}
                    )
                    
                    if response.status_code == 200:
                        # Get the analysis results
                        analysis_results = response.json()
                        
                        # Display the results in English
                        st.subheader("Sentiment Analysis Report (English)")
                        st.json(analysis_results)
                        
                        # Create a downloadable JSON file
                        json_str = json.dumps(analysis_results, indent=2, ensure_ascii=False)
                        json_bytes = json_str.encode("utf-8")
                        
                        st.download_button(
                            label="Download JSON Report",
                            data=json_bytes,
                            file_name=f"{company_name}_analysis.json",
                            mime="application/json"
                        )
                        
                        # Display Hindi translation
                        if "Hindi_Text" in analysis_results and analysis_results["Hindi_Text"]:
                            st.subheader("Hindi Translation of Sentiment Analysis Report")
                            st.write(analysis_results["Hindi_Text"])
                            
                            # Display audio player if audio data is available
                            if "Audio_Base64" in analysis_results and analysis_results["Audio_Base64"]:
                                st.subheader("Listen to the Hindi Audio Report")
                                st.write("Click the play button below to listen to the Hindi audio report.")
                                
                                # Display audio player using base64 data
                                audio_data = base64.b64decode(analysis_results["Audio_Base64"])
                                st.audio(audio_data, format="audio/mp3")
                                
                                # Provide download button for audio
                                st.download_button(
                                    label="Download Hindi Audio Report",
                                    data=io.BytesIO(audio_data),
                                    file_name=f"{company_name}_hindi_audio_report.mp3",
                                    mime="audio/mp3"
                                )
                            else:
                                st.warning("No audio data available from the API.")
                        else:
                            st.warning("No Hindi translation available from the API.")
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
        else:
            st.warning("Please enter a company name.")

# Function to show visualization of sentiment distribution
def display_sentiment_chart(analysis_results):
    if "Comparative Sentiment Score" in analysis_results and "Sentiment Distribution" in analysis_results["Comparative Sentiment Score"]:
        sentiment_data = analysis_results["Comparative Sentiment Score"]["Sentiment Distribution"]
        
        # Convert to format for chart
        chart_data = {
            "Sentiment": list(sentiment_data.keys()),
            "Count": list(sentiment_data.values())
        }
        
        # Display chart
        st.bar_chart(chart_data)

# Run the Streamlit app
if __name__ == "__main__":
    main()