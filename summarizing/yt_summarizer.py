import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# Google Gemini model setup
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# YouTube Data API setup
youtube_data_api_key = os.getenv("YOUTUBE_DATA_API_KEY")

# Prompt to generate notes
prompt_template = """You are a YouTube video lecture explainer. You will be taking the transcript text
and explaining the lecture as if I have an exam tomorrow. Provide extra explanations and details if necessary,
and include all relevant information and formulas."""

# Extract transcript details from the YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("v=")[1]  # Extract video ID from the URL

        # Check for captions availability using YouTube Data API
        captions_url = (
            f"https://www.googleapis.com/youtube/v3/captions?part=snippet&videoId={video_id}"
            f"&key={youtube_data_api_key}"
        )
        response = requests.get(captions_url)
        response_data = response.json()

        if 'items' not in response_data:
            raise ValueError("No captions available for this video.")

        # Retrieve transcript text from available captions
        # (Requires further handling to fetch and format captions into a transcript.)
        # Let's assume 'default' language caption is available.
        transcript = " ".join([item["snippet"]["text"] for item in response_data["items"]])

        return transcript

    except Exception as e:
        raise e

# Generate content summary from the transcript using Google Gemini
def generate_gemini_content(transcript_text):
    try:
        prompt = prompt_template + transcript_text
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise e
