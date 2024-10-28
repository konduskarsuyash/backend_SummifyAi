import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Google Gemini model setup
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# YouTube Data API setup
youtube_data_api_key = os.getenv("YOUTUBE_DATA_API_KEY")

# Prompt to generate notes
prompt_template = """You are a YouTube video lecture explainer. You will take the transcript text
and explain the lecture, providing extra details if necessary, with all relevant information and formulas."""

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("v=")[1]  # Extract video ID

        # Use YouTube Data API to check for captions availability
        captions_url = (
            f"https://www.googleapis.com/youtube/v3/captions?part=snippet&videoId={video_id}"
            f"&key={youtube_data_api_key}"
        )
        response = requests.get(captions_url)
        response_data = response.json()

        # Check if there are captions available
        if 'items' not in response_data or len(response_data["items"]) == 0:
            raise ValueError("No captions available for this video.")

        # Here we would need additional logic to retrieve the actual caption text
        # YouTube Data API v3 does not directly provide captions; you might need a caption URL

        # Placeholder to show how you'd combine available captions (requires actual download of captions)
        transcript = " ".join([item.get("snippet", {}).get("name", "Caption unavailable") for item in response_data["items"]])

        return transcript

    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        raise e

# Generate content summary from the transcript using Google Gemini
def generate_gemini_content(transcript_text):
    try:
        prompt = prompt_template + transcript_text
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise e
