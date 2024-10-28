import os
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# Google Gemini model setup
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


# Prompt to generate notes
prompt_template="""You are Yotube video lecture explainer . You will be taking the transcript text
and explaining me the lecture as i have exam tommorow u would need to explain me things extra if that is not explained in the video clearly.
The transcript of the video may not be a detail explanation of the video so you have to give me all the necessary information formulas related to the lecture"""

# Extract transcript details from the YouTube video
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("v=")[1]  # Extract video ID from the URL
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine all transcript chunks into one text
        transcript = " ".join([item["text"] for item in transcript_list])

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
