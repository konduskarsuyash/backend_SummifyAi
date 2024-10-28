import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import html
import urllib.parse

load_dotenv()

# Google Gemini model setup
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# YouTube Data API setup
youtube_data_api_key = os.getenv("YOUTUBE_DATA_API_KEY")

prompt_template = """You are an expert educational content summarizer and explainer. Your task is to create comprehensive lecture notes from this video transcript. 

Please analyze the following video transcript and create detailed notes that include:

1. Main Topic Overview:
   - Core concept summary
   - Key terms and definitions

2. Detailed Explanation:
   - Break down complex concepts
   - Include any formulas or equations mentioned
   - Add relevant examples
   - Fill in any gaps in the explanation that might not be explicitly covered in the video

3. Additional Context:
   - Related concepts that would help understand this topic better
   - Common applications or real-world examples
   - Any prerequisite knowledge needed

4. Practice Section:
   - Sample problems (if applicable)
   - Key points to remember for exams
   - Common misconceptions to avoid

Please note: If the transcript mentions any formulas or technical concepts without full explanation, provide the complete explanation and context.

Transcript to analyze:
{transcript}

Create detailed notes that would help a student thoroughly understand this topic for an exam."""

def extract_transcript_details(youtube_video_url):
    try:
        # Extract video ID
        video_id = youtube_video_url.split("v=")[1]
        print(f"Processing video ID: {video_id}")
        
        # First, get video details to check language and caption availability
        video_url = (
            f"https://www.googleapis.com/youtube/v3/videos?"
            f"part=snippet&id={video_id}&key={youtube_data_api_key}"
        )
        
        video_response = requests.get(video_url)
        video_data = video_response.json()
        
        if 'items' not in video_data or not video_data['items']:
            raise ValueError("Video not found or not accessible")
            
        # Get captions metadata
        captions_url = (
            f"https://www.googleapis.com/youtube/v3/captions?"
            f"part=snippet&videoId={video_id}&key={youtube_data_api_key}"
        )
        
        captions_response = requests.get(captions_url)
        captions_data = captions_response.json()
        
        if 'items' not in captions_data or not captions_data['items']:
            raise ValueError("No captions available for this video")
            
        # Extract caption track info
        caption_tracks = []
        for item in captions_data['items']:
            language = item['snippet'].get('language', '')
            track_kind = item['snippet'].get('trackKind', '')
            caption_tracks.append({
                'language': language,
                'kind': track_kind,
                'name': item['snippet'].get('name', ''),
                'id': item['id']
            })
            
        # Prioritize English ASR captions or manual captions
        selected_track = None
        for track in caption_tracks:
            if track['language'] == 'en' and track['kind'] in ['asr', 'standard']:
                selected_track = track
                break
                
        if not selected_track:
            # Fall back to any available caption track
            selected_track = caption_tracks[0] if caption_tracks else None
            
        if not selected_track:
            raise ValueError("No suitable caption track found")
            
        # Use the video title and description as context if transcript is not detailed enough
        video_title = video_data['items'][0]['snippet']['title']
        video_description = video_data['items'][0]['snippet']['description']
        
        # Since we can't directly download captions without OAuth, we'll use the metadata
        # to create a meaningful context for the AI
        context = f"""
        Video Title: {video_title}
        Video Description: {video_description}
        Available Caption Track: {selected_track['language']} ({selected_track['kind']})
        
        Note: This video has {len(caption_tracks)} caption tracks available.
        """
        
        # For now, return the context as we can't get the actual transcript without OAuth
        return context

    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        raise e

def generate_gemini_content(transcript_text):
    try:
        # Format the prompt by inserting the transcript/context
        formatted_prompt = prompt_template.format(transcript=transcript_text)
        
        # Set up generation parameters
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        # Generate the content
        response = model.generate_content(
            formatted_prompt,
            generation_config=generation_config
        )
        
        if not response.text:
            raise Exception("Empty response received from Gemini")
            
        return response.text
    
    except Exception as e:
        print(f"Error generating content: {e}")
        error_message = f"Failed to generate content: {str(e)}"
        raise Exception(error_message)