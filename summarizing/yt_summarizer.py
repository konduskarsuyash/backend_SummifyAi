import os
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Google Gemini model setup
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Updated prompt template with clear structure and instructions
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
        video_id = youtube_video_url.split("v=")[1]
        print(f"Extracting transcript for video ID: {video_id}")
        
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_list])
        
        print("Transcript extracted successfully")
        print(f"Transcript length: {len(transcript_text)} characters")
        return transcript_text

    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        raise e

def generate_gemini_content(transcript_text):
    try:
        # Format the prompt by inserting the transcript
        formatted_prompt = prompt_template.format(transcript=transcript_text)
        
        # Set up generation parameters
        generation_config = {
            "temperature": 0.7,  # Balanced between creativity and accuracy
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 4096,  # Adjust based on your needs
        }
        
        # Generate the content with safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        response = model.generate_content(
            formatted_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        if not response.text:
            raise Exception("Empty response received from Gemini")
            
        return response.text
    
    except Exception as e:
        print(f"Error generating content: {e}")
        error_message = f"Failed to generate content: {str(e)}"
        raise Exception(error_message)