from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import YouTubeSummary
from .serializers import YouTubeSummarySerializer,GeneratedQuizSerializer
from .yt_summarizer import extract_transcript_details, generate_gemini_content
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from .pdf_summarizer import pdf_to_images, process_image_with_vision_model,get_vector_store, get_conversational_chain, user_input  # Import your functions
from rest_framework.views import APIView
from .models import PDFSummary,GeneratedQuiz
from rest_framework import response
import logging
from django.core.files.storage import default_storage
from django.contrib.auth.models import User
from .mind_map import get_pdf_text, get_text_chunks, get_vector_store, generate_mindmap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import google.generativeai as genai
from user_profile.models import UserContribution,UserStatistics
import json
import re
import os
import cv2
import base64
from moviepy.editor import VideoFileClip
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from .models import Video
from .serializers import VideoSerializer
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
groq_api_key = os.getenv("GROQ_API_KEY")

class QuizGeneratorView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # Attempt to retrieve the latest PDF summary for the authenticated user
        try:
            pdf_summary = PDFSummary.objects.filter(user=request.user).latest('created_at')
            print(f"Latest PDF Summary found: {pdf_summary}")  # Log the found summary
        except PDFSummary.DoesNotExist:
            print("No PDF summary found for the user.")  # Log if not found
            return Response({"error": "No PDF summary found."}, status=status.HTTP_404_NOT_FOUND)

        system_summary = pdf_summary.summary
        print(f"System Summary: {system_summary}")  # Log the retrieved summary

        text_chunks = self.get_text_chunks(system_summary)
        self.get_vector_store(text_chunks)

        # Generate quiz
        user_question = self.get_quiz_generation_prompt()

        try:
            quiz_json = self.user_input(user_question)
            # Parse the JSON string to ensure it's valid before saving
            quiz_data = json.loads(quiz_json)
            print(quiz_data)
            # Save the generated quiz to the database
            generated_quiz = GeneratedQuiz.objects.create(
                user=request.user,
                pdf_summary=pdf_summary,
                quiz_data=quiz_json
            )

            UserContribution.objects.create(user=request.user, contribution_type='test')
            
            user_statistics = UserStatistics.objects.get(user=request.user)
            user_statistics.quizzes_taken += 1
            user_statistics.save()

            # Return quiz ID and quiz data
            return Response({
                "quiz_id": generated_quiz.id,
                "quiz_data": quiz_data
            }, status=status.HTTP_201_CREATED)
        
        except ValueError as e:
            logger.error(f"Error generating quiz: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get_text_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def get_vector_store(self, text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_quiz_summary")

    def get_quiz_generation_prompt(self):
        return """Generate a quiz based on the provided content in strict JSON format. The quiz should include a minimum of 15  questions, equally distributed across three difficulty levels: easy, medium, and hard, with 5 questions in each difficulty level. Each question type should encourage higher-order thinking skills (application, analysis, synthesis). 

        Here’s an example of the expected output:

        {
            "quiz_id": 8,
            "quiz_data": {
                "multiple_choice_questions": [
                    {
                        "question": "Which activation function is used when we want the output to be in the form of probability?",
                        "options": {
                            "A": "ReLU",
                            "B": "Sigmoidal AF",
                            "C": "Tanh AF",
                            "D": "Softmax"
                        },
                        "correct_option": "B"
                    },
                    {
                        "question": "What is the problem with ReLU?",
                        "options": {
                            "A": "Dying neurons",
                            "B": "Inconsistent predictions",
                            "C": "Constant derivative",
                            "D": "All of the above"
                        },
                        "correct_option": "D"
                    },
                    {
                        "question": "Where is the Softmax function applied?",
                        "options": {
                            "A": "Input layer",
                            "B": "Hidden layer",
                            "C": "Output layer",
                            "D": "All of the above"
                        },
                        "correct_option": "C"
                    },
                    {
                        "question": "Which activation function is used for Convolutional Neural Networks (CNNs) images input?",
                        "options": {
                            "A": "ReLU",
                            "B": "Sigmoidal AF",
                            "C": "Tanh AF",
                            "D": "Softmax"
                        },
                        "correct_option": "A"
                    },
                    {
                        "question": "What is the main advantage of Tanh AF?",
                        "options": {
                            "A": "It is computationally expensive",
                            "B": "It is bipolar",
                            "C": "It is only used in output layers",
                            "D": "It is not used in CNNs"
                        },
                        "correct_option": "B"
                    }
                ],
                "true_or_false_questions": [
                    {
                        "statement": "ReLU is computationally expensive.",
                        "options": {
                            "True": "True",
                            "False": "False"
                        },
                        "correct_option": "False"
                    },
                    {
                        "statement": "Softmax is used for binary classification.",
                        "options": {
                            "True": "True",
                            "False": "False"
                        },
                        "correct_option": "False"
                    },
                    {
                        "statement": "Softmax is always applied to the output layer of a neural network.",
                        "options": {
                            "True": "True",
                            "False": "False"
                        },
                        "correct_option": "True"
                    }
                ]
            }
        }

        Please generate the quiz strictly in this JSON format in code format avoid using backticks at the beginning and at the end without any additional text or explanation.
        """

    logger = logging.getLogger(__name__)

    def extract_and_parse_json(self, text):
        # Find the outermost braces
        start = text.find('{')
        end = text.rfind('}')
        
        if start == -1 or end == -1:
            raise ValueError("No JSON-like structure found in the response.")
        
        # Extract the content between the outermost braces
        json_str = text[start:end+1]
        
        # Clean the extracted JSON string
        cleaned_json = self.clean_json(json_str)
        
        try:
            # Try to parse it
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            # If parsing fails, try to salvage what we can
            return self.salvage_json(cleaned_json)

    def clean_json(self, raw_json):
        # Replace single quotes with double quotes
        raw_json = raw_json.replace("'", '"')
        
        # Ensure all keys are properly quoted
        raw_json = re.sub(r'(\w+)(?=\s*:)', r'"\1"', raw_json)
        
        # Remove any trailing commas before closing brackets or braces
        raw_json = re.sub(r',\s*([\]}])', r'\1', raw_json)
        
        return raw_json

    def salvage_json(self, text):
        # Try to salvage as much of the JSON structure as possible
        questions = []
        
        # Extract multiple choice questions
        mcq_pattern = r'"question"\s*:\s*"([^"]+)".*?"options"\s*:\s*\{([^}]+)\}.*?"correct_option"\s*:\s*"([^"]+)"'
        mcq_matches = re.finditer(mcq_pattern, text, re.DOTALL)
        
        for match in mcq_matches:
            question, options_str, correct_option = match.groups()
            options = {}
            for opt in re.finditer(r'"([A-D])"\s*:\s*"([^"]+)"', options_str):
                options[opt.group(1)] = opt.group(2)
            
            questions.append({
                "question": question,
                "options": options,
                "correct_option": correct_option
            })
        
        # Extract true/false questions
        tf_pattern = r'"statement"\s*:\s*"([^"]+)".*?"options"\s*:\s*\{([^}]+)\}.*?"correct_option"\s*:\s*"(True|False)"'
        tf_matches = re.finditer(tf_pattern, text, re.DOTALL)
        
        for match in tf_matches:
            statement, options_str, correct_option = match.groups()
            
            questions.append({
                "statement": statement,
                "options": {"True": "True", "False": "False"},
                "correct_option": correct_option
            })
        
        if questions:
            return {"questions": questions}
        else:
            raise ValueError("No valid questions could be extracted from the response.")

    def user_input(self, user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index_quiz_summary", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = self.get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        logger.debug(f"Response from chain: {response}")

        # The response is typically a dictionary, and the actual content is often under the 'output_text' key
        quiz_json = response.get("output_text", "")
        if not quiz_json:
            logger.error("Empty response from the language model chain")
            raise ValueError("Generated quiz data is empty.")

        logger.debug(f"Raw Quiz JSON Response: '{quiz_json}'")
        logger.debug(f"Type of raw quiz JSON: {type(quiz_json)}")

        try:
            json_object = self.extract_and_parse_json(quiz_json)
            return json.dumps(json_object)  # Return valid JSON string
        except ValueError as e:
            logger.error(f"Error extracting and parsing JSON: {e}")
            raise ValueError("Generated quiz data is not valid JSON and could not be repaired.")

    def get_conversational_chain(self):
        prompt_template = """
        You are a Quiz generator LLM based on the context.
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = ChatGroq(model_name="llama-3.1-70b-versatile", temperature=0.35)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

    
    
    
class YouTubeSummaryCreateView(generics.CreateAPIView):
    serializer_class = YouTubeSummarySerializer
    permission_classes = [IsAuthenticated]

    def perform_create(self, serializer):
        youtube_url = serializer.validated_data.get('youtube_url')
        
        # Extract transcript from YouTube
        transcript = extract_transcript_details(youtube_url)

        # Generate summary using Google Gemini
        summary = generate_gemini_content(transcript)

        # Save the object in the database with the user
        instance = serializer.save(user=self.request.user, transcript=transcript, summary=summary)
        return instance  # Return the instance for access later

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            # Create the instance and retrieve it
            instance = self.perform_create(serializer)

            # Only return the summary in the response
            summary_response = {
                "summary": instance.summary  # Access the generated summary from the saved instance
            }
            
            user_statistics = UserStatistics.objects.get(user=request.user)
            user_statistics.yt_summaries_generated += 1
            user_statistics.save()
            return Response(summary_response, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



logger = logging.getLogger(__name__)

class PDFSummarizationView(APIView):
    def post(self, request):
        print("Received request data:", request.data)  # Debugging line
        file = request.FILES.get('pdf_file')  # Updated key to 'pdf_file'
        
        if file is None:
            print("File is None")  # Debugging line
            return Response({'error': 'No file uploaded. Please upload a PDF file.'}, status=status.HTTP_400_BAD_REQUEST)

        start_page = request.data.get('start_page_number')
        end_page = request.data.get('end_page_number')

        if start_page is None or end_page is None:
            return Response({'error': 'Please provide both start_page_number and end_page_number.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            start_page = int(start_page)
            end_page = int(end_page)
        except ValueError:
            return Response({'error': 'Start and end page numbers must be valid integers.'}, status=status.HTTP_400_BAD_REQUEST)

        if start_page < 1 or end_page < 1 or end_page < start_page:
            return Response({'error': 'Invalid page numbers. Ensure start_page is less than or equal to end_page.'}, status=status.HTTP_400_BAD_REQUEST)

        images = pdf_to_images(file)
        total_pages = len(images)
        requested_pages = range(start_page, min(end_page + 1, total_pages + 1))

        summaries = {}
        for page_num in requested_pages:
            image = images[page_num - 1]  # Convert 1-indexed page number to 0-indexed
            summaries[page_num] = process_image_with_vision_model(image)

        summary_text = "\n".join([f"Page {num}: {summ}" for num, summ in summaries.items()])
        
        # Save the summary with start and end page numbers
        pdf_summary = PDFSummary(
            user=request.user,  # Ensure you're getting the user from the request
            pdf_file=file,
            start_page_number=start_page,
            end_page_number=end_page,
            summary=summary_text
        )
        pdf_summary.save()
        UserContribution.objects.create(user=request.user, contribution_type='pdf_summary')
        
                # Increment the pdfs_summarized count in UserStatistics
        user_statistics = UserStatistics.objects.get(user=request.user)
        user_statistics.pdfs_summarized += 1
        user_statistics.save()

        return Response({'summaries': summaries}, status=status.HTTP_201_CREATED)
    


        
class PDFMindmapView(APIView):
    def post(self, request):
        # Get the uploaded PDF file
        pdf_file = request.FILES.get('pdf_file')

        # Check if a file was uploaded
        if not pdf_file:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the PDF file temporarily for processing
        file_path = default_storage.save(pdf_file.name, pdf_file)

        # Extract the text from the uploaded PDF
        with default_storage.open(file_path, 'rb') as pdf_file:
            raw_text = get_pdf_text([pdf_file])  # Pass as a list to maintain compatibility

        # Split the text into chunks and generate vector store
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Query the LLM to generate the mind map in JSON format
        user_query = """Generate a mind map of the document in JSON format. 
        Here's an example of the expected output:
        {
            "title": "Document Title",
            "nodes": [
                {
                    "id": "1",
                    "text": "Main Topic",
                    "nodes": [
                        {
                            "id": "1.1",
                            "text": "Subtopic 1"
                        },
                        {
                            "id": "1.2",
                            "text": "Subtopic 2"
                        }
                    ]
                }
            ]
        }"""

        mindmap_json = generate_mindmap(user_query)  # Call your function to generate the mind map

        # Return the generated mind map as JSON
        return Response({'mindmap': mindmap_json}, status=status.HTTP_200_OK)



def process_video(video_path, seconds_per_frame=3):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    while curr_frame < total_frames - 400:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    video.release()

    # Extract audio
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    return base64Frames, audio_path

open_ai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=open_ai_key)

class VideoUploadView(APIView):
    def post(self, request):
        serializer = VideoSerializer(data=request.data)
        if serializer.is_valid():
            video_instance = serializer.save()  # Save the video file
            
            # Get the file path of the uploaded video
            video_path = default_storage.path(video_instance.video_file.name)
            
            # Process the video
            base64Frames, audio_path = process_video(video_path, seconds_per_frame=1)

            # Transcribe the audio using Whisper
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=open(audio_path, "rb"),
            )

            # Generate an explanation using the GPT model
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Explain the video and its transcript in detail."},
                    {
                        "role": "user",
                        "content": [
                            "These are the frames from the video.",
                            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames),
                            {"type": "text", "text": f"The audio transcription is: {transcription.text}"}
                        ],
                    }
                ],
                temperature=0,
            )
            
            explanation = response.choices[0].message.content
            return Response({'explanation': explanation}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)