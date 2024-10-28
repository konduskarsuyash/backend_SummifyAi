from pdf2image import convert_from_path
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import fitz
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from.env file


api_key = os.getenv("GOOGLE_API_KEY")

# Configure the generative model with your API key
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Convert uploaded PDF to images
def pdf_to_images(pdf_file):
    # Save the uploaded file to a temporary location
    temp_pdf_path = "temp.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.read())
    
    # Open the PDF file using PyMuPDF
    images = []
    try:
        document = fitz.open(temp_pdf_path)
        for page_num in range(len(document)):
            # Get the page
            page = document.load_page(page_num)
            # Render the page as a pixmap (image)
            pix = page.get_pixmap()
            # Convert pixmap to PIL image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    finally:
        # Close the document
        document.close()
        # Remove the temporary PDF after conversion
        os.remove(temp_pdf_path)

    return images

# Convert PIL Image to byte array
def pil_image_to_bytes(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')  # Save as PNG format (can be adjusted as necessary)
    img_byte_arr = img_byte_arr.getvalue()  # Get the raw image bytes
    return img_byte_arr

# Vision model inference function
def vision_model_inference(image):
    image_bytes = pil_image_to_bytes(image)  # Convert PIL image to bytes
    response = model.generate_content(
        ["""You would be given image of notes/textbooks it can be handwritten too so make sure to read
        the content carefully. There can be handmade diagrams and flowcharts too. Your job is to make 
        a detailed report or summary by reading which the user can study better.""",
        image ] # Pass image bytes to the model
    )
    
    return response.text

# Process a specific image with the vision model
def process_image_with_vision_model(image):
    text_content = vision_model_inference(image)
    return text_content

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_quiz_summary")

def get_conversational_chain():
    prompt_template = """
    You are a Quiz generator LLM based on the context.
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context", don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGroq(model_name="llama-3.1-70b-versatile", temperature=0.35)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def user_input(system_task, user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index_quiz_summary", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    quiz_json = response["output_text"]  # Rename for clarity

    return {"quiz_summary": quiz_json}  # Return the quiz JSON
