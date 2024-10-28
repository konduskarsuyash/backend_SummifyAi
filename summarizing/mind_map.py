import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

load_dotenv()
# Load the environment variables
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    """Extracts text from PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Splits the extracted text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Embeds text chunks and creates a vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    """Creates a conversational chain using the ChatGroq model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, say, 'answer is not available in the context'. Don't provide wrong answers.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGroq(model_name="llama-3.1-70b-versatile", temperature=0.35)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



import json

def generate_mindmap(user_question):
    """Generates a mind map based on the user question and PDF content."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Log the raw response
    mindmap_text = response["output_text"]
    print("Raw mind map response:", mindmap_text)  # Log the raw output for debugging

    # Clean the response to extract valid JSON
    try:
        # Find the JSON part of the response (strip extra text and backticks)
        start_index = mindmap_text.find('{')
        end_index = mindmap_text.rfind('}')
        
        if start_index == -1 or end_index == -1:
            raise ValueError("Failed to locate valid JSON in the response.")

        cleaned_json_text = mindmap_text[start_index:end_index+1]
        print("Cleaned mind map response:", cleaned_json_text)

        # Parse the cleaned JSON
        mindmap_json = json.loads(cleaned_json_text)  # Convert the string to JSON

    except (json.JSONDecodeError, ValueError) as e:
        # Log the error and raw response for debugging
        print("JSON Decode Error:", e)
        return {"error": "Failed to parse mind map JSON."}

    return mindmap_json



