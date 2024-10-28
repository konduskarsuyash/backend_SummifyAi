from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_quiz_summary")


def get_conversational_chain():

    prompt_template = """
    You are a Quiz generator llm based on the context
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(model_name="llama-3.1-70b-versatile",temperature=0.35)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index_quiz_summary", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])
    
    quiz_json = response["output_text"]  # Rename for clarity

    # Save to quiz_summary.json with UTF-8 encoding
    with open("quiz_summary.json", "w", encoding="utf-8") as f:
        f.write(quiz_json)  # Write the quiz JSON to the file
        
        
    
    
       system_task = """Generate a quiz based on the provided content in strict JSON format. The quiz should include a total of 15 questions, equally distributed across three difficulty levels: easy, medium, and hard, with 5 questions in each section. Each question type should encourage higher-order thinking skills (application, analysis, synthesis). 

Here’s an example of the expected output:

{
  "quiz": {
    "easy": {
      "multiple_choice": [
        {
          "question": "What is the basic definition of X?",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "answer": "Option A"
        },
        {
          "question": "Which of the following is a characteristic of Y?",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "answer": "Option B"
        }
      ],
      "true_false": [
        {
          "statement": "Z is true.",
          "answer": true
        },
        {
          "statement": "A is false.",
          "answer": false
        }
      ]
    },
    "medium": {
      "multiple_choice": [],
      "true_false": []
    },
    "hard": {
      "multiple_choice": [],
      "true_false": []
    }
  }
}

Please generate the quiz strictly in this JSON format in code format avoid using backticks at the beginning and at the end without any additional text or explanation.
"""


system_summary = """2:
Autoregressive Integrated moving average model (ARIMA) This is a generalization of a simple ARMA model. The objective of this model is to predict future time series movement by examining the difference b/w values in the series instead of the actual values. ARMA models are applied in the cases where the data shows evidence of non-stationarity and to transform into stationary data. The way to transform non-stationary data to stationary is to apply the differencing step. The differencing can be applied one or more times to make the data stationary. The model can be split into smaller components: * *AR* model which is linearly dependant on its own previous values * *I* - Integrated, means the differencing step to generate stationary time series data (removing trend or any other time series component) * *MA* model is dependant on the past residual values *eg: Predict the milk production for current month using first order ARIMA model **ARIMA(1,2,2) = Yt = μ + Φ1Yt-1 + Θ1Et-1 + Θ2Et-2* * *P* - 1 (AR) * *d* - 2 (MA) * *q* - 2 (Differencing) *Assume Yt → original series* * 1st diff = 2t = Yt - Yt-1 → 1st differencing * 2nd diff = a t = 2t - 2t-1 → 2nd differencing *Yt = μ + Φ1Yt-1 + Φ2Yt-2 + Θ1Et-1 + Θ2Et-2 + Et* undefined

Page 3:
Time Series Unit Roots This document explains the concept of unit roots in time series data. *Unit roots* are a problem that arises when modeling time series data. If a time series data has a unit root, then the data is non-stationary and we cannot apply models on it. To remove unit roots, we need to do some transformation. If any transformation cannot be applied, then we need to be aware that unit root exists so that other methods of analysis can be applied. *A time series has unit root if (φ) = -1 or 1* The document provides two examples of unit roots: *1. AR(1) Model:* - *at = φat-1 + εt* (where εt is a white noise process) This equation represents an AR(1) model, which can be rewritten to represent an AR(1) model in a moving average (MA) format: - *at = φ^∞a0 + Σ(φ^iεt-i)* The first term of the time series (a0) is the lagged version of εt. *2. Variance of at:* - *var(at) = σ^2[φ^0 + φ^2 + φ^4 + ... + φ^2(t-1)]* Here, the variance of εt is assumed to be constant over time. Overall, this document highlights the importance of identifying unit roots in time series data and provides a basic understanding of how to address them. undefined

Page 4:
Different Values of |Φ| This document explains how the absolute value of Φ affects the behavior of a time series. *|Φ| < 1* * As time increases, the absolute value of Φ decreases. * Assuming Φ = 0.5, the expected value of the time series E(a<sub>t</sub>) is 0, and its variance is Φ<sup>2</sup>/(1-Φ<sup>2</sup>) * The series is geometric, resulting in smaller values over time, eventually converging to approximately zero. * The graph is stationary, meaning the values fluctuate around a constant average. *|Φ| > 1* * If Φ = 1.5, the time series will grow exponentially upwards. * Since Φ > 1, anything raised to a larger exponent will give a larger value. Therefore, E(a<sub>t</sub>) will approach infinity. * The graph is non-stationary, indicating that the values are not fluctuating around a constant average. *|Φ| = 1* * When Φ = 1, the formula provides a constant value for the expected value. * However, when substituting into the variance formula, the series results in a non-constant variance, meaning the spread of values is not constant. * The graph would be non-stationary in this case. The document highlights that the value of |Φ| determines the stability and behavior of the time series. It also shows how the mathematical calculations reflect the behavior of the graphs. Understanding these concepts is crucial for interpreting and analyzing time series data. undefined"""
                text_chunks = get_text_chunks(system_summary)
                get_vector_store(text_chunks)
 