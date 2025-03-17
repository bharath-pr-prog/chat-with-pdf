import PyPDF2
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Set up Google Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Summarize extracted text
def summarize_text(text):
    prompt = f"Summarize the following PDF content:\n\n{text}"
    response = model.generate_content(prompt)
    return response.text

# Answer user questions based on the PDF content
def answer_question(text, question):
    prompt = f"Answer the question based on the PDF content:\n\nPDF Content: {text}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    return response.text

# Chunking large PDF text for RAG
def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
