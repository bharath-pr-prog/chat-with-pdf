import gradio as gr
from utils.pdf_extractor import extract_text_from_pdf
from utils.rag import chunk_text, create_and_store_embeddings, retrieve_relevant_chunks
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# RAG-Enhanced Chat with PDF
def chat_with_pdf(pdf_file, user_question):
    # Step 1: Extract and chunk PDF text
    pdf_text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(pdf_text)

    # Step 2: Create vector store (FAISS)
    index, chunk_list = create_and_store_embeddings(chunks)

    # Step 3: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(user_question, index, chunk_list)

    # Step 4: Formulate prompt with retrieved content
    context = " ".join(relevant_chunks)
    prompt = f"Answer the question using the following PDF context:\n\n{context}\n\nQuestion: {user_question}"
    response = model.generate_content(prompt)

    return response.text

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Chat with Large PDFs using RAG")
    
    pdf_input = gr.File(label="Upload your PDF")
    question_input = gr.Textbox(label="Ask a question about the PDF")
    output = gr.Textbox(label="Answer")
    
    submit_button = gr.Button("Submit")
    submit_button.click(chat_with_pdf, inputs=[pdf_input, question_input], outputs=output)

    demo.launch()
