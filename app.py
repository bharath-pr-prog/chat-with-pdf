import gradio as gr
from utils.pdf_extractor import extract_text_from_pdf, summarize_text, answer_question


# Handle the PDF upload and user interaction
def chat_with_pdf(pdf_file, user_question):
    # Step 1: Extract and summarize the PDF
    pdf_text = extract_text_from_pdf(pdf_file)
    pdf_summary = summarize_text(pdf_text)

    # Step 2: Get the answer from OpenAI
    answer = answer_question(pdf_summary, user_question)

    return answer

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 Chat with Your PDF")
    
    pdf_input = gr.File(label="Upload your PDF")
    question_input = gr.Textbox(label="Ask a question about the PDF")
    output = gr.Textbox(label="Answer")
    
    submit_button = gr.Button("Submit")
    submit_button.click(chat_with_pdf, inputs=[pdf_input, question_input], outputs=output)

    demo.launch()


