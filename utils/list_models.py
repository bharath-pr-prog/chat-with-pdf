import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available models
for m in genai.list_models():
    print(m.name)