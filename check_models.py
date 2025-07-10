import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Available models for generateContent:")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
