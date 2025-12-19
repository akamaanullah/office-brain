import google.generativeai as genai
import os

API_KEY = "AIzaSyAx9TunkefuJeNoCuex4p7xTh2akH_VOSU"
genai.configure(api_key=API_KEY)

print("Check available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")
