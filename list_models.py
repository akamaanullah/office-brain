import google.generativeai as genai
import os

API_KEY = "AIzaSyAx9TunkefuJeNoCuex4p7xTh2akH_VOSU"
genai.configure(api_key=API_KEY)

print("List of available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
