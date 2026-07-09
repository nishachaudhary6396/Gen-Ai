from dotenv import load_dotenv

load_dotenv() 

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

response = model.invoke("Hello, how are you?")

print(response.content)  