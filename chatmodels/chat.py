from dotenv import load_dotenv

load_dotenv() 

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.6,max_tokens=200,timeout=20,max_retries=3)

response = model.invoke("Hello, how are you?")

print(response.content)  