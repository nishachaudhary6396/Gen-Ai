from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt=PromptTemplate(template='give the 5 interesting facts related to {topic}',input_variables=['topic'])

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

parser=StrOutputParser()

chain=prompt | model | parser

result=chain.invoke({'topic':'Black hole'})

print(result)

# chain.get_graph().print_ascii()