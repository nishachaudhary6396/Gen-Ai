from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1=PromptTemplate(template='generate detailed report on {topic}',
                       input_variables=['topic']
)

prompt2=PromptTemplate(template='generate 5 line summary of the {text}',
                       input_variables=['text']
)

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'mental health of youth in india'})

print(result)