from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI()

prompt = PromptTemplate(
    template='Write a summary of the following text: {text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader(r"RAG\document loaders\cricket.txt", encoding='utf-8')

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({"text": docs[0].page_content}))
