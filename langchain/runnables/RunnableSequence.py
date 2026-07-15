from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt=PromptTemplate(
    template='give mea joke on {topic}',
    input_variables=['topic']
)

model=ChatMistralAI(model='mistral-medium-latest')

parser=StrOutputParser()

prompt2=PromptTemplate(
    template='explain the joke {text}',
    input_variables=['text']
)

chain=RunnableSequence(prompt,model,parser,prompt2,model,parser)

result=chain.invoke({'topic':'AI'})

print(result)