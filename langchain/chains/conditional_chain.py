from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda

load_dotenv()

model=ChatMistralAI(model='mistral-medium-latest')

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Classify the user feedback into positive and negative')

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template='Classify the feedback into postive or negative of the give feedback\n{feedback}\n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt2=PromptTemplate(
    template='give the appropriate response for the positive feedback\n{feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='give the appropriate response for the negative feedback\n{feedback}',
    input_variables=['feedback']
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model | parser),
    (lambda x:x.sentiment=='negative',prompt3 | model | parser),
    RunnableLambda(lambda x : 'cant find the sentiment')
)

classifier_chain=prompt1 | model | parser2

chain=classifier_chain | branch_chain

result=chain.invoke({'feedback':'this is a terrible laptop'})

print(result)