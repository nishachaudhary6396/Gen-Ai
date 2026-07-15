from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_core.prompts import PromptTemplate

load_dotenv()

prompt1=PromptTemplate(
    template='Generate a tweet about{topic}',
    input_variables=['topic']

)

prompt2=PromptTemplate(
    template='Generate the linked in post on {topic}',
    input_variables=['topic']
)

model=ChatMistralAI(model='mistral-medium-latest')


parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)

})

result=parallel_chain.invoke({"topic":'AI'})
print(result)