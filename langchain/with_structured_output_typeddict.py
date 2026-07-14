from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from typing import Optional, TypedDict,Annotated

load_dotenv()

model = ChatMistralAI()

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The sentiment of the review"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("Hardware is greate, but the software feels bloated.")

print(result)
print(result['summary'])
print(result['sentiment'])
print(result.keys())