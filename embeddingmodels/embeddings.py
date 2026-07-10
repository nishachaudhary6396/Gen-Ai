from langchain_cohere import CohereEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    dimensions=64
)

texts = [
    "Hello this is Nisha Chaudhary",
    "I am a data enignerr",
    "I am learning Gen AI"
]

vector = embeddings.embed_documents(texts)

# vector = embeddings.embed_query("You are going to learn Gen AI")

print(vector)