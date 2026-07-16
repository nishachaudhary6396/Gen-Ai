
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv()

embeddings = CohereEmbeddings(
    model="embed-english-v3.0"
)

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

#store in list

docs = [doc1, doc2, doc3, doc4, doc5]


vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
    collection_name="sample"
)

#add documents
result=vector_store.add_documents(docs)
print(result)

vector_store.get(include=['embeddings','documents','metadatas'])


#search documents

result = vector_store.similarity_search(
    query='Who is most famous among them?',
    k=1
) 

print(result)

result = vector_store.similarity_search_with_score(
    query='Who is most famous among them?',
    k=1
) 

print(result)

#meta-data filtering
result = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Mumbai Indians"}
)
print(result)

#update documents
updated_doc1 = Document(
    page_content="Virat Kohli is an Indian international cricketer and the former all-format captain of the Indian national cricket team. He is a right-handed batter and occasional right-arm medium-pace bowler. Considered one of the greatest batsmen in limited overs cricket",
    metadata = {"team": "RCB"}
)
vector_store.update_document(document_id='624dc199-bee4-4dce-a8b6-cbd606534185', document= updated_doc1)

#view document
vector_store.get(include=['embeddings','documents','metadatas'])

#delete document
vector_store.delete(ids=['5c997bc6-5b8b-4ab0-b1a7-1274ffb0c918'])
