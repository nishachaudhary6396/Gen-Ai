from langchain_community.document_loaders import TextLoader

data = TextLoader(r"RAG\document loaders\notes.txt")

docs = data.load()

print(docs)
print(docs[0].page_content)

print(len(docs))