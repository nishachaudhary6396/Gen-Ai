from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(r"RAG\document loaders\GRU.pdf")

docs = data.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)
