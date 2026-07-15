from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(r"RAG\document loaders\books", glob="*.pdf", loader_cls=PyPDFLoader)


# docs = loader.load()
docs = loader.lazy_load()

for document in docs:
    print(document.metadata)


print(docs[0].page_content)
print(docs[0].metadata)