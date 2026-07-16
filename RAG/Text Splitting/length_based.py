from langchain_text_splitters import CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"RAG\document loaders\Books\GRU.pdf")

docs = loader.load()


text = "This is a sample text that we will use to demonstrate the functionality of the CharacterTextSplitter. The text will be split into smaller chunks based on the specified character limit."

splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0, separator="")

# result = splitter.split_text(text)

result = splitter.split_documents(docs)

print(result[0])

