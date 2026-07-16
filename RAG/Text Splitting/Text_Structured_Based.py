from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "This is a sample text that we will use to demonstrate the functionality of the RecursiveCharacterTextSplitter. The text will be split into smaller chunks based on the specified character limit."

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)