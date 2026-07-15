from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

model2=ChatMistralAI(model='mistral-medium-latest')


prompt1=PromptTemplate(
    template='Generate the notes of the following text\n{text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate the  5 quiz questions with its option for the following text\n{text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the notes and quiz into a single document{notes} and {quiz}',
    input_variables=['notes','quiz']
    
)


parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser
})

merge_chain=prompt3 | model1 | parser

chain=parallel_chain | merge_chain
text="""Artificial Intelligence (AI) is one of the most transformative technologies of the 21st century. It refers to the ability of machines or computer systems to perform tasks that typically require human intelligence. These tasks include learning from experience, recognizing patterns, understanding natural language, solving problems, making decisions, and even generating creative content. AI has evolved significantly over the past few decades, moving from rule-based systems to advanced machine learning and deep learning models that can process massive amounts of data and improve their performance over time.

The concept of Artificial Intelligence was first introduced in the 1950s by computer scientist John McCarthy, who is often referred to as the father of AI. Early AI systems relied heavily on manually written rules and logical reasoning. These systems could perform only specific tasks and struggled when faced with situations outside their programmed knowledge. As computing power increased and larger datasets became available, researchers shifted toward machine learning, allowing computers to learn patterns directly from data rather than relying solely on predefined rules.

Machine Learning (ML) is a subset of AI that focuses on enabling computers to learn from data. Instead of explicitly programming every decision, developers provide examples, and the algorithm identifies patterns that can be used to make predictions. There are three primary types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, models are trained using labeled datasets where the correct output is already known. Examples include spam email detection, house price prediction, and disease diagnosis. Unsupervised learning works with unlabeled data and attempts to discover hidden structures or group similar items together. Customer segmentation and anomaly detection are common applications. Reinforcement learning involves an agent learning by interacting with an environment and receiving rewards or penalties based on its actions. This approach has been successfully used in robotics, autonomous vehicles, and game-playing systems.

Deep Learning is a specialized branch of machine learning that uses artificial neural networks inspired by the structure of the human brain. These neural networks consist of multiple layers that process information and gradually learn complex patterns. Deep learning has revolutionized fields such as computer vision, speech recognition, and natural language processing. Modern applications like facial recognition, self-driving cars, voice assistants, and image generation rely heavily on deep learning techniques.

Natural Language Processing (NLP) is another important branch of AI that focuses on enabling computers to understand, interpret, and generate human language. NLP combines linguistics, computer science, and machine learning to process text and speech. Applications include language translation, sentiment analysis, text summarization, question answering, speech recognition, and chatbots. Recent advances in Large Language Models (LLMs), such as GPT, Llama, and Gemini, have dramatically improved the quality of NLP systems by training on massive collections of text data.

Generative AI is a rapidly growing area within artificial intelligence that focuses on creating new content rather than simply analyzing existing data. Generative AI models can produce text, images, music, videos, code, and even 3D objects. These models are typically based on transformer architectures, which are highly effective at processing sequential data. Popular applications include AI-powered writing assistants, image generators, coding assistants, virtual tutors, and automated content creation tools.

Transformers have become the foundation of modern AI systems. Introduced in the research paper "Attention Is All You Need" in 2017, transformers replaced traditional recurrent neural networks for many language processing tasks. The key innovation was the attention mechanism, which allows the model to focus on the most relevant parts of the input while processing information. This significantly improved both accuracy and training efficiency, making it possible to build very large language models.

Large Language Models are trained on enormous datasets containing books, articles, websites, and other publicly available text. During training, they learn statistical relationships between words, phrases, and concepts. Once trained, they can answer questions, summarize documents, translate languages, generate code, write essays, and engage in conversations. However, these models do not truly "understand" information in the same way humans do. Instead, they predict the most likely next token based on the patterns learned during training.

One important challenge with large language models is hallucination. Hallucination occurs when an AI model generates information that appears correct but is actually false or unsupported. This can happen because the model is predicting text rather than retrieving verified facts. To reduce hallucinations, developers often use Retrieval-Augmented Generation (RAG), where relevant information is retrieved from external knowledge sources before generating a response. RAG combines vector databases, embedding models, and language models to provide more accurate and context-aware answers.

Embeddings play a crucial role in modern AI applications. An embedding is a numerical vector representation of text, images, or other data that captures semantic meaning. Similar pieces of information have embedding vectors that are close together in high-dimensional space. Embeddings are widely used in semantic search, recommendation systems, document retrieval, clustering, and RAG pipelines. Popular embedding providers include OpenAI, Cohere, Google, and Hugging Face.

Vector databases are designed to store and search embedding vectors efficiently. Unlike traditional databases that search using exact keywords, vector databases search based on semantic similarity. This enables users to find relevant information even if the exact words do not match. Common vector databases include Pinecone, ChromaDB, FAISS, Milvus, and Weaviate. These databases are essential components of modern retrieval systems.

AI is being applied across numerous industries. In healthcare, AI assists doctors in diagnosing diseases, analyzing medical images, predicting patient outcomes, and discovering new drugs. In finance, AI detects fraud, automates trading, evaluates credit risk, and provides personalized financial advice. Manufacturing companies use AI for predictive maintenance, quality inspection, and production optimization. Retail businesses employ AI for recommendation systems, inventory management, demand forecasting, and customer support. Transportation companies use AI in autonomous driving, route optimization, and traffic prediction.

Despite its advantages, AI also presents several challenges. Ethical concerns include bias in AI models, lack of transparency, privacy issues, misinformation, and job displacement due to automation. Responsible AI development requires fairness, accountability, transparency, and human oversight. Governments and organizations worldwide are working on regulations and guidelines to ensure AI is developed and deployed safely.

The future of Artificial Intelligence is expected to include more capable AI assistants, autonomous agents, multimodal systems that understand text, images, audio, and video simultaneously, and increasingly personalized user experiences. Researchers are also exploring Artificial General Intelligence (AGI), a theoretical form of AI capable of performing any intellectual task that a human can do. Although AGI has not yet been achieved, advancements in machine learning, computing power, and data availability continue to push AI capabilities forward.

Artificial Intelligence is no longer limited to research laboratories. It has become an integral part of everyday life through smartphones, search engines, recommendation systems, virtual assistants, online shopping, healthcare, education, finance, and countless other applications. As AI continues to evolve, understanding its principles, strengths, limitations, and ethical implications will become increasingly important for students, professionals, and organizations around the world.
"""


result=chain.invoke({'text':text})

print(result)