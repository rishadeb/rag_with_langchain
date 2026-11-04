from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings_model = OpenAIEmbeddings(
    base_url="http://localhost:1235/v1",
    model="text-embedding-embeddinggemma-300m",
    api_key="test",
    check_embedding_ctx_length=False,
)  # Initialize the embeddings model
# Generate embeddings for a list of documents
embeddings = embeddings_model.embed_documents(
    [
        "This is the Fundamentals of RAG course.",
        "Educative is an AI-powered online learning platform.",
        "There are several Generative AI courses available on Educative.",
        "I am writing this using my keyboard.",
        "JavaScript is a good programming language",
    ]
)

print(
    len(embeddings)
)  # Print the number of embeddings generated (should be equal to the number of documents)
print(len((embeddings[0])))  # Print the length of the first embedding vector

# List of example documents to be used in the database
documents = [
    "Python is a high-level programming language known for its readability and versatile libraries.",
    "Java is a popular programming language used for building enterprise-scale applications.",
    "JavaScript is essential for web development, enabling interactive web pages.",
    "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
    "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
    "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
    "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
]

# Create a Chroma database from the documents using OpenAI embeddings
db = Chroma.from_texts(documents, embeddings_model)
print(db)

# Configure the database to act as a retriever, setting the search type to
# similarity and returning the top 1 result
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)

# Perform a similarity search with the given query
result = retriever.invoke("Where can I see Mona Lisa?")
print(result)