# Introduction to Langchain and RAG

I've been using tools like openwebui with built in RAG (Retrieval augmented generation) capability using the knowledge base feature with my local AI setup. I was curious about what actually goes into building a RAG setup for future projects that I have on my mind. So I decided to do a course on RAG with Langchain that i found on educative.io. It is a basic introductory course that covers Retrieval, vectorisation using ChromaDb and Qdrant with embeddings being generated using OpenAI embeddings models and Generation using GPT-6 Astra. 

I don't want to pay for API usage on OpenAI so I used embeddinggemma and LFM3-1.7B on my macbook pro using LM Studio for inference with the OpenAI compatible endpoint.

The basic flow for RAG is:

1. Load the documents and split them into text chunks
2. Generate embeddings for each chunk
3. Store the embeddings in a vector database
4. Embed the user's query and use a retriever to find the most semantically similar chunks
5. Augment the prompt with the retrieved context and the question, and send it to the LLM to generate the answer

The course used Streamlit for a basic form for File upload and text input for sending a question and receiving and answer.

