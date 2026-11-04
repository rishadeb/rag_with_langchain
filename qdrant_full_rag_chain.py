import streamlit as st
import pymupdf

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore


from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def read_pdf(file):
    with pymupdf.open(stream=file.read(), filetype="pdf") as pdf_document:
        return "\n".join(page.get_text() for page in pdf_document)

def generate_response(uploaded_files, query_text, openai_api_key="test"):
    # Load document if file is uploaded
    if uploaded_files is None:
        return None
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            document_text = read_pdf(uploaded_file)
        else:
            document_text = uploaded_file.read().decode()
        documents.append(document_text)

    texts = []
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    for document in documents:
        texts.extend(text_splitter.create_documents([document]))

    llm = ChatOpenAI(
        base_url="http://localhost:1235/v1",
        model="lfm3-1.7b",
        openai_api_key=openai_api_key
    )

    # Select embeddings model
    embeddings = OpenAIEmbeddings(
        base_url="http://localhost:1235/v1",
        model="text-embedding-embeddinggemma-300m",
        api_key=openai_api_key,
        check_embedding_ctx_length=False,
    )

    database = QdrantVectorStore.from_documents(
        texts,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )

    # Create retriever interface
    retriever = database.as_retriever()

    # Define a template for generating answers using provided context
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say 'thanks for asking!' at the end of the answer. Include source document used in context if used.

    {context}
    Question: {question}

    Helpful Answer:"""

    # Create a custom prompt template using the defined template
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # Create QA chain
    response = rag_chain.invoke(query_text)
    return response


# File upload
uploaded_files = st.file_uploader('Upload an article', type=['txt', 'pdf'], accept_multiple_files=True)

# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_files)

# Form input and query
result = None
with st.form('myform', clear_on_submit=False, border=False):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_files and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_files, query_text)
            result = response
if result:
    st.info(result)
