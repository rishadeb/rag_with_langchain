import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_response(uploaded_file, query_text, openai_api_key="test"):
    # Load document if file is uploaded
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.create_documents(documents)

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

    # Create a vectorstore from documents
    database = Chroma.from_documents(texts, embeddings)
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
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = None
with st.form('myform', clear_on_submit=False, border=False):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, query_text)
            result = response
if result:
    st.info(result)