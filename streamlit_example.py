import streamlit as st

# Specify the filename of your local image
# image_filename = 'Educative.png'

# Use st.image to display the image
# st.image(image_filename)


# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

with st.form(key='qa_form', clear_on_submit=True, border = False):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Please upload a file and enter your question.")
