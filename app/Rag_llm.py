import streamlit as st
from functions import *
import base64
import re
import os

# Initialize the API key in session state if it doesn't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

if 'generate_clicked' not in st.session_state:
    st.session_state.generate_clicked = False

def display_pdf(uploaded_file):
    """Display a PDF file that has been uploaded to Streamlit."""
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def generate_valid_collection_name(filename):
    """Generate a valid collection name from the filename."""
    name = os.path.splitext(filename)[0]  # Remove file extension
    name = re.sub(r'\W+', '_', name)  # Replace non-alphanumeric characters with underscore
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)  # Ensure it starts with an alphanumeric character
    name = name[:63]  # Truncate to 63 characters if longer
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)  # Ensure it ends with an alphanumeric character
    return name

def validate_inputs(api_key, uploaded_file):
    """Validate user inputs for API key and file upload."""
    if not api_key or uploaded_file is None:
        return False
    return True

# Sidebar for user input
with st.sidebar:
    # Project title with emoji
    st.markdown("# 🚀 InsightFinder")

    # API key section
    st.header("Input Your OpenAI API Key")
    st.text_input('OpenAI API Key', type='password', key='api_key', label_visibility="collapsed")
    
    # File upload section
    st.header("Upload File")
    uploaded_file = st.file_uploader("Please upload your PDF document:", type="pdf")

    # Generate answer button
    if st.button("🔍 Generate Answer"):
        if validate_inputs(st.session_state.api_key, uploaded_file):
            st.session_state.generate_clicked = True
        else:
            st.warning("⚠️ Please provide both your OpenAI API key and upload a PDF document.")
    
    # Add spacing
    st.write("")  # Adds a blank line for spacing
    
    # Key Features section
    st.subheader("🌟 Key Features")
    st.markdown("""
        - **Seamless PDF Upload:** Easily upload your research papers for analysis.
        - **AI-Powered Insights:** Get detailed summaries, titles, publication dates, and future research directions with just a click.
        - **User-Friendly Interface:** Our intuitive design ensures a smooth experience for all users.
    """)


# Main area
st.header("🌟 Welcome to InsightFinder! 🚀")
st.write("InsightFinder is your AI-powered research assistant designed to help you effortlessly extract key insights from academic papers Whether you're a student, researcher, or knowledge enthusiast, our tool simplifies the process of navigating complex documents, allowing you to focus on what truly matters: understanding and leveraging valuable information.")
st.write("Upload your PDF document and provide your OpenAI API key to generate insights.")

# Conditional display based on user actions
if st.session_state.generate_clicked:
    with st.spinner("Generating answer..."):
        # Load in the documents
        documents = get_pdf_text(uploaded_file)

        # Generate a valid collection name
        collection_name = generate_valid_collection_name(uploaded_file.name)

        st.session_state.vector_store = create_vectorstore_from_texts(documents, 
                                                                      api_key=st.session_state.api_key,
                                                                      file_name=collection_name)

        answer_message = query_document(vectorstore=st.session_state.vector_store, 
                                        query="Give me the Title, Detailed summary, Methodology, publication date, authors of the research paper, and the future work.",
                                        api_key=st.session_state.api_key)

        # Display answer in the center
        st.markdown("### Generated Answer")
        st.markdown("<style>div.stMarkdown {text-align: left;}</style>", unsafe_allow_html=True)
        st.write(answer_message)

    # Reset the generate_clicked state for future submissions
    st.session_state.generate_clicked = False

elif uploaded_file:
    # Display PDF only if it's uploaded and answer hasn't been generated
    display_pdf(uploaded_file)
