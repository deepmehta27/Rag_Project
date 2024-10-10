from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import pdfplumber
import os
import tempfile
import uuid
import hashlib
import re
from langchain.docstore.document import Document  # Import the Document class to store page content from PDFs

# Function to sanitize filenames for the vectorstore collection
def clean_filename(filename):
    """
    Cleans and sanitizes the filename to be used as a collection name for Chroma.
    Replaces invalid characters, ensures length restrictions, and removes leading/trailing underscores.
    """
    cleaned_name = re.sub(r'[^a-zA-Z0-9-_]', '_', filename)  # Replace invalid characters with underscores
    cleaned_name = re.sub(r'_+', '_', cleaned_name)  # Collapse multiple underscores into one
    cleaned_name = cleaned_name.strip('_')  # Remove leading/trailing underscores

    # Ensure the name is at least 3 characters long
    if len(cleaned_name) < 3:
        cleaned_name = cleaned_name + '_default'  # Add "_default" if the name is too short
        cleaned_name = cleaned_name[:3]  # Truncate to exactly 3 characters if needed

    # Ensure the name isn't longer than 63 characters (limit for collection names)
    if len(cleaned_name) > 63:
        cleaned_name = cleaned_name[:63]

    # Ensure the filename starts and ends with an alphanumeric character
    if not (cleaned_name[0].isalnum() and cleaned_name[-1].isalnum()):
        raise ValueError("The cleaned filename must start and end with an alphanumeric character.")

    return cleaned_name

# Function to extract text from an uploaded PDF file
def get_pdf_text(uploaded_file):
    """
    Reads and extracts text from a PDF file using pdfplumber, then wraps each page's content in Document objects.
    Handles the PDF extraction process safely by using temporary files.
    """
    try:
        input_file = uploaded_file.read()  # Read the uploaded file into memory
        temp_file = tempfile.NamedTemporaryFile(delete=False)  # Create a temporary file
        temp_file.write(input_file)  # Write the file's content to temp storage
        temp_file.close()

        # Open the PDF and extract text from each page, ensuring non-empty pages
        with pdfplumber.open(temp_file.name) as pdf:
            documents = [Document(page_content=page.extract_text(), metadata={"page": i + 1}) for i, page in enumerate(pdf.pages) if page.extract_text()]
            return documents
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the PDF: {e}")
    finally:
        os.unlink(temp_file.name)  # Clean up the temporary file after reading

# Function to split long documents into smaller chunks for easier processing
def split_document(documents, chunk_size=512, chunk_overlap=50):
    """
    Splits a list of document texts into smaller chunks to manage long texts more efficiently.
    Takes in the chunk size and overlap to ensure context is preserved across chunks.
    """
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Function to create an OpenAI embedding function using the provided API key
def get_embedding_function(api_key):
    """
    Creates an OpenAI embeddings object using the OpenAI API with the 'text-embedding-ada-002' model.
    This function is used to transform text into vector embeddings for similarity search.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
    return embeddings

# Global dictionary to store document hashes, avoiding duplicate storage
added_document_hashes = {}

# Helper function to create a unique hash from the document content
def generate_sha256_hash_from_text(text: str) -> str:
    """
    Generates a SHA-256 hash for a given string of text.
    This is useful for tracking and avoiding duplication of document content.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text.encode('utf-8'))
    return sha256_hash.hexdigest()

# Function to generate a unique hash for a list of documents
def get_document_hash(documents):
    """
    Combines the content of all the documents and generates a unique hash to represent them.
    This is used to detect if a document has already been processed before.
    """
    combined_content = "".join(doc.page_content for doc in documents)  # Join all document content
    return generate_sha256_hash_from_text(combined_content)

# Function to create a vector store from text chunks
def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    """
    Creates a persistent vector store (using Chroma) from the document chunks.
    Assigns a unique ID to each chunk and stores them with their vector embeddings for later retrieval.
    """
    collection_name = clean_filename(file_name)  # Sanitize the file name to use as a collection name
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]  # Generate unique IDs for chunks

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name=collection_name,
        embedding=embedding_function,
        ids=ids,
        persist_directory=vector_store_path
    )

    vectorstore.persist()  # Save the vector store to disk
    return vectorstore

# Function to create or load a vector store from documents, avoiding duplicates
def create_vectorstore_from_texts(documents, api_key, file_name):
    """
    Checks if the document has been processed already (using a hash) and creates a vector store from the text chunks if not.
    Returns the vector store object for querying.
    """
    document_hash = get_document_hash(documents)  # Generate a hash for the document

    # If document already exists, return the existing vector store
    if document_hash in added_document_hashes:
        print("Document already exists in the vector store.")
        return added_document_hashes[document_hash]

    # Split the document into chunks, then create a new vector store
    docs = split_document(documents, chunk_size=1500, chunk_overlap=200)
    embedding_function = get_embedding_function(api_key)

    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    added_document_hashes[document_hash] = vectorstore  # Store the hash for future reference

    return vectorstore

# Function to load a previously saved vector store from disk
def load_vectorstore(file_name, api_key, vectorstore_path="db"):
    """
    Loads a pre-existing vector store from disk, allowing you to query previously processed documents.
    """
    embedding_function = get_embedding_function(api_key)  # Get the embedding function
    return Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function,
        collection_name=clean_filename(file_name)
    )

# Template for the prompt that will be used to query documents
PROMPT_TEMPLATE = """
You are an assistant for extracting information from research articles.
Extract the following pieces of information accurately: title, summary, methodology,
publication year, authors, future work, and citation. Please format the citation
in APA style, including the authors, publication year, title, and source.

Use the following context to answer the question:
{context}

If you don't know the answer, say "I don't know." Do not make up answers.

Answer the following question based on the context: {question}
"""

# Models to structure the answers with sources
class AnswerWithSources(BaseModel):
    """Represents an answer extracted from the document, along with the sources and reasoning."""
    answer: str
    sources: str
    reasoning: str

class ExtractedInfoWithSources(BaseModel):
    """Represents the extracted information from the research article, structured with all required fields."""
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources
    future_work: AnswerWithSources
    methodology: AnswerWithSources

# Function to format document content into a single string
def format_docs(docs):
    """
    Concatenates the content of the Document objects into a single string for easier processing.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Function to query the vector store and get a structured response based on the query
def query_document(vectorstore, query, api_key):
    """
    Queries the vector store using an LLM to extract information based on the provided query.
    Returns a structured response with the requested details (title, summary, etc.).
    """
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)  # Instantiate the LLM model
    retriever = vectorstore.as_retriever(search_type="similarity")  # Set up a retriever for similarity search

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)  # Prepare the prompt template

    # Chain for extracting information using the LLM and the retriever
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
    )

    structured_response = rag_chain.invoke(query)  # Invoke the query on the chain

    citation_authors = structured_response.paper_authors.answer
    citation_year = structured_response.publication_year.answer
    citation_title = structured_response.paper_title.answer
    # Format the citation nicely
    citation_work = f"{citation_authors} ({citation_year}). {citation_title}."
    
    # Format the structured response into a readable message
    response_message = (
        f"**Title:** {structured_response.paper_title.answer}\n\n"
        f"**Summary:** {structured_response.paper_summary.answer}\n\n"
        f"**Methodology:** {structured_response.methodology.answer}\n\n"
        f"**Publication Year:** {structured_response.publication_year.answer}\n\n"
        f"**Authors:** {structured_response.paper_authors.answer}\n\n"
        f"**Future Work:** {structured_response.future_work.answer}\n\n"
        f"**Citation:** {citation_work}"  # Ensure the citation is included in the output
    )

    return response_message  # Return the formatted response
