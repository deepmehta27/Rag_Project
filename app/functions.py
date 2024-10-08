
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import os
import tempfile
import uuid
import hashlib
import pandas as pd
import re

def clean_filename(filename):
    """
    Clean and ensure the filename meets the criteria for Chroma collection names.

    Parameters:
        filename (str): The filename to clean.

    Returns:
        str: A cleaned and valid filename.
    """
    # Remove invalid characters (anything that's not alphanumeric, _, or -)
    cleaned_name = re.sub(r'[^a-zA-Z0-9-_]', '_', filename)  
    cleaned_name = re.sub(r'_+', '_', cleaned_name)  # Replace multiple underscores with a single underscore
    cleaned_name = cleaned_name.strip('_')  # Remove leading/trailing underscores

    # Handle filenames that are too short
    if len(cleaned_name) < 3:
        cleaned_name = cleaned_name + '_default'  # Append "_default" to meet the minimum length
        cleaned_name = cleaned_name[:3]  # Ensure it's exactly 3 characters long if it's still too short

    # Handle filenames that are too long
    if len(cleaned_name) > 63:
        cleaned_name = cleaned_name[:63]  # Trim to 63 characters if too long

    # Ensure it starts and ends with an alphanumeric character
    if not (cleaned_name[0].isalnum() and cleaned_name[-1].isalnum()):
        raise ValueError("The cleaned filename must start and end with an alphanumeric character.")

    return cleaned_name


def get_pdf_text(uploaded_file): 
    """
    Load a PDF document from an uploaded file and return it as a list of documents.
    """
    try:
        input_file = uploaded_file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        return documents

    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the PDF: {e}")

    finally:
        os.unlink(temp_file.name)



def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks.
    chunk_size: The desired maximum size of each chunk (default: 400)
    chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

    Returns:
        list: A list of smaller text chunks created from the generic text
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                          chunk_overlap=chunk_overlap,
                                          length_function=len,
                                          separators=["\n\n", "\n", " "])
    
    return text_splitter.split_documents(documents)


def get_embedding_function(api_key):
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
    as an argument to the function.

    Parameters:
        api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=api_key
    )
    return embeddings

added_document_hashes = {}

def generate_sha256_hash_from_text(text: str) -> str:
    """Generate a SHA256 hash for the given text."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text.encode('utf-8'))
    return sha256_hash.hexdigest()

def get_document_hash(documents):
    """Generate a unique hash for a list of documents."""
    combined_content = "".join(doc.page_content for doc in documents)
    return generate_sha256_hash_from_text(combined_content)

def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    """Create a vector store from a list of text chunks."""
    # Create a unique identifier for each document
    
    collection_name = clean_filename(file_name)
    
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=chunks,
                                        collection_name=collection_name,
                                        embedding=embedding_function,
                                        ids=ids,
                                        persist_directory=vector_store_path)

    vectorstore.persist()
    return vectorstore

def create_vectorstore_from_texts(documents, api_key, file_name):
    """Create a vector store from a list of texts."""
    # Generate a hash for the document
    document_hash = get_document_hash(documents)

    # Check if the document hash already exists in the tracking dictionary
    if document_hash in added_document_hashes:
        # Return the existing vector store instead of creating a new one
        print("Document already exists in the vector store.")
        return added_document_hashes[document_hash]

    # Split the documents for processing
    docs = split_document(documents, chunk_size=1500, chunk_overlap=200)

    # Define embedding function
    embedding_function = get_embedding_function(api_key)

    # Create a new vector store
    vectorstore = create_vectorstore(docs, embedding_function, file_name)

    # Store the document hash in the tracking dictionary
    added_document_hashes[document_hash] = vectorstore

    return vectorstore
# def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):

#     """
#     Create a vector store from a list of text chunks.

#     :param chunks: A list of generic text chunks
#     :param embedding_function: A function that takes a string and returns a vector
#     :param file_name: The name of the file to associate with the vector store
#     :param vector_store_path: The directory to store the vector store

#     :return: A Chroma vector store object
#     """

#     # Create a list of unique ids for each document based on the content
#     ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
#     # Ensure that only unique docs with unique ids are kept
#     unique_ids = set()
#     unique_chunks = []
    
#     unique_chunks = [] 
#     for chunk, id in zip(chunks, ids):     
#         if id not in unique_ids:       
#             unique_ids.add(id)
#             unique_chunks.append(chunk)        

#     # Create a new Chroma database from the documents
#     vectorstore = Chroma.from_documents(documents=unique_chunks, 
#                                         collection_name=clean_filename(file_name),
#                                         embedding=embedding_function, 
#                                         ids=list(unique_ids), 
#                                         persist_directory = vector_store_path)

#     # The database should save automatically after we create it
#     # but we can also force it to save using the persist() method
#     vectorstore.persist()
    
#     return vectorstore


# def create_vectorstore_from_texts(documents, api_key, file_name):
    
#     # Step 2 split the documents  
#     """
#     Create a vector store from a list of texts.

#     :param documents: A list of generic text documents
#     :param api_key: The OpenAI API key used to create the vector store
#     :param file_name: The name of the file to associate with the vector store

#     :return: A Chroma vector store object
#     """
#     docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
    
#     # Step 3 define embedding function
#     embedding_function = get_embedding_function(api_key)

#     # Step 4 create a vector store  
#     vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
#     return vectorstore


def load_vectorstore(file_name, api_key, vectorstore_path="db"):

    """
    Load a previously saved Chroma vector store from disk.

    :param file_name: The name of the file to load (without the path)
    :param api_key: The OpenAI API key used to create the vector store
    :param vectorstore_path: The path to the directory where the vector store was saved (default: "db")
    
    :return: A Chroma vector store object
    """
    embedding_function = get_embedding_function(api_key)
    return Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))

# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    

class ExtractedInfoWithSources(BaseModel):
    """Extracted information about the research article"""
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources
    future_work:AnswerWithSources
    methodology: AnswerWithSources

def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects

    :return: A string containing the text of all the documents joined by two newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)

def query_document(vectorstore, query, api_key):
    """
    Query a vector store with a question and return a structured response.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store
    :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

    :return: A formatted string containing the answer and reasoning
    """
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

    retriever = vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
    )

    structured_response = rag_chain.invoke(query)
    
    # Now access the individual fields of the response
    response_message = (
        f"**Title:** {structured_response.paper_title.answer}\n\n"
        f"**Summary:** {structured_response.paper_summary.answer}\n\n"
        f"**Methodology:** {structured_response.methodology.answer}\n\n"
        f"**Publication Year:** {structured_response.publication_year.answer}\n\n"
        f"**Authors:** {structured_response.paper_authors.answer}\n\n"
        f"**Future Work:** {structured_response.future_work.answer}\n\n"
        f"**Sources:** {structured_response.paper_title.sources}"  # Including one source for example
    )

    return response_message

