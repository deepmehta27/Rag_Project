from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pdfplumber

import os
import tempfile
import uuid
import hashlib
import re

from langchain.docstore.document import Document  # Import Document class

def clean_filename(filename):
    cleaned_name = re.sub(r'[^a-zA-Z0-9-_]', '_', filename)  
    cleaned_name = re.sub(r'_+', '_', cleaned_name)  
    cleaned_name = cleaned_name.strip('_')  

    if len(cleaned_name) < 3:
        cleaned_name = cleaned_name + '_default'  
        cleaned_name = cleaned_name[:3]  

    if len(cleaned_name) > 63:
        cleaned_name = cleaned_name[:63]  

    if not (cleaned_name[0].isalnum() and cleaned_name[-1].isalnum()):
        raise ValueError("The cleaned filename must start and end with an alphanumeric character.")

    return cleaned_name

def get_pdf_text(uploaded_file):
    try:
        input_file = uploaded_file.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        with pdfplumber.open(temp_file.name) as pdf:
            documents = [Document(page_content=page.extract_text()) for page in pdf.pages if page.extract_text()]  # Wrap text in Document objects
            return documents
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the PDF: {e}")
    finally:
        os.unlink(temp_file.name)

def split_document(documents, chunk_size=512, chunk_overlap=50):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def get_embedding_function(api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
    return embeddings

added_document_hashes = {}

def generate_sha256_hash_from_text(text: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(text.encode('utf-8'))
    return sha256_hash.hexdigest()

def get_document_hash(documents):
    combined_content = "".join(doc.page_content for doc in documents)  # Ensure you are dealing with Document objects
    return generate_sha256_hash_from_text(combined_content)

def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    collection_name = clean_filename(file_name)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

    vectorstore = Chroma.from_documents(
        documents=chunks,
        collection_name=collection_name,
        embedding=embedding_function,
        ids=ids,
        persist_directory=vector_store_path
    )

    vectorstore.persist()
    return vectorstore

def create_vectorstore_from_texts(documents, api_key, file_name):
    document_hash = get_document_hash(documents)

    if document_hash in added_document_hashes:
        print("Document already exists in the vector store.")
        return added_document_hashes[document_hash]

    docs = split_document(documents, chunk_size=1500, chunk_overlap=200)
    embedding_function = get_embedding_function(api_key)

    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    added_document_hashes[document_hash] = vectorstore

    return vectorstore

def load_vectorstore(file_name, api_key, vectorstore_path="db"):
    embedding_function = get_embedding_function(api_key)
    return Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function,
        collection_name=clean_filename(file_name)
    )

PROMPT_TEMPLATE = """
You are an assistant for extracting information from research articles.
Extract the following pieces of information accurately: title, summary, methodology,
publication year, authors, and future work.

Use the following context to answer the question:
{context}

If you don't know the answer, say "I don't know." Do not make up answers.

Answer the following question based on the context: {question}
"""

class AnswerWithSources(BaseModel):
    answer: str
    sources: str
    reasoning: str

class ExtractedInfoWithSources(BaseModel):
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources
    future_work: AnswerWithSources
    methodology: AnswerWithSources

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def query_document(vectorstore, query, api_key):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    retriever = vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
    )

    structured_response = rag_chain.invoke(query)

    response_message = (
        f"**Title:** {structured_response.paper_title.answer}\n\n"
        f"**Summary:** {structured_response.paper_summary.answer}\n\n"
        f"**Methodology:** {structured_response.methodology.answer}\n\n"
        f"**Publication Year:** {structured_response.publication_year.answer}\n\n"
        f"**Authors:** {structured_response.paper_authors.answer}\n\n"
        f"**Future Work:** {structured_response.future_work.answer}\n\n"
        f"**Sources:** {structured_response.paper_title.sources}"
    )

    return response_message
