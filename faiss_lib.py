from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from azure.identity import InteractiveBrowserCredential, DefaultAzureCredential #, get_bearer_token_provider
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
#from langchain.document_loaders import Document
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
import os
import logging
from colorama import Fore, Style, init
import re
import pandas as pd
from langchain.vectorstores import FAISS
import json
import numpy as np
import pickle
import hashlib
from uuid import uuid4

def load_file(file_path):
    try:
        loader = TextLoader(file_path)

        docs = loader.load()
        print(Fore.GREEN + f"##[debug] Loaded file: {file_path}")
        # Note: If you're using PyPDFLoader then it will split by page for you already
        print(f'You have {len(docs)} document(s) in your data')
        print(f'There are {len(docs[0].page_content)} characters in your sample document')
        print(Fore.YELLOW + f'Here is a PAGE CONTENT: {docs[0].page_content[:200]}')
        print(Fore.MAGENTA + f'Here is a METADATA: {docs[0].metadata}')

        return docs
    except Exception as e:
        print(Fore.RED + f"##[error] Error loading file: {e}")
        return None

# Helper functions for CSV and FAISS processing (mocked for now)
def load_csv(file_path):
    try:
        print(Fore.GREEN + "##[debug]Loading CSV data...")
        df = pd.read_csv(file_path)
        print(Fore.GREEN + f"##[debug]CSV file loaded successfully: {file_path}")
        return df
    except Exception as e:
        print(Fore.RED + f"[##error] Error loading CSV: {e}")
        return None

def json_to_documents(json_file_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        documents = []
        for index, item in enumerate(data):
            page_content = ' '.join(str(value) for value in item.values())
            metadata = {key: value for key, value in item.items()}
            metadata['source'] = json_file_path
            metadata['record_number'] = index + 1

            documents.append(Document(page_content=page_content, metadata=metadata))

        print(Fore.GREEN + f"##[debug] Loaded {len(documents)} documents from JSON.")
        return documents
    except Exception as e:
        print(Fore.RED + f"##[error] Error reading JSON file: {e}")
        return []

def csv_to_documents(df, source_name):
    try:
        print(Fore.GREEN + "##[debug]Converting CSV data to documents...")
        documents = []
        for index, row in df.iterrows():
            #page_content = ' '.join(row.values)
            page_content = ' '.join([str(value) for value in row.values])
            #metadata = {key: value for key, value in row.items()}
            metadata = {str(key): str(value) for key, value in row.items()}
            metadata['source'] = source_name
            metadata['record_number'] = index + 1

            documents.append(Document(page_content=page_content, metadata=metadata))

        print(Fore.GREEN + f"Successfully converted {len(documents)} documents.")
        
        # print first 2 documents
        print(Fore.CYAN + f"##[debug]First 2 documents: {documents[:2]}")
        
        return documents
    except Exception as e:
        print(Fore.RED + f"[##error] Error converting CSV to documents: {e}")
        return []

def split_documents(documents, chunk_size=512, chunk_overlap=50):
    try:
        print(Fore.GREEN + "##[debug]Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []

        # Iterate over each document and split it
        for doc in documents:  # Assuming 'documents' is a list of strings
            if doc:
                doc_chunks = splitter.split_text(doc.page_content)
                chunks.extend(doc_chunks)

        print(Fore.GREEN + f"##[debug]Split into {len(chunks)} chunks")
        print(Fore.CYAN + f"##[debug]First 2 chunks: {chunks[:2]}")
        return chunks
    except Exception as e:
        print(Fore.RED + f"[##error] Error splitting documents: {e}")
        return []

def get_open_ai_token():
    try:
        print(Fore.GREEN + "##[debug]Getting OpenAI token...")
        credential = DefaultAzureCredential() #InteractiveBrowserCredential()
        token = credential.get_token("https://cognitiveservices.azure.com/.default")

        if not token:
            raise Exception("Failed to obtain a valid token.")

        print(Fore.GREEN + "##[debug]Token retrieved successfully!")
        return token.token
    except Exception as e:
        print(Fore.RED + f"[##error] Error getting OpenAI token: {e}")
        return None

def get_embeddings(token):
    try:
        print(Fore.GREEN + "##[debug]Initializing OpenAI Embeddings...")

        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_endpoint="https://carecenter-dev-openai.openai.azure.com/",
            api_key=token,
            openai_api_version="2023-03-15-preview",
        )

        print(Fore.GREEN + "##[debug]OpenAI Embeddings initialized successfully!")
        return embeddings
    except Exception as e:
        print(Fore.RED + f"[##error] Error initializing embeddings: {e}")
        return None

# Function to hash the chunks for easy identification
def hash_chunk(chunk):
    return hashlib.md5(chunk.encode('utf-8')).hexdigest()

import os
from datetime import datetime

import os
from datetime import datetime

def store_embeddings_in_faiss(token, chunks, faiss_index_path="medical_faiss_index", source_file_path="./datasets/pregHelp.txt"):
    try:
        print(Fore.GREEN + "##[debug]Storing embeddings in FAISS...")

        embeddings = get_embeddings(token)

        if not embeddings:
            raise ValueError("Failed to get embeddings. The embeddings object is None or empty.")

        print(Fore.YELLOW + "......................###########################..........................")

        update_index = False  # This flag will determine whether we need to recreate/update the FAISS index

        # Check if both the FAISS index and the source file exist
        if os.path.exists(faiss_index_path) and os.path.exists(source_file_path):
            # Get modification times for the source file and the FAISS index
            faiss_mtime = os.path.getmtime(faiss_index_path)
            source_mtime = os.path.getmtime(source_file_path)

            # Compare modification times to decide if the index should be updated
            if source_mtime > faiss_mtime:
                print(Fore.YELLOW + "Source file is newer than the FAISS index, updating FAISS index...")
                update_index = True
            else:
                # CHECK IF CURRENT TIME AND faiss_mtime IS WITHIN 30 MINUTES THEN UPDATE INDEX
                current_time = datetime.now().timestamp()
                
                if (current_time - faiss_mtime) < 1800:
                    print(Fore.YELLOW + "FAISS index is NEWER than 30 minutes, updating FAISS index...")
                    update_index = True
                else:
                    print(Fore.GREEN + "FAISS index is up-to-date. No need to update.")
        else:
            # If the FAISS index doesn't exist, we need to create it
            print(Fore.YELLOW + "FAISS index not found, creating a new one...")
            update_index = True

        # If the index needs to be updated or created
        if update_index:
            # If the FAISS index exists, load it, else create a new one
           
            # Prepare documents with metadata
            documents = [
                Document(page_content=chunk, metadata={"source": source_file_path})
                for chunk in chunks
            ]

            # Add indexing metadata to the documents
            #for doc in documents:
            #    doc.metadata["index"] = hash_chunk(doc.page_content)

            if os.path.exists(faiss_index_path):
                print(Fore.YELLOW + "Loading existing FAISS index...")
                vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

                # Get the number of vectors after appending
                current_vector_count = vector_store.index.ntotal
                print(Fore.GREEN + f"##[debug]Number of vectors before appending: {current_vector_count}")
                 # Add new documents to the FAISS index
                print(Fore.YELLOW + "Adding new documents to the FAISS index...")
                uuids = [str(uuid4()) for _ in range(len(documents))]
                vector_store.add_documents(documents=documents, ids=uuids)
            else:
                print(Fore.YELLOW + "No FAISS index found. Creating a new one from the source data.")
                vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

            # Save the updated FAISS index
            print(Fore.YELLOW + "Saving the updated FAISS index...")
            vector_store.save_local(faiss_index_path)

             # Get the number of vectors after appending
            update_vector_count = vector_store.index.ntotal
            print(Fore.YELLOW + f"##[debug]Number of vectors after appending: {update_vector_count}")

            print(Fore.GREEN + "##[debug]Embeddings stored successfully in FAISS!")
           
        else:
            # If no update is needed, load the existing FAISS index
            print(Fore.YELLOW + "Loading the existing FAISS index without updating...")
            vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        print(Fore.YELLOW + "......................###########################..........................")
        return vector_store

    except Exception as e:
        print(Fore.RED + f"##[error]Error storing embeddings in FAISS: {e}")



def store_embeddings_in_faiss2(token, chunks, faiss_index_path="medical_faiss_index"):
    try:
        print(Fore.GREEN + "##[debug]Storing embeddings in FAISS...")

        embeddings = get_embeddings(token)

        if not embeddings:
            raise ValueError("Failed to get embeddings. The embeddings object is None or empty.")

        print(Fore.YELLOW + "...........................................")
        # Initialize or load FAISS Vector Store
        if os.path.exists(faiss_index_path):
            # Load existing FAISS index
            
            print(Fore.YELLOW + "Loading existing FAISS index...")
            vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization= True)
        else:
            # Create a new FAISS index
            print(Fore.YELLOW + "Creating a new FAISS index...")
            vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        print(Fore.YELLOW + "...........................................")
        # Add new embeddings (if existing, they will be appended)
        vector_store.add_texts(texts=chunks)

        # Save the updated FAISS index
        vector_store.save_local(faiss_index_path)

        print(Fore.GREEN + "##[debug]Embeddings stored successfully in FAISS!")
        # Get the number of vectors after appending
        current_vector_count = vector_store.index.ntotal
        print(Fore.GREEN + f"##[debug]Number of vectors after appending: {current_vector_count}")

        return vector_store

    except Exception as e:
        print(Fore.RED + "##[error]Error storing embeddings in FAISS: {e}")

def create_faiss_rag(token, faiss_store, n=5):
    try:
        # Initialize the LLM model
        print(Fore.GREEN + "##[debug]Initializing LLM model...")

        llm_model = AzureChatOpenAI(
            azure_endpoint="https://carecenter-dev-openai.openai.azure.com/",
            azure_ad_token=token,
            api_version="2023-03-15-preview",
            deployment_name="gpt-35-turbo",
            model_name="gpt-35-turbo"
        )
        print(Fore.GREEN + "##[debug]LLM model initialized successfully.")

        print(Fore.GREEN + "##[debug]Initializing vector store retriever...")

        #retriever = vector_store.as_retriever(
        #    search_type="similarity", search_kwargs={"k": 5})

        retriever = faiss_store.as_retriever()

        print(Fore.GREEN + "##[debug]Initializing RAG chain...")

        # Create a retrieval-based QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=retriever,
            return_source_documents=True
        )
        logging.info(Fore.GREEN + "QA chain created successfully.")
        return qa_chain
    except Exception as e:
        print(Fore.RED + "##[error]Error performing RAG: {e}")
        return None

def display_query(query):
    try:
        print(f"\tQuery: {query}\n")
    except Exception as e:
        print(Fore.RED + "##[error]Error displaying query: {e}")


def display_result(result):
    try:
        formatted_result = result.replace("\n\n", " ").replace("\n", "")
        print(f"\tResult: {formatted_result}\n")
    except Exception as e:
        print(Fore.RED + "##[error] Error displaying result: {e}")


def display_source_documents(source_documents):
    try:
        print(f"\tSource Documents : {source_documents}")
        print(f"\tSource Documents count: {len(source_documents)}")
        for idx, doc in enumerate(source_documents):
            print(f"\tDocument {idx + 1}: {doc}")
    except Exception as e:
        print(Fore.RED + "##[error] Error displaying source documents: {e}")

def json_objects_to_strings(json_file_path):
    try:
        # Load JSON data
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Convert each object (dictionary) to a string
        string_objects = [json.dumps(obj) for obj in data]

        # display first 2 records
        print(Fore.GREEN + f"##[debug]First 2 records: {string_objects[:2]}")


        return string_objects
    except Exception as e:
        print(Fore.RED + "##[error] Error reading JSON file: {e}")
        return []

def get_formatted_response(results):
    try:
        # Display results
        print("-------------------------------------------------\n")
        print("Displaying results:\n")
        # Displaying the query, result, and source documents using the defined functions
        display_query(results.get("query", ""))
        display_result(results.get("result", ""))
        display_source_documents(results.get("source_documents", []))
        print("-------------------------------------------------\n")


        # Collect sources if available
        sources = []
        seen_content = set()  # To track unique page contents

        source_documents = results.get("source_documents", [])
        if source_documents:
            for idx, source_doc in enumerate(source_documents):
                page_content = source_doc.page_content
                metadata = source_doc.metadata  # If metadata is needed, you can include it too

                # Ensure that only unique page content is added
                if page_content and page_content not in seen_content:
                    seen_content.add(page_content)  # Add to seen set to avoid duplicates

                    # Print the content and metadata for logging purposes
                    if page_content:
                        print(Fore.YELLOW + f"\t\tPage Content {idx + 1}: {page_content}" + Fore.RESET)
                    if metadata:
                        print(Fore.MAGENTA + f"\t\tMetadata {idx + 1}: {metadata}" + Fore.RESET)

                    # Append the unique source to the sources list
                    sources.append({
                        "index": idx + 1,  # Including idx in the output (1-based index)
                        "page_content": page_content,
                        "metadata": metadata  # Optional, include if needed
                    })

        return sources
    except Exception as e:
        logging.error(Fore.RED + f"##[error] Error processing get_formatted_response: {e}")
        return []