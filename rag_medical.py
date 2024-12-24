# Please add an authentication token to the request header to get the response.

# Please compare token provided by the user with the token provided by the system (system token: 111-1111-11111).

# if token is not correct give forbidden error.




from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from flask_restx import Api, Resource, fields
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from azure.identity import InteractiveBrowserCredential, DefaultAzureCredential #, get_bearer_token_provider
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import logging
from colorama import Fore, Style, init
import re
import pandas as pd
from langchain.vectorstores import FAISS
import json
import markdown
import numpy as np
from faiss_lib import load_file, json_to_documents, csv_to_documents, split_documents, get_open_ai_token, store_embeddings_in_faiss, create_faiss_rag, get_formatted_response


# Initialize Colorama for colored logging
init(autoreset=True)

# Set up logging
logging.basicConfig(filename='myapp.log', level=logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Create Flask app
app = Flask(__name__)


# Create API instance for Swagger documentation
api = Api(
    app,
    version="1.0",
    title="CareCircle RAG API",
    description="REST API for Retrieval-Augmented Generation (RAG) Powered Searchable Database for CareCircle to answer the questions raised by users in the ChatBot.",
    doc="/swagger",  # Swagger UI URL endpoint
    defaultModelsExpandDepth=1,  # Expand default namespace
    defaultModelExpandDepth=1    # Expand model definitions
)

# Enable CORS for all domains on all routes
CORS(app)

# Define models for Swagger documentation
query_model = api.model('Query', {
    'query': fields.String(required=True, description="User's query from the chatbot.", example="Who is CTO of CareCircle?")
})

# Define the response model
response_model = api.model('Response', {
    'content': fields.String(required=True, description='Detailed response content', example=""),
    'role': fields.String(required=True, description='Role of the responder', example="assistant")
})

# System token for authentication
SYSTEM_TOKEN = "111-1111-11111"

# API Endpoint
@api.route('/query')
class QueryEndpoint(Resource):
    @api.expect(query_model)
    @api.doc(description="Executes a query using Retrieval-Augmented Generation (RAG) for CareCircle Data to provide accurate answers to user questions based on the organization's knowledge base.")
    def post(self):
        try:

            logging.info(Fore.CYAN + "##[section] Received a request from Chatbot App...")
            data = request.json
            question = data.get('query', '')

            logging.info(Fore.YELLOW + f"##[debug] User query: {question}")
            
            '''
            # Get the token from request headers
            auth_token = request.headers.get('Authorization')

            # Check if token matches the system token
            if auth_token != SYSTEM_TOKEN:
                logging.error(Fore.RED + "##[error] Invalid token provided.")
                return jsonify({"error": "Forbidden: Invalid token"}), 403

            '''
            prompt = get_chat_context(question)

            # Execute the query with updated parameters
            results = qa_chain({
                "query": question,
                "chat_history": prompt,
            })

            print(Fore.CYAN + f"Results from QA Chain: {results}\n")

            sources = get_formatted_response(results)

            

            print(Fore.GREEN + "##[debug]RAG completed successfully!")
            # Return the formatted result as JSON, including sources if available
            return jsonify({
                    "role": "assistant",
                    "content": md_to_html(results.get("result", "")),
                    "sources": sources
                })

        except Exception as e:
            logging.error(Fore.RED + f"##[error] Error processing query: {e}")
            return jsonify({"error": str(e)}), 500

def get_chat_context(question):
    # Define a more concise and professional prompt template
    template = f"""
            You are a support assistant for a pregnancy help organization called CareCircle.
            Your role is to provide clear, concise, and accurate answers to the user's questions using the organization's knowledge base.
    
            Important notes:
            - If you don't know the answer, simply state that you don't know, and avoid attempting to create an answer.
            - Provide only helpful and precise answers, such as the services offered by the organization.
            - If the question is unclear or ambiguous, ask for clarification before providing an answer.
            - Ask for more details if needed to provide a more accurate response.
            - Avoid sharing confidential or sensitive information about the organization as well as patient.
            - Obey HIPAA regulations and maintain patient confidentiality at all times.
            - Ask for the pregnant lady's name, age, and month of pregnancy for proper guidance.
            - Ask to drop an email to info@CareCircle.com for more details.
            - For emergency conditions, please contact 911 emergency services immediately.
            - For any medical issues, please contact the medical team at medical@CareCircle.com
            - For any counseling-related queries, please contact the counseling team at counseling@CareCircle.com
            - For any other queries, please contact the general support team at info@CareCircle.com
    
            Question: {question}
    
            Please provide the most accurate and helpful answer below:
            Helpful answer:
        """

    # Generate the final prompt with the user's question
    prompt = template.format(question=question)

    # Display the generated prompt
    print(Fore.GREEN + f"Prompt generated: {prompt}")

    # Create the chat message context, with the system messages guiding the assistant
    messages = [
        {"role": "system", "content": "CareCircle is a dedicated medical group providing comprehensive support to pregnant women and their families. Our mission is to ensure that every mother receives the care and guidance she needs throughout her pregnancy journey. We offer a range of services designed to meet the physical, emotional, and educational needs of expectant mothers."},
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]

    # Display the chat message for debugging/logging purposes
    print(Fore.YELLOW + f"Chat message generated: {messages}")

    return messages



def md_to_html(md_content):
    """
    Convert Markdown content to HTML.

    Args:
    - md_content (str): A string containing Markdown formatted text.

    Returns:
    - str: The HTML representation of the Markdown content.
    """
    html_content = markdown.markdown(md_content)
    return html_content

def init_qa_chain():
    try:
        # Define file paths and flags for loading
        knowledgebases = [
            {
                "path": "./datasets/CareCircle.txt",
                "use": True,
                "type": "company"
            },
            {
                "path": "./datasets/pregnancy.txt",
                "use": True,
                "type": "medical"
            }
        ]

        # Ensure vector_store is defined before this block
        vector_store = None  # or the appropriate logic to obtain the vector_store
        index_name = "medical_faiss_index"
        token = get_open_ai_token()

        for kb in knowledgebases:
            if kb["use"]:
                chunks = []
                print(Fore.BLUE + f"Loading {kb['type']} data...")
                if kb["type"] == "product":
                    documents = json_to_documents(kb["path"])
                else:
                    file_extension = os.path.splitext(kb["path"])[1].lower()
                    if file_extension == ".json":
                        documents = json_to_documents(kb["path"])
                    elif file_extension in [".txt", ".csv", ".xlsx"]:  # Add other extensions as needed
                        documents = load_file(kb["path"])
                    else:
                        raise ValueError(f"Unsupported file extension: {file_extension}")
                if documents is not None and len(documents) > 0:
                    # Split Documents into Chunks
                    document_chunks = split_documents(documents)
                    print(Fore.YELLOW + f"Number of {kb['type']} chunks: {len(document_chunks)}")
                    print(Fore.MAGENTA + f"First 2 {kb['type']} chunks: {document_chunks[:2]}")
                    chunks.extend(document_chunks)

                if not chunks:
                    print(Fore.RED + f"##[warning] No chunks loaded for {kb['path']}. Exiting initialization.")
                    #exit if loop and go to while loop
                    continue


                batch_size = 500  # Adjust the batch size based on your rate limits
                start_index = 0
                total_chunks = len(chunks)
                print(Fore.CYAN + f"##[debug]Number of chunks: {len(chunks)} in {kb['path']}")

                while start_index < total_chunks:
                    # Get the current batch
                    end_index = min(start_index + batch_size, total_chunks)
                    current_batch = chunks[start_index:end_index]
                    # Store embeddings for the current batch
                    vector_store = store_embeddings_in_faiss(token, current_batch, index_name, kb["path"])
                    print(Fore.CYAN + f"Processed batch of {kb['path']}:{start_index} to {end_index}")

                    # Move to the next batch
                    start_index = end_index

                print(Fore.GREEN + f"COMPLETED: Stored embeddings for {kb['type']} from {kb['path']}.")

        if vector_store:
            qa_chain = create_faiss_rag(token, vector_store)
            if qa_chain:
                print(Fore.GREEN + "##[debug]QA Chain initialized successfully!")
                return qa_chain
            else:
                print(Fore.RED + "##[error] Failed to initialize QA Chain.")
                return None
    except Exception as e:
        logging.error(Fore.RED + f"##[error] An error occurred during initialization: {e}")
        raise


# Initialize the QA chain
qa_chain = init_qa_chain()


# Start Flask app
if __name__ == '__main__':
    try:
        #app.run(debug=True)
        logging.info(Fore.CYAN + "##[section] Starting Flask application...")
        print(Fore.CYAN + f"##[debug] Running REST API on port: 5000...")
        app.run(host='localhost', port=5000)
    except Exception as e:
        logging.error(Fore.RED + f"##[error] An error occurred while starting the Flask application: {e}")
