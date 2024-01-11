import pinecone
import requests
from dotenv import load_dotenv
import os

load_dotenv()

def get_openai_embedding(input_text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    data = {
        "input": input_text,
        "model": "text-embedding-ada-002",
        "encoding_format": "float",
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result['data'][0]['embedding']
    else:
        print("Error:", response.status_code, response.text)
        return None

def upload_embedding_to_pinecone(employee_name, title, description):
    # Generate embeddings using OpenAI API
    input_text = f"{description}"
    embedding = get_openai_embedding(input_text)

    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv('PINECONE_ENVIRONMENT'))

    # Create an index (if not already created)
    index_name = "consultants"
    # Define metadata
    metadata = {'name': employee_name, 'title': title, 'description': description}

    # Upsert the embedding and metadata into the Pinecone index
    pinecone.Index(index_name).upsert(
        vectors=[{"id": employee_name, "values": embedding, "metadata": metadata}],
        namespace="testing"
    )
    print(f"Embedding for {employee_name} uploaded successfully.")

def query_pinecone(query):
    embeddings = get_openai_embedding(query)
    # Initialize Pinecone
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv('PINECONE_ENVIRONMENT'))

    # Create an index (if not already created)
    index_name = "consultants"
    index = pinecone.Index(index_name)

    # Query the Pinecone index
    result = index.query(vector=embeddings,
    top_k=10,
    include_metadata=True,
    namespace="testing")

    return result['matches']
    


employee_name = "Aisha Patel"
title = "Content Writer and SEO Specialist"
description = '''Background: Aisha has a degree in English Literature and 5 years of experience in content creation and SEO. She has a proven track record of improving website rankings through strategic content development.
Skills:
Exceptional writing and editing skills with a focus on SEO optimization.
Proficient in using content management systems (WordPress, Drupal).
Experience with keyword research and implementation.
Familiarity with Google Analytics and other SEO tools to track and analyze website performance.'''

# upload_embedding_to_pinecone(employee_name, title, description, api_key_openai)

query = '''I need a web developer with experience in React and Node.js.'''
print(query_pinecone(query))