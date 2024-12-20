# Task-2
# Chat with website using rag pipline
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Scrape your website content
url = "https://www.stanford.edu/"  # Replace with your website URL
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Assuming you're scraping paragraphs (<p>) or similar elements
content = soup.find_all('p')  # You can modify this based on your website structure

# Extract text from the paragraphs
website_text = [p.get_text() for p in content if p.get_text().strip()]

# Check if content has been successfully extracted
if len(website_text) == 0:
    print("No content found on the website.")
else:
    print(f"Found {len(website_text)} documents to process.")

# Step 2: Initialize the model and encode the documents
model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = model.encode(website_text)

# Step 3: Create a FAISS index and add the document embeddings
dimension = document_embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(np.array(document_embeddings, dtype=np.float32))  # Add the document embeddings to the index

# Step 4: Query from user (e.g., asking about the company's mission)
query = "When it is founded?"  # Replace this with any query
query_embedding = model.encode([query])

# Step 5: Retrieve top k relevant documents
k = 3  # Number of documents you want to retrieve
D, I = index.search(np.array(query_embedding, dtype=np.float32), k)

# Step 6: Ensure the retrieved indices are valid and print the documents
if len(I[0]) == 0:
    print("No relevant documents found.")
    response = "Sorry, I couldn't find relevant information."
else:
    # If fewer than k results are retrieved, adjust k
    if len(I[0]) < k:
        print(f"Only {len(I[0])} relevant documents found instead of {k}.")
        k = len(I[0])

    try:
        # Retrieve the corresponding documents
        retrieved_documents = [website_text[i] for i in I[0]]
        print("Retrieved Documents:", retrieved_documents)

        # Generate a response (you can combine the documents as a response)
        response = " ".join(retrieved_documents)

    except IndexError as e:
        print("IndexError:", e)
        response = "Sorry, there was an issue retrieving the relevant documents."

# Final output
print("Response:", response)
