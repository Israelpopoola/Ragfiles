# rag_chatbot_free_files.py
# Free RAG chatbot for church FAQs and pastor info
# Uses Hugging Face models + FAISS vector search
# Loads all .txt files in the folder

import os  # for listing files in the folder
from sentence_transformers import SentenceTransformer  # creates embeddings for text
from transformers import pipeline  # runs a small text generation model
import faiss  # stores and searches vector embeddings quickly

# ---- STEP 1: Load all .txt files in the folder ----
# Go through every file in this folder
# If the file ends with .txt, open it and read its content
# Split content by blank lines (each Q&A block)
# Extract question (Q:) and answer (A:) from each block
faq_text = []
for filename in os.listdir("."):
    if filename.endswith(".txt"):
        with open(filename, "r", encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")  # each block separated by blank line
            for block in blocks:
                lines = block.split("\n")
                if len(lines) >= 2:
                    # Remove "Q: " and "A: " from lines and strip spaces
                    q = lines[0].replace("Q: ", "").strip()
                    a = lines[1].replace("A: ", "").strip()
                    # store as tuple (question, answer)
                    faq_text.append((q, a))

# ---- STEP 2: Convert questions into vectors (embeddings) ----
# This allows the bot to find the closest matching question later
embedder = SentenceTransformer("all-MiniLM-L6-v2")
questions = [q for q, a in faq_text]  # list of all questions
answers = [a for q, a in faq_text]  # corresponding answers
embeddings = embedder.encode(questions)  # turn questions into vector numbers

# ---- STEP 3: Store embeddings in FAISS ----
# FAISS is like a smart search index for vectors
dim = embeddings.shape[1]  # dimension of each vector
index = faiss.IndexFlatL2(dim)  # create a flat (simple) index
index.add(embeddings)  # add all question embeddings to the index

# ---- STEP 4: Small model to optionally rephrase answers ----
# This model can make the answer sound a bit more natural
qa_model = pipeline("text-generation", model="distilgpt2")

# ---- STEP 5: Simple chatbot loop ----
print("Free Church FAQ Chatbot (type 'exit' to quit)")
while True:
    query = input("\nYou: ")  # get user input
    if query.lower() in ["exit", "quit", "q"]:  # exit conditions
        print("Goodbye!")
        break

    # Convert user question into a vector
    query_vec = embedder.encode([query])

    # Find the closest question from our index
    D, I = index.search(query_vec, k=1)  # k=1 → closest match
    retrieved_answer = answers[I[0][0]]  # get corresponding answer

    # Optionally rephrase the answer with small GPT model
    response = qa_model(
        f"Q: {query}\nA: {retrieved_answer}",
        max_length=80,
        do_sample=True
    )[0]["generated_text"]

    # Print the answer (strip everything before "A:")
    print("Bot:", response.split("A:")[-1].strip())
