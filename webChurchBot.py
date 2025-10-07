# py
# Freeform RAG chatbot for church info using GPT-4.1-mini

import streamlit as st
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- STEP 1: Load all .txt files and split into chunks ----
all_chunks = []

# for(open) all the files in the current directory
for filename in os.listdir("."):
    # (called filename) in read mode ("r") using UTF-8 encoding (so all letters, symbols, accents display correctly).
    # While the file is open, call it f. When I’m done inside this block, automatically close the file for me.

    if filename.endswith(".txt"):
        # Look at every file in this folder, one at a time, and call it filename.
        # Then check if it’s a .txt file before reading it.”

        with open(filename, "r", encoding="utf-8") as f:
            # ""“Open this text file (called filename) in read mode ("r")
            #  using UTF-8 encoding (so all letters, symbols, accents display correctly).
            #  While the file is open, call it (temporary name (f)).
            # When I’m done inside this block, automatically close the file for me.”"""

            # Without 'with' , you’d have to write file.close() yourself.
            # If you forget, the file might stay open and cause issues.
            # 'with' handels opening and closing the file for you

            text = f.read().strip()  # read and strip it of all the white extra white space
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300, chunk_overlap=20)  # chunk it
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            # Take all chunks and make it a flat list .extend() instead of a nested list .append()

# ---- STEP 2: Convert chunks into embeddings ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(
    all_chunks, convert_to_numpy=True, show_progress_bar=True, padding=True, truncation=True
)

# ---- STEP 3: Store embeddings in FAISS ----
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# ---- STEP 4: Set OpenAI API key from environment variable ----
openai.api_key = "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"
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY not found")

# ---- STEP 5: Chatbot loop ----
st.subheader("MFM Dallas Texas Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []


for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**MFM Bot:** {a}")
    st.markdown("---")  # horizontal line between exchanges

query = st.text_input("You: ")

if query:

    with st.spinner("Bot is Thinking..."):
        # Embed the user question
        query_vec = embedder.encode([query], padding=True, truncation=True)

        # Retrieve top "k" closest chunks
        D, I = index.search(query_vec, k=2)
        retrieved_chunk = "\n---\n".join([all_chunks[i] for i in I[0]])

        # Construct messages with strict instructions
        messages = [
            {
                "role": "system",
                "content": (
                        "You are a helpful assistant providing information about the church (MFM Dallas Texas). "
                        "You MUST answer ONLY using the information provided in the assistant role. "
                        "Do NOT make up any information or reference anything outside of that context. "
                        "Answer should be brief, maximum two sentences. "
                        "If the answer is not contained in the context, respond: "
                        "'I don't know. Please contact support@mfmdallastexas.com.'"
                )
            },
            {"role": "assistant", "content": retrieved_chunk},
            {"role": "user", "content": query}
        ]

        # Call OpenAI GPT-4.1-mini model
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            max_tokens=130,
            temperature=0.0
        )

        # Extract and print the answer
        answer = response.choices[0].message.content.strip()
        st.write("MFM Bot:", answer)

        # Add to chat history
        st.session_state.history.append((query, answer))
