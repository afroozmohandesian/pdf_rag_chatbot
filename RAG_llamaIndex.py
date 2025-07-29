
import os
import faiss
import numpy as np
from openai import OpenAI

from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.text_splitter import SentenceSplitter

client = OpenAI(api_key="")

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_documents_from_folder(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            content = extract_text_from_txt(os.path.join(folder_path, file))
            if content.strip():
                documents.append({"filename": file, "content": content})
            else:
                print(f"Warning: file {file} has no text.")
    return documents

def split_documents(documents):
    splitter = SentenceSplitter()
    texts = []
    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            texts.append({
                "text": chunk,
                "meta": {
                    "source": doc["filename"],
                    "chunk": i
                }
            })
    return texts

def embed(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def build_faiss_index(texts):
    for entry in texts:
        entry["embedding"] = embed(entry["text"])

    dimension = len(texts[0]["embedding"])
    index = faiss.IndexFlatL2(dimension)

    vectors = np.array([t["embedding"] for t in texts]).astype("float32")
    index.add(vectors)
    return index

def search(index, query, texts, top_k=5):
    query_vector = np.array(embed(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [texts[i] for i in indices[0]]
    return results

def generate_answer(index, texts, query):
    results = search(index, query, texts)
    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
Answer the question below using the context provided.

### Context:
{context}

### Question:
{query}

### Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    folder_path = "data/txts"
    documents = load_documents_from_folder(folder_path)

    if not documents:
        print("❌ No .txt files found in the folder.")
        exit()

    texts = split_documents(documents)
    index = build_faiss_index(texts)

    while True:
        query = input("\nAsk your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = generate_answer(index, texts, query)
        print("\nAnswer:\n", answer)


# استخراج متن از فایل‌های متنی (.txt)
def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# خواندن تمام فایل‌های متنی از پوشه مشخص‌شده
def load_documents_from_folder(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if not file.endswith(".txt"):
            print(f"Warning: file {file} is not a .txt file and will be skipped.")
            continue
        content = extract_text_from_txt(os.path.join(folder_path, file))
        if content.strip():
            documents.append({"filename": file, "content": content})
        else:
            print(f"Warning: file {file} has no text.")
    return documents

# Sliding window chunking based on sentences
def split_documents_sentence_window(documents, window_size=5, overlap=2):
    parser = SentenceSplitter()

    all_text_chunks = []
    chunk_id = 0

    for doc in documents:
        llama_doc = LlamaDocument(text=doc["content"], metadata={"source": doc["filename"]})
        sentence_nodes = parser.get_nodes_from_document(llama_doc)  # list of nodes (one sentence per node)

        # create sliding windows of sentences
        start = 0
        while start < len(sentence_nodes):
            end = start + window_size
            window_nodes = sentence_nodes[start:end]

            # combine sentences in window into one chunk text
            chunk_text = " ".join(node.text for node in window_nodes)

            all_text_chunks.append({
                "text": chunk_text,
                "meta": {
                    "source": doc["filename"],
                    "chunk": chunk_id
                }
            })

            chunk_id += 1
            # slide window forward by (window_size - overlap)
            start += window_size - overlap if (window_size - overlap) > 0 else 1

    return all_text_chunks

# گرفتن امبدینگ با OpenAI
def embed(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# ساخت ایندکس FAISS
def build_faiss_index(texts):
    for entry in texts:
        entry["embedding"] = embed(entry["text"])

    dimension = len(texts[0]["embedding"])
    index = faiss.IndexFlatL2(dimension)

    vectors = np.array([t["embedding"] for t in texts]).astype("float32")
    index.add(vectors)
    return index

# جستجو با FAISS
def search(index, query, texts, top_k=5):
    query_vector = np.array(embed(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [texts[i] for i in indices[0]]
    return results

# تولید پاسخ با GPT از متن‌های بازیابی شده
def generate_answer(index, texts, query):
    results = search(index, query, texts)
    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
Answer the question below using the context provided.

### Context:
{context}

### Question:
{query}

### Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# -- اجرای کامل برنامه --
if __name__ == "__main__":
    folder_path = "data/txts"  # ← مسیر پوشه فایل‌های متنی
    documents = load_documents_from_folder(folder_path)

    if not documents:
        print("❌ No .txt files found in the folder.")
        exit()

    texts = split_documents_sentence_window(documents, window_size=5, overlap=2)
    index = build_faiss_index(texts)

    while True:
        query = input("\nAsk your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = generate_answer(index, texts, query)
        print("\nAnswer:\n", answer)
