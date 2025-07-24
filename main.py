import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from openai import OpenAI

# تنظیم کلاینت جدید OpenAI
client = OpenAI(api_key="")  # یا "your-api-key" را اینجا بذار


#  استخراج متن از فایل‌های متنی (.txt)
def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
         return f.read()

# خواندن تمام فایل‌های متنی از پوشه مشخص‌شده
def load_documents_from_folder(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if not file.endswith(".txt"):
            print(f"Warning: file {file} is not a .txt file and will be skipped.")
        if file.endswith(".txt"):
            content = extract_text_from_txt(os.path.join(folder_path, file))
            if content.strip():
                documents.append({"filename": file, "content": content})
            else:
                print(f"Warning: file {file} has no text.")
    return documents

       
    
# 2. تقسیم متن به چانک‌های کوچک‌تر
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

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

# 3. تعریف تابع گرفتن امبدینگ با OpenAI API


def embed(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 4. ساخت ایندکس FAISS
def build_faiss_index(texts):
    for entry in texts:
        entry["embedding"] = embed(entry["text"])

    dimension = len(texts[0]["embedding"])
    index = faiss.IndexFlatL2(dimension)

    vectors = np.array([t["embedding"] for t in texts]).astype("float32")
    index.add(vectors)
    return index


# 5. ساخت ایندکس FAISS برای جستجوی سریع
def search(index, query, top_k=5):
    query_vector = np.array(embed(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [texts[i] for i in indices[0]]
    return results



# 7. تولید پاسخ با GPT با استفاده از متن‌های بازیابی شده
def generate_answer(index, texts, query):
    results = search(index, query)
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

# -- اجرای کامل برنامه--

if __name__ == "__main__":
    folder_path = "data/txts"  # ← مسیر پوشه فایل‌های متنی
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