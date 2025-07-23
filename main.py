import fitz  
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from openai import OpenAI

# تنظیم کلاینت جدید OpenAI
client = OpenAI(api_key="")  # یا "your-api-key" را اینجا بذار


# 1. استخراج متن از PDFها
def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

documents = []
for file in os.listdir("data/pdfs"):
    if file.endswith(".pdf"):
        content = extract_text_from_pdf(f"data/pdfs/{file}")
        if content.strip():
            documents.append({"filename": file, "content": content})
        else:
            print(f"Warning: file {file} has no text.")

        
    
# 2. تقسیم متن به چانک‌های کوچک‌تر
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

# 3. تعریف تابع گرفتن امبدینگ با OpenAI API


def embed(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 4. ساخت امبدینگ‌ها برای هر چانک
for entry in texts:
    entry["embedding"] = embed(entry["text"])

# 5. ساخت ایندکس FAISS برای جستجوی سریع
dimension = len(texts[0]["embedding"])
index = faiss.IndexFlatL2(dimension)

vectors = np.array([t["embedding"] for t in texts]).astype("float32")
index.add(vectors)

# 6. تابع جستجو در ایندکس
def search(query, top_k=5):
    query_vector = np.array(embed(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [texts[i] for i in indices[0]]
    return results

# 7. تولید پاسخ با GPT با استفاده از متن‌های بازیابی شده
def generate_answer(query):
    results = search(query)
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

# -- حالا می‌تونی این دو خط رو برای تست کنی --

if __name__ == "__main__":
    query = input("Ask your question: ")
    answer = generate_answer(query)
    print("\nAnswer:\n", answer)
