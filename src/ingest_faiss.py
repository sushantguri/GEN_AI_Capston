import os
import PyPDF2
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def ingest():
    print("Ingesting PDFs into FAISS...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pdf_paths = [
        "pdf1.pdf", "pdf2.pdf", "pdf3.pdf", 
        "pdf4.pdf", "pdf5.pdf", "pdf6.pdf"
    ]
    
    docs = []
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
        try:
            reader = PyPDF2.PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Increased chunk size to 1000 for better "big picture" understanding
                    for i in range(0, len(text), 800):
                        docs.append(text[i:i+1000])
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    if not docs:
        print("No documents were processed!")
        return
        
    print(f"Extracted {len(docs)} chunks. Building embeddings...")
    embeddings = model.encode(docs)
    
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    
    # Save the index and the documents so app.py / agent.py can load them
    os.makedirs('models', exist_ok=True)
    faiss.write_index(index, 'models/faiss_index.bin')
    with open('models/docs.pkl', 'wb') as f:
        pickle.dump(docs, f)
        
    print("Ingestion complete. Saved to models/faiss_index.bin and models/docs.pkl")

if __name__ == "__main__":
    ingest()
