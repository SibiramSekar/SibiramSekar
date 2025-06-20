from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np
from pymongo import MongoClient
import time

class ColbertChatbot:
    def __init__(self, retriever_model_name, generator_model_name, mongo_uri, database_name, collection_name):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        
        self.tokenizer = BertTokenizer.from_pretrained(retriever_model_name)
        self.model = BertModel.from_pretrained(retriever_model_name).to(device)

        
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name).to(device)

        self.device = device
        print("Models initialized.")
        
        self.documents = self._fetch_documents(mongo_uri, database_name, collection_name)
        print(f"Fetched {len(self.documents)} documents.")
        
        self.index, self.doc_map = self._build_index(self.documents)
        print("Index built.")

    def _fetch_documents(self, mongo_uri, database_name, collection_name):
        client = MongoClient(mongo_uri)
        db = client[database_name]
        collection = db[collection_name]
        documents = [
            {
                "title": doc.get("title", ""),
                "brand": doc.get("brand", ""),
                "categories": doc.get("categories", []),
                "review": doc.get("top_review", ""),
                "price": doc.get("final_price", ""),
                "description": doc.get("description", ""),
            }
            for doc in collection.find({}, {"_id": 0})
        ]
        client.close()
        return documents

    def _build_index(self, documents):
        
        text_documents = [f"{doc['title']} {doc['brand']} {doc['categories']} {doc['description']}" for doc in documents]
        embeddings = self._encode_documents_in_batches(text_documents)
        
        
        index = faiss.IndexFlatL2(embeddings.shape[1])  
        index.add(np.array(embeddings, dtype=np.float32))
        doc_map = {i: doc for i, doc in enumerate(documents)}
        return index, doc_map

    def _encode_documents_in_batches(self, documents, batch_size=8):
        
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings_batch = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(embeddings_batch)
        
        return np.vstack(embeddings)

    def retrieve(self, query, top_k=5):
        
        query_embedding = self._encode_documents_in_batches([query])[0]  
        distances, indices = self.index.search(np.array([query_embedding]), top_k)

        
        max_distance = np.max(distances)
        scores = [(1 - (distance / max_distance)) * 100 for distance in distances[0]]

        
        retrieved_docs_with_scores = [
            {"doc": self.doc_map[i], "score": score}
            for i, score in zip(indices[0], scores)
        ]
        return retrieved_docs_with_scores

    def format_context(self, docs_with_scores):
        context = []
        for entry in docs_with_scores:
            doc = entry["doc"]
            score = entry["score"]
            
            
            description = doc["description"][:300] + "..." if len(doc["description"]) > 300 else doc["description"]
            
            formatted = (
                f"Title: {doc['title']}\n"
                f"Brand: {doc['brand']}\n"
                f"Price: {doc['price']}\n"
                f"Description: {description}\n"
                f"Relevance: {score:.2f}%\n"
            )
            context.append(formatted)
        return "\n\n".join(context)

    def generate(self, query, context):
        input_text = f"Question: {query}\nContext:\n{context}\nAnswer:"
        inputs = self.generator_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.generator.generate(**inputs, max_length=250, num_beams=5)
        return self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def chat(self, query):
        start_time = time.time()
        print(f"Received query: {query}")
        
        retrieved_docs_with_scores = self.retrieve(query)
        if not retrieved_docs_with_scores:
            return "I'm sorry, I couldn't find relevant information."
        
        context = self.format_context(retrieved_docs_with_scores)
        response = self.generate(query, context)
        
        print(f"Response generated in {time.time() - start_time:.2f} seconds.")
        return response

if __name__ == "__main__":
    
    mongo_uri = "your_own_mongo_uri"
    database_name = "vector_db"
    collection_name = "vector_collection"

    chatbot = ColbertChatbot(
        retriever_model_name="bert-base-uncased",  
        generator_model_name="t5-small",  
        mongo_uri=mongo_uri,
        database_name=database_name,
        collection_name=collection_name,
    )

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        response = chatbot.chat(user_query)
        print(f"Bot: {response}")
