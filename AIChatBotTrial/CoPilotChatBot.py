from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pymongo import MongoClient
import os

class ChatGPTLikeChatbot:
    def __init__(self, retriever_model_name, generator_model_name, mongo_uri, database_name, collection_name):
        
        self.retriever = SentenceTransformer(retriever_model_name)
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

        
        self.documents = self._fetch_documents(mongo_uri, database_name, collection_name)

        
        self.index, self.document_embeddings, self.doc_map = self._build_index(self.documents)

    def _fetch_documents(self, mongo_uri, database_name, collection_name):
        try:
            client = MongoClient(mongo_uri)
            db = client[database_name]
            collection = db[collection_name]
            documents = [
                {
                    "title": doc.get("title", ""),
                    "brand": doc.get("brand", ""),
                    "categories": doc.get("categories", ""),
                    "review": doc.get("top_review", ""),
                    "price": doc.get("final_price", ""),
                    "description": doc.get("description", "")
                }
                for doc in collection.find({}, {"title": 1, "categories": 1, "review": 1, "final_price": 1, "description": 1, "brand": 1, "_id": 0})
            ]
            client.close()
            return documents
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return []

    def _build_index(self, documents):
        try:
            embeddings = []
            batch_size = 64
            for i in range(0, len(documents), batch_size):
                batch_embeddings = self.retriever.encode([doc["description"] for doc in documents[i:i+batch_size]], convert_to_tensor=False)
                embeddings.extend(batch_embeddings)
            embeddings = np.array(embeddings)
            doc_map = {i: doc for i, doc in enumerate(documents)}
            
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            return index, embeddings, doc_map
        except Exception as e:
            print(f"Error building index: {e}")
            return None, None, {}

    def retrieve(self, query, top_k=5):
        try:
            query_embedding = self.retriever.encode([query], convert_to_tensor=False)
            query_embedding = np.array(query_embedding)
            
            distances, indices = self.index.search(query_embedding, top_k)
            retrieved_docs = [self.doc_map[i] for i in indices[0]]
            return retrieved_docs
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def generate(self, query, retrieved_docs):
        try:
            
            context = "\n".join([doc["description"] for doc in retrieved_docs])
            input_text = f"Question: {query}\nContext: {context}\nAnswer:"
            
            
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
            outputs = self.generator.generate(**inputs, max_length=500, num_beams=5)
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response."

    def chat(self, query):
        retrieved_docs = self.retrieve(query)
        answer = self.generate(query, retrieved_docs)
        return answer

if __name__ == "__main__":
    
    mongo_uri = os.getenv("your_own_mongo_uri")  
    database_name = "vector_db"  
    collection_name = "vector_collection"  

    
    chatbot = ChatGPTLikeChatbot(
        retriever_model_name="sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name="t5-small",
        mongo_uri=mongo_uri,
        database_name=database_name,
        collection_name=collection_name
    )

    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        response = chatbot.chat(user_query)
        print(f"Bot: {response}")
