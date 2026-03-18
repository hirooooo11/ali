import json

class Retriever:
    def __init__(self, kb_path="data/knowledge_base.json"):
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.kb = json.load(f)
            
    def search(self, question, top_k=3):
        clean_question = question.replace("？", "").replace("?", "").replace(" ", "")
        keywords = set(clean_question)
        
        scored_passages = []

        for record in self.kb:
            score = sum(1 for kw in keywords if kw in record["content"])
            if score > 0:
                scored_passages.append({
                    "score": score,
                    "source": record["source"],
                    "content": record["content"]
                })
                

        scored_passages.sort(key=lambda x: x["score"], reverse=True)
        

        return [{"source": item["source"], "content": item["content"]} for item in scored_passages[:top_k]]