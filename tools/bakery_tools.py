# tools/bakery_tools.py
from app import RAGAssistant

def get_bakery_knowledge(query: str):
    assistant = RAGAssistant() 
    results = assistant.vector_db.search(query)
    return "\n\n".join(results["documents"])