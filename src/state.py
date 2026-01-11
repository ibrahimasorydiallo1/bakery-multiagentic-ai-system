from typing import TypedDict, List

class BakeryState(TypedDict):
    question: str
    context: str        # Le texte extrait du RAG
    recipe_proposal: str # La réponse du Chef
    financials: str      # L'analyse du Gestionnaire
    safety_report: str    # Le rapport d'allergènes