# agents/quality.py
from langchain_core.messages import HumanMessage

class QualityAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state):
        proposal = state.get('recipe_proposal', "Aucune recette fournie.")
        
        if not proposal or proposal == "I'm sorry, that information is not in this document.":
            return {"safety_report": "Analyse impossible : aucune recette valide à examiner."}

        prompt = f"""
        Tu es un expert en sécurité alimentaire. 
        Analyse la proposition du Chef ci-dessous et identifie TOUS les allergènes potentiels.
        
        PROPOSITION DU CHEF :
        {proposal}
        
        CONSIGNE : Liste les allergènes en **GRAS ET MAJUSCULES**. 
        Si aucun allergène n'est présent, dis 'RAS'.
        """
        
        try:
            # Utilisation d'une liste de messages pour plus de compatibilité
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # On s'assure de retourner le contenu texte
            return {"safety_report": response.content}
        except Exception as e:
            print(f"Erreur dans QualityAgent : {e}")
            return {"safety_report": f"Erreur lors de l'analyse : {str(e)}"}