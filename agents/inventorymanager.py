# agents/inventory.py
from langchain_core.messages import HumanMessage

class InventoryManager:
    def __init__(self, llm, tools):
        # On lie Tavily au LLM
        self.model = llm.bind_tools(tools)
        self.tools_map = {tool.name: tool for tool in tools}

    def run(self, state):
        print("--- AGENT GESTIONNAIRE : RECHERCHE DE PRIX RÉELS ---")
        recipe = state.get('recipe_proposal', "")
        
        prompt = f"""Tu es la gestionnaire des finances Safiatou DIALLO. 
        1. Utilise Tavily pour chercher le prix actuel des ingrédients principaux de cette recette : {recipe}.
        2. Calcule un prix de vente conseillé (coût ingrédients x 3).
        3. Tu dois impérativement répondre en suivant ce format exact :
            - PRIX DES DEPENSES TOTAL : [Montant]€
            - PRIX DE VENTE CONSEILLÉ : [Montant]€
            - BÉNÉFICE ESTIMÉ : [Montant]€
            - CLIENTS CIBLES : [Description]
            - LIEUX DE VENTE : [Description]

        Sois concis, ne donne pas de détails techniques de recherche."""

        # L'IA va détecter qu'elle doit utiliser 'tavily_search_results_json'
        response = self.model.invoke([HumanMessage(content=prompt)])
        
        if response.tool_calls:
            results = []
            for tool_call in response.tool_calls:
                tool = self.tools_map[tool_call["name"]]
                out = tool.invoke(tool_call["args"])
                results.append(str(out))
            
            # On renvoie les données brutes de Tavily au State
            # (Ou on peut refaire un appel au LLM pour qu'il synthétise, c'est encore mieux)
            analysis = f"Données du marché trouvées : {results}"
        else:
            analysis = response.content

        return {"financials": analysis}