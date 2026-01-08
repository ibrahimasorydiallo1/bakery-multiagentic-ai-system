from langchain_core.messages import HumanMessage

class InventoryManager:
    def __init__(self, llm, tools):
        # On garde une version du modèle avec outils et une version normale pour la synthèse
        self.model_with_tools = llm.bind_tools(tools)
        self.model_raw = llm 
        self.tools_map = {tool.name: tool for tool in tools}

    def run(self, state):
        print("--- AGENT GESTIONNAIRE : RECHERCHE ET SYNTHÈSE FINANCIÈRE ---")
        recipe = state.get('recipe_proposal', "")
        
        # 1. Appel pour déclencher la recherche
        search_prompt = f"Cherche les prix actuels du marché pour les ingrédients de cette recette : {recipe}"
        response = self.model_with_tools.invoke([HumanMessage(content=search_prompt)])
        
        search_context = ""
        
        # 2. Exécution réelle des outils si nécessaire
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool = self.tools_map[tool_call["name"]]
                # On récupère le texte brut des résultats de recherche
                search_context += str(tool.invoke(tool_call["args"]))
        else:
            search_context = "Pas de données web trouvées, utilise tes connaissances générales."

        # 3. DEUXIÈME APPEL : La synthèse propre
        # On utilise model_raw (sans outils) pour forcer la rédaction du rapport
        final_prompt = f"""Tu es Safiatou DIALLO, gestionnaire financière.
        Basé sur ces données de recherche : {search_context}
        
        Analyse la recette suivante : {recipe}
        
        Rédige ton rapport strictement au format suivant :
        - PRIX DES DEPENSES TOTAL : [Montant]€
        - PRIX DE VENTE CONSEILLÉ : [Montant]€
        - BÉNÉFICE ESTIMÉ : [Montant]€
        - CLIENTS CIBLES : [Description]
        - LIEUX DE VENTE : [Description]

        Sois directe et ne donne aucune explication technique."""

        final_response = self.model_raw.invoke([HumanMessage(content=final_prompt)])
        
        return {"financials": final_response.content}