# Bakery Intelligence System : Une orchestration multi-agents pour la pâtisserie moderne.

Le domaine de la pâtisserie est parfait pour illustrer la collaboration multi-agents, car il mélange créativité, gestion des stocks et rigueur technique. Le système sera développé en s'inspirant de ce domaine et en se focalisant sur le travail bien accompli par nos quelques agents IA.

Ce projet déploie un écosystème d'IA collaboratif où chaque agent joue un rôle clé : un Chef (RAG) conçoit des recettes sur mesure, un Gestionnaire (Tavily) analyse les coûts et le marché en temps réel, et un expert Qualité assure la sécurité alimentaire. Propulsé par LangGraph pour la gestion d'état et ChromaDB pour la mémoire documentaire, le système transforme une simple idée en un rapport commercial complet et prêt à l'emploi. Une démonstration concrète de la puissance des agents autonomes au service de l'artisanat.

# Public cible

J'ai eu l'opportunité de réaliser ce projet afin de valider mon premier module dans mon parcours d'apprentissage avec Ready Tensor AI : https://www.readytensor.ai/

Ce projet s'adresse aux autres étudiants qui s'efforcent de devenir les meilleurs dans ce domaine. Il est également destiné aux recruteurs, car je suis actuellement à la recherche d'un stage pour l'obtention de mon diplôme.

# Qu'allons-nous développer ?

En réalisant ce projet, nous allons concevoir un assistant IA spécialisé pour la gestion d'une boulangerie-pâtisserie capable de :

- 📄 Charger les documents métiers (Fiches recettes, protocoles d’hygiène HACCP, catalogues fournisseurs, inventaires).

- 🔍 Explorer la base de connaissances pour retrouver instantanément une information précise (ex: temps de cuisson, allergènes, dosage spécifique).

- 💬 Répondre aux questions en s'appuyant exclusivement sur les données internes de l'entreprise.

- 🧠 Synthétiser plusieurs sources pour fournir des réponses complètes (ex: ajuster une recette en fonction du stock disponible ou des coûts matières).


# Outils & Frameworks

🛠️ Cœur de l'Orchestration

- LangChain : Le framework principal qui permet de lier les modèles de langage (LLM) à des données externes et des outils. C'est lui qui gère la logique de "chaîne" entre les composants.

- LangGraph : Une extension de LangChain utilisée pour créer des flux d'agents cycliques et gérer un "État" (State). C'est grâce à lui que les agents peuvent se transmettre des informations de manière structurée.

🧠 Intelligence & Modèles

- Groq : Le moteur d'inférence (LLM) qui alimente tes agents. Groq permet d'obtenir des réponses extrêmement rapides, ce qui est crucial pour un système multi-agents où plusieurs appels sont faits à la suite.

- ChromaDB : Notre base de données vectorielle. Elle stocke les documents techniques sur la boulangerie et permet à l'agent Chef de faire du RAG (Retrieval-Augmented Generation) pour trouver des informations fiables au lieu d'halluciner.

🌐 Outils & Monitoring

- Tavily Search : Un moteur de recherche optimisé pour les IA. Contrairement à Google, il renvoie des données structurées que l'agent Gestionnaire peut analyser directement pour trouver les prix réels du marché.

- LangSmith : La plateforme de monitoring. Elle permet de tracer chaque message passé entre les agents, de déboguer les erreurs et de voir exactement combien de temps chaque tâche a pris.


# Comment ça marche ?

Le projet suit un flux de travail rigoureux, de l'indexation des connaissances à l'exécution multi-agents :

1. Préparation des documents – Stockage de vos manuels de pâtisserie et fiches techniques dans le répertoire   data.

2. Indexation Vectorielle (RAG) – Découpage des textes en segments (chunks) et transformation en vecteurs stockés dans ChromaDB pour une recherche sémantique rapide.

3. Initialisation du StateGraph – Configuration de LangGraph pour définir l'état partagé (BakeryState) et l'ordre de passage entre les agents.

4. Orchestration du Chef – L'agent Chef interroge la base vectorielle pour extraire le contexte métier et générer une recette techniquement exacte.

5. Recherche de Marché (Action Tooling) – L'agent Gestionnaire utilise Tavily pour naviguer sur le web, récupérer les prix réels des ingrédients et calculer la viabilité économique.

6. Audit de Sécurité (Raisonnement) – L'agent Qualité analyse la sortie combinée du Chef et du Gestionnaire pour valider les allergènes et la conformité.

7. Synthèse Finale & Monitoring – Consolidation de toutes les analyses dans un rapport unique, avec un suivi complet de chaque étape via LangSmith pour garantir la transparence du processus.


## Structure du repo

```

bakery-multiagentic-ai-system
├── src/
│   ├── app.py                          # Main RAG application
│   ├── vectordb.py                     # Vector database wrapper
|    └── state.py
│   ├── agents/                               # List of agents
│       ├── chef.py
│       ├── inventorymanager.py
│       └── quality.py
├── data/                               # Sample publications
│   ├── recette_brownies_chocolat.txt
│   ├── recette_flan_patissier.txt
│   └── recette_gateau_vanille.txt
├── .gitignore
├── LICENSE
├── README.md             # This guide
└── requirements.txt      # All dependencies included

```

## Installation & Setup

1. **Clone le repository:**

   ```bash
   git clone https://github.com/ibrahimasorydiallo1/bakery-multiagentic-ai-system.git
   cd bakery-multiagentic-ai-system
   ```

2. **Installe les dépendances:**

   ```bash
   pip install -r requirements.txt

   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Prépare ta Groq API key:**

   Crée un fichier `.env` à la racine du projet et stocke  API key dans .env:

   ```
   GROQ_API_KEY=the-api-key-here
   ```

   Lien pour générer une API key [Groq](https://console.groq.com/).

4. **Prépare ta clé Tavily**

    Dans le fichier `.env` à la racine du projet, stocke

    ```
    TAVILY_API_KEY=votre_cle_api_tavily
    ```

    Lien pour générer une API key [Tavily](https://tavily.com/).


5. **Prépare ta clé LangSmith**

    Dans le fichier `.env` à la racine du projet, stocke

    ```
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=votre_cle_api_langsmith
    LANGCHAIN_PROJECT="Bakery-Agentic-RAG"
    ```
    Lien pour générer une API key [LangSmith](https://smith.langchain.com/).
---


# Resultats


# Conclusion et perspectives

Ce projet démontre comment l'intelligence artificielle peut sortir du cadre purement conversationnel pour devenir un véritable outil opérationnel. En orchestrant des agents spécialisés capables de manipuler des données métier (RAG) et d'interagir avec le monde réel (Tavily), nous passons d'une IA qui "discute" à une IA qui "exécute".

Le Bakery Intelligence System n'est qu'une première étape vers l'automatisation intelligente de l'artisanat. Plusieurs pistes d'évolution sont envisageables :

- Expansion du Marketing : Ajout d'un agent capable de générer des visuels produits (via DALL-E) et des publications pour les réseaux sociaux.

- Gestion des Stocks : Connexion à des API de fournisseurs pour passer commande automatiquement dès qu'une recette est validée.

- Optimisation Énergétique : Analyse des temps de cuisson pour réduire l'empreinte carbone et les coûts d'électricité.

En combinant LangGraph et la puissance des LLMs, ce projet ouvre la voie à une nouvelle génération d'assistants capables de comprendre la complexité d'un métier tout en garantissant une précision technique et financière.


# Licence et Droits d'Utilisation

Ce projet est publié sous la Licence MIT, offrant une flexibilité maximale pour l'utilisation, la modification et la distribution.

- Autorisations : La licence MIT accorde aux utilisateurs le droit d'utilisation commerciale sans restrictions ni redevances, la modification et la création d'œuvres dérivées, la distribution de versions originales ou modifiées, l'utilisation privée à des fins internes et l'utilisation de brevets pour les implémentations. Les organisations de toute taille peuvent adopter ce projet, l'intégrer dans des produits commerciaux, le modifier pour répondre à des besoins spécifiques et le déployer dans n'importe quel contexte commercial sans contraintes légales ni frais de licence.

- Limitations : Le logiciel est fourni « en l'état », sans aucune garantie d'aucune sorte. Aucune responsabilité n'est acceptée pour les dommages ou pertes découlant de son utilisation. Aucun droit de marque n'est accordé au-delà de ceux explicitement énoncés. Ces limitations standard protègent le projet tout en maintenant une large utilisabilité.

- Conditions : Les utilisateurs doivent inclure l'avis de droit d'auteur (copyright) original dans les distributions ainsi que le texte de la licence avec les copies du logiciel. Ces exigences minimales garantissent une attribution appropriée tout en permettant une flexibilité maximale de déploiement et de modification.

Le texte complet de la licence se trouve dans le fichier LICENSE à la racine du dépôt. Cette approche permissive maximise l'impact potentiel du projet sur l'accessibilité de l'intelligence d'affaires (Business Intelligence) pour divers contextes organisationnels, des startups aux grandes entreprises.


# Auteur

Ibrahima Sory DIALLO
Etudiant en Bachelor IA / DATA
Disponible sur linkedin https://www.linkedin.com/in/ibrahima-sory-diallo-isd/