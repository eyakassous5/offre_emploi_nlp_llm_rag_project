Analyse intelligente des offres d'emploi — NLP, graphes et Graph-RAG

Projet académique (ESPRIT, 2026) : analyse d'offres d'emploi Data avec du NLP, un graphe hétérogène Entreprises – Postes – Compétences, un GNN simplifié, un système de recommandation de compétences et un mini Q&A de type Graph-RAG.

Les résultats chiffrés sont produits à l'exécution du notebook. Aucune valeur n'est préremplie.

Objectifs
Analyser les tendances du marché (compétences, pays, dates, salaires lorsque ces colonnes sont exploitables).
Classifier les catégories de postes (job_title_short) en comparant plusieurs représentations de texte.
Construire un graphe de connaissances et détecter des communautés.
Recommander des compétences et évaluer la recommandation.
Répondre à des questions en langage naturel à partir du graphe, avec une génération LLM locale optionnelle.
Contenu du projet
Étape	Description
1. Chargement et exploration	Dataset lukebarousse/data_jobs (Hugging Face), nettoyage et normalisation
2. Tendances du marché	Compétences les plus fréquentes, analyse géographique, temporelle et salaires (si disponibles)
3. NLP	Prétraitement des titres, TF-IDF, Word2Vec (compétences), SBERT (titres)
4. Classification des postes	Comparaison TF-IDF / Word2Vec / SBERT, découpage par groupes (GroupShuffleSplit) et vectorisation ajustée sur le train uniquement pour éviter les fuites de données
5. Clustering	K-Means sur embeddings SBERT (postes et compétences), PCA / t-SNE
6. Extraction d'entités	Taxonomie de compétences et enrichissement sémantique (SBERT)
7. Graphe hétérogène	Graphe NetworkX Entreprises–Postes–Compétences, centralités, communautés Louvain
8. GNN	Prédiction de liens Poste–Compétence avec un GraphSAGE simplifié (PyTorch natif)
9. Recommandation	Baseline fréquence + co-occurrence, évaluée en Precision@K / Recall@K par masquage de compétences
10. Mini Q&A (Graph-RAG simplifié)	Index inversé compétence → offres, retrieval dans le graphe, réponse traçable ; reformulation optionnelle par LLM local (Ollama)
11. Export	Modèles, métriques et données dans artifacts/
12. Application Streamlit	Génère app.py : onglets Marché, Recherche par compétences, Graphe
Limites (à lire)
Le dataset ne fournit pas toujours de description libre d'offre. Sans description, l'étape 6 est une extraction / normalisation d'entités à partir de champs structurés, et non un NER complet sur texte.
Le GNN est une expérience GraphSAGE simplifiée, pas un Graph Transformer Network (GTN).
Les résultats dépendent de l'échantillon (SAMPLE_SIZE, 20 000 par défaut), des seuils de construction du graphe et de la distribution du dataset.
Les recommandations et le Q&A ne sont pas des prédictions garanties du marché réel.
Salaires et dates ne sont analysés que si les colonnes contiennent assez de valeurs valides.
L'application Streamlit n'inclut pas l'onglet Q&A : le Q&A est utilisable dans le notebook.
Installation et exécution
Option 1 : Google Colab (recommandée)
Ouvrir Cahier_des_charges_Offres_Emploi_NLP_Graphes_Complet.ipynb dans Colab.
Exécuter toutes les cellules dans l'ordre. Le premier lancement télécharge le dataset et le modèle SBERT.
Option 2 : en local
bash
git clone https://github.com/eyakassous5/<nom-du-depot>.git
cd <nom-du-depot>
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install datasets sentence-transformers gensim spacy networkx scikit-learn \
            seaborn wordcloud pandas numpy matplotlib joblib streamlit torch jupyter
jupyter notebook

Ajuster la taille de l'échantillon dans la cellule de configuration (SAMPLE_SIZE, mettre None pour toutes les lignes, avec plus de RAM et de temps).

Lancer l'application Streamlit

Après exécution complète du notebook (qui crée artifacts/ et app.py) :

bash
streamlit run app.py
Génération LLM optionnelle (Ollama)

Le Q&A fonctionne sans LLM. Pour reformuler les faits récupérés avec un modèle local :

bash
ollama pull llama3.2
ollama serve

Puis décommenter l'appel answer_with_ollama(...) dans la section 10.1. Le prompt impose de répondre uniquement à partir des faits récupérés.

Résultats

À compléter avec les valeurs obtenues à l'exécution.

Tâche	Métrique	Résultat
Classification des postes	Accuracy / F1 macro (TF-IDF, Word2Vec, SBERT)	à compléter
Recommandation de compétences	Precision@K / Recall@K (baseline)	à compléter
GNN (liens Poste–Compétence)	à compléter	à compléter
Communautés Louvain	Nombre de communautés / modularité	à compléter

Captures d'écran suggérées (dossier docs/) : top compétences, graphe, application Streamlit.

Structure du dépôt
text
.
├── Cahier_des_charges_Offres_Emploi_NLP_Graphes_Complet.ipynb
├── README.md
├── app.py                # généré par le notebook (section 12)
├── artifacts/            # généré par le notebook (modèles, graphe, métriques)
└── docs/                 # captures d'écran (optionnel)
Technologies

Python, pandas, scikit-learn, Sentence-Transformers (SBERT), Gensim (Word2Vec), spaCy, NetworkX, PyTorch, Streamlit, Ollama (optionnel).

Données

Dataset : lukebarousse/data_jobs (Hugging Face). Consulter la page du dataset pour sa licence avant toute réutilisation.

Auteure

Eya Kassous — Étudiante ingénieure ERP & Business Intelligence, ESPRIT LinkedIn · GitHub
