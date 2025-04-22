# app.py (Révisé pour un meilleur RAG)

import os
import time
import warnings
import streamlit as st
import google.generativeai as genai
from google.generativeai import types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb

# --- Configuration ---
# Lire la clé API depuis les secrets ou .env
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Clé API Google non trouvée. Définir GOOGLE_API_KEY dans .env ou les secrets Streamlit.")
    st.stop()

PERSIST_DIRECTORY = "chroma_db_bible"
DB_COLLECTION_NAME = "bible_rag_collection"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
# --- CHANGER ICI LE MODELE LLM POUR UN PLUS DIRECT SI BESOIN ---
# gemini-1.5-flash est souvent bon pour RAG et rapide
LLM_MODEL_NAME = "models/gemini-1.5-flash-latest"
# LLM_MODEL_NAME = "models/gemini-1.5-pro-latest" # Garder si vous préférez

# Configurer le client GenAI
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Erreur configuration client GenAI: {e}")
    st.stop()

# --- Fonction d'embedding (identique à create_index.py) ---
is_retriable = lambda e: isinstance(e, (genai.APIError, types.GoogleAPIError)) and hasattr(e, 'reason') and e.reason in ['RATE_LIMIT_EXCEEDED', 'SERVICE_UNAVAILABLE', 'INTERNAL']
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, task_type="retrieval_document"):
        self._model_name = model_name
        self._task_type = task_type
        print(f"Embedding Function Initialized: Model='{self._model_name}', Task='{self._task_type}'") # Log initialisation
    @retry.Retry(predicate=is_retriable, initial=1.0, maximum=15.0, multiplier=2.0)
    def embed_content_with_retry(self, text: str):
         return genai.embed_content(model=self._model_name, content=text, task_type=self._task_type)
    def __call__(self, input_texts: Documents) -> Embeddings:
        embeddings = []
        # Ne pas logguer excessivement dans l'app Streamlit
        is_query_task = self._task_type == "retrieval_query"
        for i, doc in enumerate(input_texts):
            try:
                response = self.embed_content_with_retry(doc)
                embeddings.append(response['embedding'])
            except Exception as e:
                st.error(f"Erreur embedding (Task: {self._task_type}): {e}")
                raise e
        return embeddings
    def set_task_type(self, task_type: str):
        self._task_type = task_type
        print(f"Embedding task type set to: '{self._task_type}'") # Garder ce log utile

# --- Fonctions mises en cache ---

@st.cache_resource
def get_embedding_function_instance():
    print("Création (cache) instance fonction d'embedding...")
    return GeminiEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_chroma_collection(_persist_dir, _collection_name):
    print(f"Chargement (cache) collection ChromaDB depuis '{_persist_dir}'...")
    if not os.path.exists(_persist_dir):
        st.error(f"ERREUR: Dossier ChromaDB '{_persist_dir}' introuvable.")
        st.info("Exécutez 'python create_index.py' d'abord.")
        return None
    try:
        chroma_client = chromadb.PersistentClient(path=_persist_dir)
        # NE PAS passer embed_fn ici, Chroma utilisera celle associée à la collection
        collection = chroma_client.get_collection(name=_collection_name)
        print(f"Collection '{_collection_name}' chargée avec {collection.count()} docs.")
        return collection
    except Exception as e:
        st.error(f"Erreur chargement collection ChromaDB: {e}")
        return None

@st.cache_resource
def get_generative_model(_model_name):
    print(f"Initialisation (cache) modèle génératif '{_model_name}'...")
    try:
        # Ajouter des paramètres de génération par défaut ici si nécessaire
        return genai.GenerativeModel(_model_name)
    except Exception as e:
        st.error(f"Erreur initialisation modèle génératif: {e}")
        return None

# --- Fonction RAG Streamlit Révisée ---
def ask_bible_streamlit(question: str, collection, embed_fn, gen_model, n_results: int = 3): # Réduit n_results par défaut
    if not question:
        return None, None

    start_time = time.time()
    st.spinner("Recherche et génération...")

    # 1. Préparer l'embedding de la requête
    try:
        embed_fn.set_task_type("retrieval_query")
        # Embedder la question MANUELLEMENT
        query_embedding = embed_fn([question])[0] # Appel __call__ avec une liste d'un élément
    except Exception as e:
        st.error(f"Erreur lors de l'embedding de la question: {e}")
        return f"Erreur embedding question: {e}", []

    # 2. Interroger ChromaDB avec l'embedding
    try:
        results = collection.query(
            query_embeddings=[query_embedding], # Utiliser l'embedding pré-calculé
            n_results=n_results,
            include=['documents', 'distances'] # Inclure distances pour info
        )
        retrieved_documents = results['documents'][0]
        distances = results['distances'][0]
        if not retrieved_documents:
            return "Aucun passage pertinent trouvé.", []
        print(f"Distances des chunks récupérés: {distances}") # Log pour voir la pertinence
    except Exception as e:
        st.error(f"Erreur recherche ChromaDB: {e}")
        return f"Erreur recherche: {e}", []

    # 3. Construire le Prompt (PLUS DIRECTIF)
    # Inspiré du prompt du notebook
    prompt_template = """Instructions: Tu es un assistant spécialisé dans la Bible. Réponds à la QUESTION de l'utilisateur en te basant EXCLUSIVEMENT sur les PASSAGES bibliques fournis ci-dessous. Ne fais PAS référence à tes connaissances générales. Si les passages ne contiennent pas la réponse, indique clairement que l'information n'est pas présente dans les extraits fournis. Lorsque tu utilises une information d'un passage, essaie de mentionner 'selon le texte fourni' ou une formulation similaire. Reste factuel et neutre.

QUESTION:
{question}

PASSAGES PERTINENTS FOURNIS:
--- DEBUT DES PASSAGES ---
{passages}
--- FIN DES PASSAGES ---

RÉPONSE BASÉE UNIQUEMENT SUR LES PASSAGES FOURNIS:
"""
    passages_text = "\n\n---\n\n".join(retrieved_documents)
    final_prompt = prompt_template.format(question=question, passages=passages_text)

    # 4. Générer la réponse avec generate_content
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.1) # Garder basse température
        safety_settings = [ # Garder safety settings
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Utiliser generate_content au lieu de start_chat
        response = gen_model.generate_content(
            final_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        answer_text = response.text
    except Exception as e:
        st.error(f"Erreur génération Gemini: {e}")
        feedback = "N/A"
        try:
            if hasattr(response, 'prompt_feedback'): feedback = str(response.prompt_feedback)
        except: pass
        answer_text = f"Erreur lors de la génération: {e}\nFeedback: {feedback}"

    end_time = time.time()
    print(f"Question traitée en {end_time-start_time:.2f}s")
    return answer_text, retrieved_documents

# --- Interface Streamlit (Légèrement modifiée) ---

st.set_page_config(page_title="Bible Q&A (RAG)", layout="wide")
st.title("❓ Bible Q&A avec RAG et Gemini")
st.caption("Réponses basées sur le texte de la Bible (English PDF version)")

# Charger les ressources
embed_fn_instance = get_embedding_function_instance()
db_collection_instance = load_chroma_collection(PERSIST_DIRECTORY, DB_COLLECTION_NAME) # Ne passe plus embed_fn ici
generative_model_instance = get_generative_model(LLM_MODEL_NAME)

# Vérifier chargement
if db_collection_instance is None or generative_model_instance is None or embed_fn_instance is None:
    st.error("Échec chargement ressources. Vérifiez les logs et le dossier chroma_db_bible.")
    st.stop()

# Interface utilisateur
col1, col2 = st.columns([2,1]) # Colonne pour question/réponse, colonne pour options

with col1:
    user_question = st.text_area("Entrez votre question ici:", height=100, key="user_question")
    submit_button = st.button("🔍 Poser la question")

with col2:
    st.markdown("**Options de Récupération**")
    top_k = st.slider("Nombre de passages à récupérer (pertinence)", min_value=1, max_value=10, value=3, key="top_k")

# Logique d'exécution
if submit_button and user_question:
    answer, sources = ask_bible_streamlit(user_question, db_collection_instance, embed_fn_instance, generative_model_instance, n_results=top_k)
    with col1: # Afficher la réponse dans la colonne principale
        st.subheader("Réponse Générée:")
        if answer:
            st.markdown(answer)
        else:
            st.info("Aucune réponse générée ou erreur.")

        # Afficher les sources dans un expander
        if sources:
            with st.expander("Voir les passages sources utilisés"):
                for i, doc in enumerate(sources):
                    st.caption(f"Source {i+1} (Extrait)")
                    st.markdown(f"> {doc[:500]}...") # Afficher un extrait
elif submit_button and not user_question:
    st.warning("Veuillez entrer une question.")

st.sidebar.info(f"Index: {db_collection_instance.count()} passages.")
st.sidebar.info(f"LLM: {LLM_MODEL_NAME}")
st.sidebar.info(f"Embedding: {EMBEDDING_MODEL_NAME}")