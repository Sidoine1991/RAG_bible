# app.py pour Hugging Face Spaces : Bible RAG Streamlit
import os
import streamlit as st
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import PyPDF2
import google.generativeai as genai
from google.generativeai import types
from google.api_core import retry
from dotenv import load_dotenv

# --- Config ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Clé API Google non trouvée. Définir GOOGLE_API_KEY dans les secrets Hugging Face.")
    st.stop()

PERSIST_DIRECTORY = "chroma_db_bible"
DB_COLLECTION_NAME = "bible_rag_collection"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
LLM_MODEL_NAME = "models/gemini-1.5-flash-latest"
PDF_PATH = "bible_english.pdf"

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Erreur configuration client GenAI: {e}")
    st.stop()

is_retriable = lambda e: isinstance(e, (genai.APIError, types.GoogleAPIError)) and hasattr(e, 'reason') and e.reason in ['RATE_LIMIT_EXCEEDED', 'SERVICE_UNAVAILABLE', 'INTERNAL']
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, task_type="retrieval_document"):
        self._model_name = model_name
        self._task_type = task_type
    @retry.Retry(predicate=is_retriable, initial=1.0, maximum=15.0, multiplier=2.0)
    def embed_content_with_retry(self, text: str):
         return genai.embed_content(model=self._model_name, content=text, task_type=self._task_type)
    def __call__(self, input_texts: Documents) -> Embeddings:
        embeddings = []
        for doc in input_texts:
            try:
                response = self.embed_content_with_retry(doc)
                embeddings.append(response['embedding'])
            except Exception as e:
                st.error(f"Erreur embedding: {e}")
                raise e
        return embeddings
    def set_task_type(self, task_type: str):
        self._task_type = task_type

@st.cache_resource
def get_embedding_function_instance():
    return GeminiEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def ensure_chroma_index(pdf_path, persist_dir, collection_name, embed_fn):
    if not os.path.exists(persist_dir):
        st.info("Index ChromaDB absent, création en cours (1ère exécution, peut prendre 1-2 min)...")
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            full_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        chunk_size = 1000
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.create_collection(name=collection_name, embedding_function=embed_fn)
        collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
        st.success("Index ChromaDB créé avec succès !")
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(name=collection_name)

@st.cache_resource
def get_generative_model(_model_name):
    try:
        return genai.GenerativeModel(_model_name)
    except Exception as e:
        st.error(f"Erreur initialisation modèle génératif: {e}")
        return None

def ask_bible_streamlit(question: str, collection, embed_fn, gen_model, n_results: int = 3):
    if not question:
        return None, None
    embed_fn.set_task_type("retrieval_query")
    question_emb = embed_fn([question])[0]
    results = collection.query(query_embeddings=[question_emb], n_results=n_results)
    if not results["documents"] or len(results["documents"][0]) == 0:
        return None, None
    context = "\n\n".join(results["documents"][0])
    gen_prompt = f"Voici des extraits de la Bible :\n{context}\n\nQuestion : {question}\nRéponse :"
    response = gen_model.generate_content(gen_prompt)
    return response.text, context

# --- Interface Streamlit ---
st.set_page_config(page_title="Bible Q&A (RAG)", layout="wide")
st.title("❓ Bible Q&A avec RAG et Gemini")
st.markdown(":book: Posez une question sur la Bible (en anglais)")

with st.sidebar:
    st.info("Ce Space indexe le PDF à la première exécution (quelques minutes). Clé API Gemini requise dans les secrets.")
    n_results = st.slider("Nombre de passages à utiliser", 1, 5, 3)

question = st.text_input("Votre question sur la Bible (en anglais)")

embed_fn = get_embedding_function_instance()
collection = ensure_chroma_index(PDF_PATH, PERSIST_DIRECTORY, DB_COLLECTION_NAME, embed_fn)
gen_model = get_generative_model(LLM_MODEL_NAME)

if st.button("Envoyer"):
    if not question:
        st.warning("Entrez une question.")
        st.stop()
    with st.spinner("Recherche et génération de la réponse..."):
        answer, context = ask_bible_streamlit(question, collection, embed_fn, gen_model, n_results)
        if answer:
            st.success(answer)
            with st.expander("Passages utilisés"):
                st.write(context)
        else:
            st.error("Aucune réponse trouvée ou erreur lors de la génération.")
