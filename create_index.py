# create_index.py

import os
import time
import math
import warnings
import PyPDF2 # Import avec majuscules
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from google.generativeai import types
from google.api_core import retry
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb # Import ChromaDB

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
# ATTENTION: Mettre votre vraie clé API ici ou la lire depuis les variables d'environnement
# Pour un script local, ne pas utiliser Kaggle Secrets. Mettre la clé directement (Moins sécurisé)
# ou mieux, utiliser python-dotenv pour la charger depuis un fichier .env
# Exemple avec clé directe (À NE PAS FAIRE SI VOUS PARTAGEZ LE CODE) :
# GOOGLE_API_KEY = "VOTRE_VRAIE_CLE_API_ICI"
# Exemple avec variable d'environnement (Mieux)
from dotenv import load_dotenv
load_dotenv() # Créez un fichier .env avec GOOGLE_API_KEY="VOTRE_CLE"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Clé API Google non trouvée. Définissez GOOGLE_API_KEY dans l'environnement ou un fichier .env")

PDF_PATH = "bible_english.pdf" # Assurez-vous que ce fichier est dans le même dossier que le script
# Chemin où sauvegarder la base de données ChromaDB persistante
PERSIST_DIRECTORY = "chroma_db_bible"
DB_COLLECTION_NAME = "bible_rag_collection"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 100 # Pour l'ajout à ChromaDB

# Configurer le client GenAI
genai.configure(api_key=GOOGLE_API_KEY)

# --- Fonctions du Notebook (Adaptées) ---

# Fonction d'extraction PDF (Cellule 5)
def extract_text_from_pdf(pdf_path):
    # (Copiez le code de la fonction extract_text_from_pdf de la Cellule 5 ici)
    # ... (assurez-vous qu'elle retourne cleaned_text) ...
    print(f"Chargement du texte depuis: {pdf_path}...")
    extracted_text = ""
    start_time_extraction = time.time()
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"Le PDF contient {num_pages} pages.")
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + " "
            cleaned_text = ' '.join(extracted_text.split()) # Nettoyage final simplifié
            print(f"Extraction terminée. Longueur: {len(cleaned_text)}")
            return cleaned_text
    except Exception as e:
        print(f"Erreur extraction PDF: {e}")
        raise # Propage l'erreur

# Fonction d'embedding personnalisée (Cellule 7)
is_retriable = lambda e: isinstance(e, (genai.APIError, types.GoogleAPIError)) and hasattr(e, 'reason') and e.reason in ['RATE_LIMIT_EXCEEDED', 'SERVICE_UNAVAILABLE', 'INTERNAL']
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, task_type="retrieval_document"):
        self._model_name = model_name
        self._task_type = task_type
        self._api_call_count = 0
        self._docs_embedded_count = 0
        print(f"Embedding Function Initialized: Model='{self._model_name}', Task='{self._task_type}'")

    @retry.Retry(predicate=is_retriable, initial=1.0, maximum=15.0, multiplier=2.0)
    def embed_content_with_retry(self, text: str):
        self._api_call_count += 1
        resp = genai.embed_content(model=self._model_name, content=text, task_type=self._task_type)
        self._docs_embedded_count += 1
        # resp structure: {'embedding': [...], ...}
        return resp.get('embedding') or resp.embedding

    def __call__(self, texts):
        # Accept a single string or list of strings
        inputs = [texts] if isinstance(texts, str) else texts
        embeddings = []
        for t in inputs:
            emb = self.embed_content_with_retry(t)
            embeddings.append(emb)
        return embeddings

    def set_task_type(self, task_type: str):
        self._task_type = task_type

# --- Processus d'Indexation ---

if __name__ == "__main__":
    print("--- Démarrage du Script d'Indexation ---")

    # 1. Charger le texte du PDF
    print(f"\n[1/4] Chargement du texte depuis {PDF_PATH}...")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Fichier PDF non trouvé: {PDF_PATH}")
    bible_text = extract_text_from_pdf(PDF_PATH)
    if not bible_text:
        raise ValueError("Échec du chargement du texte PDF.")
    print("✅ Texte chargé.")

    # 2. Découper le texte en chunks (Cellule 6)
    print("\n[2/4] Découpage du texte en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""]
    )
    text_chunks = text_splitter.split_text(bible_text)
    total_chunks = len(text_chunks)
    print(f"✅ Texte découpé en {total_chunks} chunks.")
    del bible_text # Libérer mémoire

    # 3. Initialiser ChromaDB avec Persistance
    print(f"\n[3/4] Initialisation de ChromaDB (Persistant dans '{PERSIST_DIRECTORY}')...")
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"  Répertoire '{PERSIST_DIRECTORY}' existant. Il sera réutilisé/écrasé si nécessaire.")
    # Utiliser PersistentClient pour sauvegarder sur disque
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    print("✅ Client ChromaDB persistant initialisé.")

    # Instancier la fonction d'embedding
    embed_fn = GeminiEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME, task_type="retrieval_document")

    # Créer/Obtenir la collection
    print(f"  Accès/Création de la collection '{DB_COLLECTION_NAME}'...")
    db_collection = chroma_client.get_or_create_collection(
        name=DB_COLLECTION_NAME,
        embedding_function=embed_fn # IMPORTANT: Passer la fonction ici
    )
    print(f"✅ Collection prête. Compte initial: {db_collection.count()}")

    # 4. Indexer les chunks (Cellule 8)
    print(f"\n[4/4] Indexation des {total_chunks} chunks (peut être long)...")
    chunk_ids = [f"chunk_{i}" for i in range(total_chunks)]
    start_indexing_time = time.time()
    num_batches = math.ceil(total_chunks / BATCH_SIZE)

    # Vérifier si l'indexation est nécessaire
    if db_collection.count() >= total_chunks:
        print("Indexation déjà complète. Fin du script.")
    else:
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            print(f"  Traitement batch {batch_num}/{num_batches}...")
            batch_texts = text_chunks[i:min(i+BATCH_SIZE, total_chunks)]
            batch_ids = chunk_ids[i:min(i+BATCH_SIZE, total_chunks)]
            try:
                db_collection.add(documents=batch_texts, ids=batch_ids)
                print(f"  ✅ Batch {batch_num} ajouté. Collection: {db_collection.count()} docs.")
            except Exception as e:
                print(f"  ❌ ERREUR batch {batch_num}: {e}. Arrêt.")
                break # Arrêter en cas d'erreur
            time.sleep(0.5) # Petite pause

        end_indexing_time = time.time()
        print(f"\n✅ Indexation terminée en {end_indexing_time - start_indexing_time:.2f} secondes.")
        print(f"   Vérification finale: {db_collection.count()} documents dans la collection.")

    print("\n--- Script d'Indexation Terminé ---")