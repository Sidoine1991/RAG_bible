# Bible RAG Streamlit App

This repository contains a Streamlit application that uses Retrieval-Augmented Generation (RAG) to answer questions on the Bible. It employs ChromaDB for vector retrieval and Google Gemini (Generative AI) for response generation.

## 🚀 Production Deployment with Docker

### Prerequisites
- Docker 20.10+
- Google API Key with Generative AI access

### Setup
1. Copy your `.env` file alongside `app.py`, e.g.:  
   ```text
   GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
   ```
2. Build the Docker image:
   ```bash
   docker build -t bible-rag-app .
   ```
3. Run the container:
   ```bash
   docker run -d \
     -p 8501:8501 \
     -v "$PWD/chroma_db_bible:/app/chroma_db_bible" \
     --env GOOGLE_API_KEY \
     --name bible_rag_streamlit \
     bible-rag-app
   ```
4. Open your browser at [http://localhost:8501](http://localhost:8501).

### Notes
- The `chroma_db_bible` directory is mounted as a volume to persist your index between container restarts.
- To re-index the PDF, exec into the container or run locally:
  ```bash
  docker exec -it bible_rag_streamlit python create_index.py
  ```

## ⚙️ CI/CD (Optional)
Integrate with GitHub Actions or any CI to build and push the Docker image to a registry (Docker Hub, AWS ECR, etc.). Then deploy to Kubernetes, AWS ECS, or any Docker-compatible host.

---

*Happy coding!*
