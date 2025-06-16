import streamlit as st
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.utils import convert_files_to_docs
from pathlib import Path
import json
import os
from haystack.nodes import CohereReranker

reranker = CohereReranker(api_key="TA_CLE_COHERE", top_k=3)

# Puis dans ton pipeline Haystack :
# pipeline.add_node(component=reranker, name="Reranker", inputs=["Retriever"])

UPLOAD_DIR = Path("uploaded_documents")
LOG_PATH = Path("index_log.json")

def get_file_mod_times():
    return {
        f.name: os.path.getmtime(f)
        for f in UPLOAD_DIR.iterdir()
        if f.is_file()
    }

def load_previous_log():
    if LOG_PATH.exists():
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    return {}

def save_log(log_data):
    with open(LOG_PATH, "w") as f:
        json.dump(log_data, f)

def get_files_to_index(current_log, previous_log):
    changed = []
    for filename, mod_time in current_log.items():
        if filename not in previous_log or mod_time != previous_log[filename]:
            changed.append(filename)
    return changed

# ----------------------
# CONFIGURATION
# ----------------------

UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Recherche IA dans vos documents")
st.title("📄 Recherche intelligente avec Haystack")


# ----------------------
# UPLOAD & INDEXATION
# ----------------------

st.header("1. 📤 Importer vos documents")

uploaded_files = st.file_uploader("Déposez vos fichiers ici (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        file_path = UPLOAD_DIR / file.name
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.success("✅ Fichiers enregistrés. Prêts pour l'indexation.")

# ----------------------
# INDEXATION (uniquement si des fichiers sont dans le dossier)
# ----------------------

current_log = get_file_mod_times()
previous_log = load_previous_log()
files_to_index = get_files_to_index(current_log, previous_log)

if files_to_index:
    st.info("📥 Nouveaux fichiers ou fichiers modifiés détectés.")
    docs = convert_files_to_docs(dir_path=UPLOAD_DIR, file_paths=files_to_index)
    embedded_docs = embedder.run(docs)["documents"]
    writer.run(documents=embedded_docs)
    save_log(current_log)
    st.success("✅ Indexation mise à jour.")
else:
    st.success("✅ Aucun nouveau fichier détecté. Indexation sautée.")

if any(UPLOAD_DIR.iterdir()):
    st.header("2. ⚙️ Indexation des documents")

    with st.spinner("Indexation en cours..."):

        # 1. Initialiser le DocumentStore (en mémoire, changeable)
        document_store = InMemoryDocumentStore(embedding_dim=1536, similarity="cosine")

        # 2. Initialiser l’embedder OpenAI
        embedder = OpenAIDocumentEmbedder(
            api_key="sk-...",  # 🔐 Remplace par ta clé API OpenAI
            model="text-embedding-3-small"
        )

        # 3. Convertir et indexer les documents
        docs = convert_files_to_docs(dir_path=UPLOAD_DIR)
        embedded_docs = embedder.run(docs)["documents"]

        # 4. Écrire dans la base
        writer = DocumentWriter(document_store=document_store)
        writer.run(documents=embedded_docs)

        # 5. Créer le retriever
        retriever = InMemoryEmbeddingRetriever(
            document_store=document_store,
            embedding_model=embedder
        )

    st.success("✅ Documents indexés avec succès !")

    # ----------------------
    # INTERFACE DE RECHERCHE
    # ----------------------

    st.header("3. 🔍 Poser une question")

    query = st.text_input("Entrez votre question ici...")

    if query:
        with st.spinner("Recherche en cours..."):
            results = retriever.run(query=query, top_k=5)["documents"]

        if results:
            st.markdown("### 📌 Résultats les plus pertinents :")
            for doc in results:
                st.markdown(f"**📄 {doc.meta.get('name', 'Document')}**")
                st.write(doc.content[:1000])  # Tronquer si nécessaire
                st.caption("—" * 20)
        else:
            st.warning("❌ Aucun résultat trouvé.")