# Copyright 2026 Google LLC
# RAG Corpus Creation Script (Serverless Mode)

import vertexai
from vertexai.preview import rag
from vertexai.preview.rag.utils import resources as rr

PROJECT_ID = "qwiklabs-gcp-01-f41b114e9e04"
LOCATION = "us-central1"  # Serverless RAG is us-central1 only
GCS_PATH = "gs://greenhouse-agent-assets-f41b114e/rag/pg49513.txt"

print(f"Initializing Vertex AI for project {PROJECT_ID} in {LOCATION}...")
vertexai.init(project=PROJECT_ID, location=LOCATION)

print("1. Configuring RAG engine for serverless mode...")
cfg = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragEngineConfig"
rag.update_rag_engine_config(
    rag_engine_config=rag.RagEngineConfig(
        name=cfg,
        rag_managed_db_config=rag.RagManagedDbConfig(mode=rr.Serverless()),
    )
)

print("2. Creating RAG corpus 'herbal-corpus'...")
corpus = rag.create_corpus(
    display_name="herbal-corpus",
    embedding_model_config=rag.EmbeddingModelConfig(
        publisher_model="publishers/google/models/text-embedding-005"
    ),
)
print(f"RAG Corpus Created: {corpus.name}")

print(f"3. Importing and indexing {GCS_PATH}...")
resp = rag.import_files(
    corpus_name=corpus.name,
    paths=[GCS_PATH],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
    ),
)
print(f"Import complete! Files imported: {resp.imported_rag_files_count}")
print(f"CORPUS_NAME={corpus.name}")
