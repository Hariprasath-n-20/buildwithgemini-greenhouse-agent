# Copyright 2026 Google LLC
# Standalone RAG Retrieval Test Script

import vertexai
from vertexai.preview import rag

PROJECT_ID = "qwiklabs-gcp-01-f41b114e9e04"
LOCATION = "us-central1"
CORPUS_NAME = (
    "projects/964783681132/locations/us-central1/ragCorpora/1394523791647834112"
)

vertexai.init(project=PROJECT_ID, location=LOCATION)

resp = rag.retrieval_query(
    text="What medicinal properties or uses does Rosemary have according to Culpeper?",
    rag_resources=[rag.RagResource(rag_corpus=CORPUS_NAME)],
    rag_retrieval_config=rag.RagRetrievalConfig(top_k=3),
)

print("Retrieval Query Results:")
for c in getattr(resp.contexts, "contexts", []):
    print(f"Distance/Score: {getattr(c, 'score', 0.0):.4f}")
    print(f"Passage: {getattr(c, 'text', '')[:300]}...\n---")
