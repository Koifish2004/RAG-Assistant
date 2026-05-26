from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    FusionQuery,
    Prefetch,
    Fusion,
    MultiVectorConfig,
    MultiVectorComparator,
    HnswConfigDiff
)
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
import hashlib
import atexit

COLLECTION_NAME = "study_agent"
DENSE_VECTOR_SIZE = 768
COLBERT_VECTOR_SIZE = 128





client = QdrantClient(path="vectorstore/qdrant_db")
atexit.register(client.close)


sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
colbert_model = LateInteractionTextEmbedding(
    model_name="answerdotai/answerai-colbert-small-v1"
)

def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(collection_name=COLLECTION_NAME, vectors_config={

            "dense": VectorParams(
                size = DENSE_VECTOR_SIZE,
                distance = Distance.COSINE
            ),
            "multi": VectorParams(
                size = COLBERT_VECTOR_SIZE,
                distance = Distance.COSINE,
                multivector_config = MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=HnswConfigDiff(m=0)
            )

        }, sparse_vectors_config={
            "sparse":SparseVectorParams(
                index=SparseIndexParams(on_disk = False)
            )

        }
        )
        print(f"Created Collection '{COLLECTION_NAME}'")


def store(embedded_chunks):
    ensure_collection()
    points = []


    for chunk in embedded_chunks:
        source    = chunk["source"].split("/")[-1]
        page      = chunk["page"]
        chunk_idx = chunk["chunk_index"]
        text      = chunk["text"]

        chunk_id = hashlib.md5(f"{source}-page{page}-chunk{chunk_idx}".encode()).hexdigest()
        sparse_result = list(sparse_model.embed([text]))[0]
        colbert_result = list(colbert_model.embed([text]))[0]
        point = PointStruct(
            id=chunk_id,
            vector={
                "dense": chunk["embedding"],
                "sparse": SparseVector(
                    indices=sparse_result.indices.tolist(),
                    values=sparse_result.values.tolist()
                ),
                "multi": colbert_result.tolist()
            },
            payload={
                "source":      source,
                "page":        page,
                "chunk_index": chunk_idx,
                "text":        text
            }
        )
        points.append(point)
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Stored {len(points)} chunks in Qdrant")


def query(query_vector, query_text, n_results=5):

    sparse_result = list(sparse_model.embed([query_text]))[0]
    colbert_result = list(colbert_model.embed([query_text]))[0]


    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=query_vector,
                using="dense",
                limit=n_results * 4
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_result.indices.tolist(),
                    values=sparse_result.values.tolist()
                ),
                using="sparse",
                limit=n_results * 2
            )
        ],
        query=colbert_result.tolist(),
        using="multi",
        limit=n_results
    )

    chunks = [point.payload for point in results.points]
    scores = [point.score for point in results.points]
    return chunks, scores