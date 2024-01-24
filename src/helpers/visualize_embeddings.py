import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from umap import UMAP
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src import CFG


def project_embeddings(embeddings, umap_transform):
    # Transform embeddings one at a time
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings


def visualize_embeddings(
    projected_dataset_embeddings,
    projected_query_embedding,
    projected_retrieved_embeddings,
):
    # Plot the projected query and retrieved documents in the embedding space
    fig = plt.figure()
    plt.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
    )
    plt.scatter(
        projected_query_embedding[:, 0],
        projected_query_embedding[:, 1],
        s=150,
        marker="X",
        color="r",
    )
    plt.scatter(
        projected_retrieved_embeddings[:, 0],
        projected_retrieved_embeddings[:, 1],
        s=100,
        facecolors="none",
        edgecolors="g",
    )
    plt.gca().set_aspect("equal", "datalim")
    plt.axis("off")
    return fig


embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=os.path.join(CFG.MODELS_DIR, CFG.EMBEDDINGS_PATH)
)

reducer = UMAP(random_state=0, transform_seed=0)

# # How to use
# embeddings = chroma_db.get(include=['embeddings'])['embeddings']
# umap_transform = reducer.fit(embeddings)
# projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# query = "..."
# retriever = chroma_db.as_retriever()
# retrieved_documents = retriever.get_relevant_documents(query)

# query_embeddings = embedding_function([query])
# projected_query_embedding = project_embeddings(query_embeddings, umap_transform)
# retrieved_embeddings = embedding_function([d.page_content for d in retrieved_documents])
# projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

# visualize_embeddings(
#     projected_dataset_embeddings,
#     projected_query_embedding,
#     projected_retrieved_embeddings,
# )
