import os

import numpy as np
from tqdm import tqdm
from umap import UMAP
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src import CFG


def project_embeddings(embeddings, umap_transform):
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


# embedding_function = SentenceTransformerEmbeddingFunction(model_name=os.path.join(CFG.MODELS_DIR, CFG.EMBEDDINGS_PATH))

# embeddings = chroma_db.get(include=['embeddings'])['embeddings']
# umap_transform = UMAP(random_state=0, transform_seed=0).fit(embeddings)
# projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# query = "What is the total revenue?"
# retriever = chroma_db.as_retriever()
# retrieved_documents = retriever.get_relevant_documents(query)

# query_embedding = embedding_function([query])[0]
# retrieved_embeddings = embedding_function([d.page_content for d in retrieved_documents])

# projected_query_embedding = project_embeddings([query_embedding], umap_transform)
# projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)
