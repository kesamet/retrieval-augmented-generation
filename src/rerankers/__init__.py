import os

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder

from src import CFG


def load_reranker():
    model = HuggingFaceCrossEncoder(
        model_name=os.path.join(CFG.MODELS_DIR, CFG.RERANKER_PATH),
        model_kwargs={"device": CFG.DEVICE},
    )
    compressor = CrossEncoderReranker(model=model, top_n=4)
    return compressor

    # if CFG.RERANKER_TYPE == "bge":
    #     from src.reranker.bge import BGERerank

    #     return BGERerank(top_n=CFG.RERANK_RETRIEVER_CONFIG.TOP_N)

    # if CFG.RERANKER_TYPE == "tart":
    #     from src.reranker.tart import TARTRerank

    #     return TARTRerank(top_n=CFG.RERANK_RETRIEVER_CONFIG.TOP_N)

    # raise NotImplementedError
