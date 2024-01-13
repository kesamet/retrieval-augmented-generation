from src import CFG


def build_reranker():
    if CFG.RERANKER_TYPE == "bge":
        from src.reranker.bge import BGEReranker

        return BGEReranker(top_n=CFG.RERANK_RETRIEVER_CONFIG.TOP_N)
    if CFG.RERANKER_TYPE == "tart":
        from src.reranker.tart import TARTReranker

        return TARTReranker(top_n=CFG.RERANK_RETRIEVER_CONFIG.TOP_N)

    raise NotImplementedError
