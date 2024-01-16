from src import CFG


class QA:
    system = (
        "You are a helpful, respectful and honest assistant. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    )
    user = "{question}\nContext = {context}"


class Hyde:
    system = (
        "You are a helpful, respectful and honest assistant. "
        "Please answer the user's question about a document."
    )
    user = "{question}"


class MultipleQueries:
    system = (
        "You are a helpful assistant. Your users are asking questions about documents. "
        "Suggest up to three additional related questions to help them find the information they need "
        "for the provided question. "
        "Suggest only short questions without compound sentences. "
        "Suggest a variety of questions that cover different aspects of the topic. "
        "Make sure they are complete questions, and that they are related to the original question. "
        "Output one question per line and without numbering."
    )
    user = "{question}"


_llama_format = """<s>[INST] <<SYS>>{system}<</SYS>>
Question: {user}
Answer:[/INST]"""

_mistral_format = """<s>[INST] {system}
Question: {user}
Answer:[/INST]"""

_zephyr_format = """<|system|>
{system}</s>
<|user|>
Question: {user}</s>
<|assistant|>"""

_openbuddy_format = """{system}
User: {user}
Assistant:"""


if CFG.PROMPT_TYPE == "llama":
    QA_TEMPLATE = _llama_format.format(system=QA.system, user=QA.user)
    HYDE_TEMPLATE = _llama_format.format(system=Hyde.system, user=Hyde.user)
    MULTI_QUERIES_TEMPLATE = _llama_format.format(
        system=MultipleQueries.system, user=MultipleQueries.user
    )

elif CFG.PROMPT_TYPE == "mistral":
    QA_TEMPLATE = _mistral_format.format(system=QA.system, user=QA.user)
    HYDE_TEMPLATE = _mistral_format.format(system=Hyde.system, user=Hyde.user)
    MULTI_QUERIES_TEMPLATE = _mistral_format.format(
        system=MultipleQueries.system, user=MultipleQueries.user
    )

elif CFG.PROMPT_TYPE == "zephyr":
    QA_TEMPLATE = _zephyr_format.format(system=QA.system, user=QA.user)
    HYDE_TEMPLATE = _zephyr_format.format(system=Hyde.system, user=Hyde.user)
    MULTI_QUERIES_TEMPLATE = _zephyr_format.format(
        system=MultipleQueries.system, user=MultipleQueries.user
    )

elif CFG.PROMPT_TYPE == "openbuddy":
    QA_TEMPLATE = _openbuddy_format.format(system=QA.system, user=QA.user)
    HYDE_TEMPLATE = _openbuddy_format.format(system=Hyde.system, user=Hyde.user)
    MULTI_QUERIES_TEMPLATE = _openbuddy_format.format(
        system=MultipleQueries.system, user=MultipleQueries.user
    )

else:
    raise NotImplementedError
