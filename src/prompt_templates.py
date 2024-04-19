from src import CFG

if CFG.PROMPT_TYPE == "llama2":
    _chat_format = """<s> [INST] <<SYS>>{system}<</SYS>>
{user}
[/INST]"""

elif CFG.PROMPT_TYPE == "mistral":
    _chat_format = """<s> [INST] {system}
{user}
[/INST]"""

elif CFG.PROMPT_TYPE == "zephyr":
    _chat_format = """<|system|>
{system}</s>
<|user|>
{user}</s>
<|assistant|>"""

elif CFG.PROMPT_TYPE == "gemma":
    _chat_format = """<start_of_turn>user
{system}
{user}<end_of_turn>
<start_of_turn>model"""

elif CFG.PROMPT_TYPE == "llama3":
    _chat_format = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

else:
    raise NotImplementedError("Chat prompt format not implemented")


class QA:
    system = (
        "You are a helpful, respectful and honest assistant. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    )
    user = "Context:\n{context}\nQuestion: {question}\nAnswer:"


class CondenseQuestion:
    system = ""
    user = (
        "Given the following conversation and a follow up question, "
        "rephrase the follow up question to be a standalone question, in its original language.\n\n"
        "Chat History:\n{chat_history}\n"
        "Follow Up Input: {question}\n"
        "Standalone question:"
    )


class Hyde:
    system = (
        "You are a helpful, respectful and honest assistant. "
        "Please answer the user's question about a document."
    )
    user = "Question: {question}"


class MultipleQueries:
    system = (
        "You are a helpful assistant. Your users are asking questions about documents. "
        "Suggest up to three additional related questions to help them find the information they need "
        "for the provided question.\n"
        "Suggest only short questions without compound sentences.\n"
        "Suggest a variety of questions that cover different aspects of the topic.\n"
        "Make sure they are complete questions, and that they are related to the original question.\n"
        "Output one question per line and without numbering."
    )
    user = "Question: {question}"


QA_TEMPLATE = _chat_format.format(system=QA.system, user=QA.user)
CONDENSE_QUESTION_TEMPLATE = _chat_format.format(
    system=CondenseQuestion.system, user=CondenseQuestion.user
)
HYDE_TEMPLATE = _chat_format.format(system=Hyde.system, user=Hyde.user)
MULTI_QUERIES_TEMPLATE = _chat_format.format(
    system=MultipleQueries.system, user=MultipleQueries.user
)
