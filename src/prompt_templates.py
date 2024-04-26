from src import CFG

CHAT_FORMAT = {
    "llama2": """<s> [INST] <<SYS>>{system}<</SYS>>
{user}
[/INST]""",
    "mistral": """<s> [INST] {system}
{user}
[/INST]""",
    "zephyr": """<|system|>
{system}</s>
<|user|>
{user}</s>
<|assistant|>""",
    "gemma": """<start_of_turn>user
{system}
{user}<end_of_turn>
<start_of_turn>model""",
    "llama3": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
}


class Prompts:
    def __init__(self, prompt_type: str):
        self.chat_format = CHAT_FORMAT.get(prompt_type)
        if self.chat_format is None:
            raise NotImplementedError("Chat prompt format not implemented")

    @property
    def qa(self):
        system = (
            "You are a helpful, respectful and honest assistant. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
        )
        user = "Context:\n{context}\nQuestion: {question}\nAnswer:"
        return self.chat_format.format(system=system, user=user)

    @property
    def condense_question(self):
        system = ""
        user = (
            "Given the following conversation and a follow up question, "
            "rephrase the follow up question to be a standalone question, in its original language.\n\n"
            "Chat History:\n{chat_history}\n"
            "Follow Up Input: {question}\n"
            "Standalone question:"
        )
        return self.chat_format.format(system=system, user=user)

    @property
    def hyde(self):
        system = (
            "You are a helpful, respectful and honest assistant. "
            "Please answer the user's question about a document."
        )
        user = "Question: {question}"
        return self.chat_format.format(system=system, user=user)

    @property
    def multiple_queries(self):
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
        return self.chat_format.format(system=system, user=user)


prompts = Prompts(CFG.PROMPT_TYPE)
QA_TEMPLATE = prompts.qa
CONDENSE_QUESTION_TEMPLATE = prompts.condense_question
HYDE_TEMPLATE = prompts.hyde
MULTI_QUERIES_TEMPLATE = prompts.multiple_queries
