from src import CFG, logger

CHAT_FORMATS = {
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
    "phi3": """<|user|>
{system}
{user}<|end|>
<|assistant|>""",
    "gemini": "{system}\n{user}",
    "gpt": "{system}\n{user}",
}


class Prompts:
    def __init__(self, prompt_type: str):
        self.chat_format = CHAT_FORMATS.get(prompt_type)
        if self.chat_format is None:
            logger.warning("Chat prompt format not implemented. Using generic format.")
            self.chat_format = "{system}\n{user}"

    @property
    def qa(self):
        system = (
            "You are a helpful, respectful and honest assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer the user's question. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
        )
        user = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.chat_format.format(system=system, user=user)

    @property
    def condense_question(self):
        system = ""
        user = (
            "Given the following chat history and a follow up question, "
            "rephrase the follow up question to be a standalone question, in its original language.\n\n"
            "Chat History:\n{chat_history}\n\nFollow Up Question: {question}\nStandalone question:"
        )
        return self.chat_format.format(system=system, user=user)

    @property
    def hyde(self):
        system = "You are a helpful, respectful and honest assistant for question-answering tasks."
        user = (
            "Please answer the user's question about a document.\nQuestion: {question}"
        )
        return self.chat_format.format(system=system, user=user)

    @property
    def multiple_queries(self):
        system = (
            "You are a helpful, respectful and honest assistant for question-answering tasks. "
            "Your users are asking questions about documents. "
            "Suggest up to three additional related questions to help them find the information they need "
            "for the provided question.\n"
            "Suggest only short questions without compound sentences.\n"
            "Suggest a variety of questions that cover different aspects of the topic.\n"
            "Make sure they are complete questions, and that they are related to the original question.\n"
            "Output one question per line and without numbering."
        )
        user = "Question: {question}"
        return self.chat_format.format(system=system, user=user)

    @property
    def generated_result(self):
        system = "You are a helpful assistant."
        user = (
            "Provide an example answer to the given question, that might be found in a document.\n"
            "Question: {question}\nOutput:"
        )
        return self.chat_format.format(system=system, user=user)


prompts = Prompts(CFG.PROMPT_TYPE)
QA_TEMPLATE = prompts.qa
CONDENSE_QUESTION_TEMPLATE = prompts.condense_question
HYDE_TEMPLATE = prompts.hyde
MULTI_QUERIES_TEMPLATE = prompts.multiple_queries
GENERATED_RESULT_TEMPLATE = prompts.generated_result
