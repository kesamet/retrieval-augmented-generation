from loguru import logger

from langchain_core.messages import SystemMessage

from src import CFG

CHAT_FORMATS = {
    "llama2": "<s> [INST] <<SYS>>{system}<</SYS>>\n{user}\n[/INST]",
    "mistral": "<s> [INST] {system}\n{user}\n[/INST]",
    "zephyr": "<|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>",
    # RuntimeWarning: Detected duplicate leading "<bos>" in prompt,
    # this will likely reduce response quality, consider removing it...
    "gemma": "<start_of_turn>user\n{system}\n{user}<end_of_turn>\n<start_of_turn>model",
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    ),
    "phi3": "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>",
    "olmoe": "<|endoftext|><|user|>\n{system}\n{user}\n<|assistant|>",
    # RuntimeWarning: Detected duplicate leading "<BOS_TOKEN>" in prompt,
    # this will likely reduce response quality, consider removing it...
    "aya-expanse": (
        "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{user}<|END_OF_TURN_TOKEN|>"
        "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    ),
    "qwen": (
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
        "{user}<|im_end|>\n<|im_start|>assistant"
    ),
    "deepseek": "{system}\n<｜User｜>{user}<｜Assistant｜>",
    "gemini": "{system}\n{user}",
    "gpt": "{system}\n{user}",
}


class Prompts:
    def __init__(self, prompt_type: str):
        self.chat_format = CHAT_FORMATS.get(prompt_type)
        if self.chat_format is None:
            logger.warning("Chat prompt format not implemented. Using no format.")
            self.chat_format = "{system}\n{user}"

    @property
    def qa(self):
        system = (
            "You are a helpful, respectful and honest assistant for question-answering tasks. "
            "Only use the following pieces of retrieved context to answer the user's question. "
            "If the answer cannot be found from context, just say that you don't know, "
            "don't try to make up an answer. Answer in the same language as the question."
        )
        user = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.chat_format.format(system=system, user=user)

    @property
    def condense_question(self):
        system = ""
        user = (
            "Given the following chat history and a follow up question, "
            "rephrase the follow up question to be a standalone question, "
            "in its original language.\n\n"
            "Chat History:\n{chat_history}\n\nFollow Up Question: {question}\n"
            "Standalone question:"
        )
        return self.chat_format.format(system=system, user=user)

    @property
    def hyde(self):
        system = "You are a helpful, respectful and honest assistant for question-answering tasks."
        user = "Please answer the user's question about a document.\nQuestion: {question}"
        return self.chat_format.format(system=system, user=user)

    @property
    def multiple_queries(self):
        system = (
            "You are a helpful, respectful and honest assistant for question-answering tasks. "
            "Your users are asking questions about documents. "
            "Suggest up to three additional related questions to help them find the information "
            "they need for the provided question.\n"
            "Suggest only short questions without compound sentences.\n"
            "Suggest a variety of questions that cover different aspects of the topic.\n"
            "Make sure they are complete questions, and that they are related to "
            "the original question.\n"
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
CHAT_FORMAT = prompts.chat_format
QA_TEMPLATE = prompts.qa
CONDENSE_QUESTION_TEMPLATE = prompts.condense_question
HYDE_TEMPLATE = prompts.hyde
MULTI_QUERIES_TEMPLATE = prompts.multiple_queries
GENERATED_RESULT_TEMPLATE = prompts.generated_result

REACT_SYSTEM_MESSAGE = SystemMessage(
    content="""Y\ou are a helpful, respectful and honest assistant for question-answering tasks.
Respond to the user query only using the provided context.

<instructions>
- If you don't know the answer, clearly state that.
- If uncertain, ask the user for clarification.
- Respond in the same language as the user's query.
- If the context is unreadable or of poor quality, inform the user and provide the best possible answer.
</instructions>
"""
)
