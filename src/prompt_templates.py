from loguru import logger

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
            "If you don't know the answer, just say that you don't know, "
            "don't try to make up an answer."
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
QA_TEMPLATE = prompts.qa
CONDENSE_QUESTION_TEMPLATE = prompts.condense_question
HYDE_TEMPLATE = prompts.hyde
MULTI_QUERIES_TEMPLATE = prompts.multiple_queries
GENERATED_RESULT_TEMPLATE = prompts.generated_result


REACT_JSON_TEMPLATE = """Answer the following questions as best you can. \
You have access to the following tools:

{tools}

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) \
and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. \
Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.

Question: {input}
Thought: {agent_scratchpad}"""


REACT_TEMPLATE = """Answer the following questions as best you can. \
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
