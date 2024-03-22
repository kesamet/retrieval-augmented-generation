"""
Evaluate and track LLM experiments in llama-index with trulens-eval
LLM used here is gemini-pro
"""

import numpy as np
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.litellm import LiteLLM

from langchain_google_genai import ChatGoogleGenerativeAI
from trulens_eval.feedback.provider.langchain import Langchain
from trulens_eval import Tru, Feedback, TruLlama
from trulens_eval.feedback import Groundedness

# Setup RAG
index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="base_index"),
    embed_model="local:../models/bge-small-en-v1.5",
)

llm = LiteLLM(model="gemini/gemini-pro", temperature=0.1)
query_engine = index.as_query_engine(llm=llm)


# Evaluate with trulens-eval

# Define provider and database
_llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
provider = Langchain(chain=_llm)

database_url = "sqlite:///data/trulens.db"
tru = Tru(database_url=database_url, database_redact_keys=True)
# tru.reset_database()


# Using TruLlama
f_qa_relevance = Feedback(
    provider.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()

f_context_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

grounded = Groundedness(groundedness_provider=provider)
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(TruLlama.select_source_nodes().node.text)
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

app_id = "Chain2"

tru_recorder = TruLlama(
    query_engine,
    app_id=app_id,
    feedbacks=[
        f_qa_relevance,
        f_context_relevance,
        f_groundedness,
    ],
)

qns = ...
for qn in qns:
    with tru_recorder as recording:
        res = query_engine.query(qn)


# Results
# dashboard
tru.run_dashboard(port=8601)

# # dataframe
# records_df, feednack = tru.get_records_and_feednack(app_ids=[app_id])
# records_df.head()
