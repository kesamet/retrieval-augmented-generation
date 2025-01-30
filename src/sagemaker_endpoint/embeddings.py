import json

from langchain_community.embeddings.sagemaker_endpoint import EmbeddingsContentHandler


class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: dict) -> bytes:
        body = {"inputs": inputs, "parameters": model_kwargs}
        input_str = json.dumps(body)
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        model_output = json.loads(output.read().decode("utf-8"))
        return [sublist[0][0] for sublist in model_output]
