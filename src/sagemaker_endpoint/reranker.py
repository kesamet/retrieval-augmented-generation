import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.utils import pre_init
from langchain.schema import Document
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pydantic import BaseModel, ConfigDict


class CrossEncoderContentHandler:
    """Content handler for CrossEncoder class."""

    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, text_pairs: List[Tuple[str, str]]) -> bytes:
        input_str = json.dumps({"text_pairs": text_pairs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[float]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["scores"]


class SagemakerEndpointCrossEncoder(BaseModel):
    """SageMaker Inference CrossEncoder endpoint."""
    client: Any = None  #: :meta private:

    endpoint_name: str = ""
    """The name of the endpoint from the deployed Sagemaker model.
    Must be unique within an AWS Region."""

    region_name: str = ""
    """The aws region where the Sagemaker model is deployed, eg. `us-west-2`."""

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    content_handler: CrossEncoderContentHandler = CrossEncoderContentHandler()

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    endpoint_kwargs: Optional[Dict] = None
    """Optional attributes passed to the invoke_endpoint
    function. See `boto3`_. docs for more info.
    .. _boto3: <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", protected_namespaces=()
    )

    @pre_init
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that AWS credentials to and python package exists in environment."""
        try:
            import boto3

            try:
                if values.get("credentials_profile_name"):
                    session = boto3.Session(profile_name=values["credentials_profile_name"])
                else:
                    # use default credentials
                    session = boto3.Session()

                values["client"] = session.client(
                    "sagemaker-runtime", region_name=values["region_name"]
                )

            except Exception as e:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        return values

    def score(self, text_pairs: List[List[str]]) -> List[float]:
        """Call out to SageMaker Inference CrossEncoder endpoint."""
        _endpoint_kwargs = self.endpoint_kwargs or {}

        body = self.content_handler.transform_input(text_pairs)
        content_type = self.content_handler.content_type
        accepts = self.content_handler.accepts

        try:
            response = self.client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=body,
                ContentType=content_type,
                Accept=accepts,
                **_endpoint_kwargs,
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        return self.content_handler.transform_output(response["Body"])


class SagemakerRerank(BaseDocumentCompressor):
    """Document compressor using SagemakerEndpointCrossEncoder."""

    model: SagemakerEndpointCrossEncoder
    top_n: int = 4

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Optional[Callbacks] = None) -> Sequence[Document]:
        """
        Compress documents.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results

    def rerank(self, query: str, docs: Sequence[str]) -> Sequence[Tuple[int, float]]:
        """
        Reranks a list of documents based on a given query using a pre-trained model.

        Args:
            query: The query string.
            docs: The list of documents to be reranked.

        Returns:
            A list of tuples containing the index of the document and its reranked score.
        """
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.score(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[: self.top_n]
