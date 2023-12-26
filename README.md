# retrieval-augmented-generation
Retrieval augmented generation demos with Llama-2 or Mistral-7b

The demos use quantized models and run on CPU with acceptable inference time (~1 min).


## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda env create --name rag -f environment.yaml --force
```

Activate the environment.
```bash
conda activate rag
```

Download and save the models in `./models` and update `config.yaml`. The models used in this demo are:
- Embeddings
    - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    - [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- LLMs
    - [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
    - [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- Rerankers:
    - [facebook/tart-full-flan-t5-xl](https://huggingface.co/facebook/tart-full-flan-t5-xl): save in `models/tart-full-flan-t5-xl/`
    - [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base): save in `models/bge-reranker-base/`
- Propositionizer
    - [chentong00/propositionizer-wiki-flan-t5-large](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large) save in `models/propositionizer-wiki-flan-t5-large/`

Ensure that the LLM can be loaded by `langchain.llms.ctransformers.CTransformers`.


## 💻 App

We use Streamlit as the interface for the demos. There are two demos:
- Retrieval QA
```bash
streamlit run app.py
```

- Conversational retrieval QA
```bash
streamlit run app_conv.py
```


## 🔍 Usage

To get started, upload a PDF and click on `Build VectorDB`. Creating vector DB will take a while.

![screenshot](./assets/screenshot.png)
