<h2 align="center">
  <b>Retrieval augmented generation with quantized LLM</b><br>
</h2>

Retrieval augmented generation (RAG) demos with Mistral, Zephyr, Phi, Gemma, Llama, Aya-Expanse, Qwen

The demos use quantized models and run on CPU with acceptable inference time. They can run **offline** without Internet access, thus allowing deployment in an air-gapped environment.

The demos also allow user to
- apply propositionizer to document chunks
- perform reranking upon retrieval
- perform hypothetical document embedding (HyDE)


## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda env create --name rag python=3.11
conda activate rag
pip install -r requirements.txt
```

We shall use `unstructured` to process PDFs. Refer to [nstallation Instructions for Local Development](https://github.com/Unstructured-IO/unstructured?tab=readme-ov-file#installation-instructions-for-local-development).

You would also need to download `punkt_tab` and `averaged_perceptron_tagger_eng` from nltk.

```python
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

Note that we shall only use `strategy="fast"` in this demo. WIP for extraction of tables from PDFs.

Activate the environment.
```bash
conda activate rag
```

### 🧠 Use different LLMs

**Using a different LLM might lead to poor responses and even fail to output a response. It will require testing, prompt engineering and code refactoring.**

Download and save the models in `./models` and update `config.yaml`. The models used in this demo are:
- Embeddings
    - [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
    - [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
    - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Rerankers:
    - [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3): save in `models/bge-reranker-v2-m3/`
    - [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base): save in `models/bge-reranker-base/`
    - [facebook/tart-full-flan-t5-xl](https://huggingface.co/facebook/tart-full-flan-t5-xl): save in `models/tart-full-flan-t5-xl/`
- Propositionizer
    - [chentong00/propositionizer-wiki-flan-t5-large](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large) save in `models/propositionizer-wiki-flan-t5-large/`
- LLMs
    - [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
    - [bartowski/aya-expanse-8b-GGUF](https://huggingface.co/bartowski/aya-expanse-8b-GGUF)
    - [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)
    - [allenai/OLMoE-1B-7B-0924-Instruct-GGUF](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF)
    - [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
    - [microsoft/Phi-3-mini-4k-instruct-gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
    - [QuantFactory/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)
    - [lmstudio-ai/gemma-2b-it-GGUF](https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF)
    - [TheBloke/zephyr-7B-beta-GGUF](https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF)
    - [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
    - [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

The LLMs can be loaded directly in the app, or they can be first deployed with [**Ollama**](https://github.com/ollama/ollama) server.

You can also choose to use models from [**Groq**](https://wow.groq.com/). Set `GROQ_API_KEY` in `.env`.


### Add prompt format

Since each model type has its own prompt format, include the format in `./src/prompt_templates.py`. For example, the format used in `openbuddy` models is
```python
"""{system}
User: {user}
Assistant:"""
```

### 🤖 Tracing

We shall use [Phoenix](https://docs.arize.com/phoenix) for LLM tracing. Phoenix is an open-source observability library designed for experimentation, evaluation, and troubleshooting. Before running the app, start a phoenix server
```bash
python3 -m phoenix.server.main serve
```
The traces can be viewed at `http://localhost:6006`.


## 💻 App

We use Streamlit as the interface for the demos. There are three demos:

- Conversational Retrieval

```bash
streamlit run app_conv.py
```

- Retrieval QA

```bash
streamlit run app_qa.py
```

- Conversational Retrieval using ReAct

Create vectorstore first and update `config.yaml`
```bash
python -m vectorize --filepaths <your-filepath>
```
Run the app
```bash
streamlit run app_react.py
```


## 🔍 Usage

To get started, upload a PDF and click on `Build VectorDB`. Creating vector DB will take a while.

![screenshot](./assets/screenshot.png)
