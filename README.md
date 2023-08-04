# retrieval-augmented-generation
Retrieval augmented generation demo with Llama-2

The demos use quantized llama-2-7b-chat GGML model and run on CPU with acceptable inference time (~1 min).


## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda env create --name rag -f environment.yaml --force
```

Activate the environment.
```bash
conda activate rag
```

Download the LLM artefact. The model used in this demo is downloaded from [TheBloke](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main).
```bash
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin -P ./models/llama-2-7b-chat-ggml
```

You can also use another model of your choice. Ensure that it can be loaded by `langchain.llms.CTransformers` and update `config.yaml`.


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
