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

Download the LLM model and save it in `./models`. The models used in this demo are downloaded from [TheBloke](https://huggingface.co/TheBloke):
- [Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)
- [OpenHermes-2-Mistral-7B-GGUF](https://huggingface.co/TheBloke/OpenHermes-2-Mistral-7B-GGUF/tree/main)

You can also use another model of your choice. Ensure that it can be loaded by `langchain.llms.ctransformers.CTransformers` and update `config.yaml`.

To use TART, we shall use [facebook/tart-full-flan-t5-xl"](https://huggingface.co/facebook/tart-full-flan-t5-xl). Download the artefacts and store them in `./models/tart-full-flan-t5-xl`.


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
