# retrieval-augmented-generation
Retrieval augmented generation demo with Llama-2

This demo can run on CPU.


## 🔧 Getting Started

You will need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```bash
conda env create --name rag -f environment.yaml --force
```

Activate the environment
```bash
conda activate rag
```

Download the LLM artefact. The model used in this demo is downloaded from [TheBloke](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)
```bash
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin -P ./models/llama2-7b-chat-ggml
```

### 💻 App

We use Streamlit as the interface for the demo. To run the app,

```bash
streamlit run app.py
```
