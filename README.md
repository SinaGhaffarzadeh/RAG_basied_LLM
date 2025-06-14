# 🦙📚 LLM-Powered PDF Question Answering System

This project implements a **PDF-based Question Answering System** using **LlamaIndex**, **SentenceTransformers**, and **Hugging Face LLMs**. The pipeline extracts and indexes information from PDFs, embeds the text for semantic search, and utilizes a compact LLM (TinyLlama-1.1B-Chat) for natural language querying.

---

## 🔧 Features

- 🧠 **Local RAG (Retrieval-Augmented Generation)** pipeline
- 📄 Ingest and index multiple PDF files from a directory
- 🔍 Query using natural language and get intelligent answers
- ⚙️ Leverages **Hugging Face Transformers** and **SentenceTransformers**
- 🛠️ Error handling included via `try-except` blocks
- 🚀 GPU acceleration with CUDA support

---

## 📁 Project Structure

```
├── Data/                      # Directory containing PDF documents
├── main.py                   # Main application script
├── requirements.txt          # Project dependencies
└── README.md                 # Project description and instructions
```

---

## 🧠 Models Used

- 🔡 **LLM**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (lightweight causal language model)
- 📏 **Embedding**: `sentence-transformers/all-MiniLM-L6-v2`

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/SinaGhaffarzadeh/RAG_basied_LLM
cd llm-pdf-qa
```

### 2. Install dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Authenticate with Hugging Face

You must have a [Hugging Face account](https://huggingface.co/) and access token.

```python
from huggingface_hub import login
login(token="your_huggingface_token")
```

Alternatively, you can set it in your environment:

```bash
export HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

---

## 📦 Usage

Make sure you have placed your PDF files in the `Data/` directory. Then run the main script:

```bash
python main.py
```

This will:

- Load and parse the PDFs
- Embed the text content using SentenceTransformer
- Query the index with a hardcoded question (e.g., `"what is motion?"`)
- Print the answer from the LLM

---

## 🔍 Example Output

```bash
Cuda is available! True
The version of Cuda is: 12.1
{'name': 'your-username', 'email': 'your@email.com'}
Response: Motion is defined as...
```

---

## 🧪 Sample Query Flow

1. PDFs are loaded using `SimpleDirectoryReader`
2. Text is parsed into semantic nodes
3. Nodes are embedded and indexed
4. LLM answers queries using retrieval-augmented generation

---

## 🛡️ Error Handling

Basic `try-except` blocks have been added to handle common runtime issues like model download errors, missing CUDA devices, and file I/O errors.

---

## 💡 Future Improvements

- Add a Web UI using Gradio or Streamlit
- Accept user input for queries
- Expand to support DOCX and TXT files
- Store embeddings in persistent vector DB (e.g., FAISS, Chroma)

---
