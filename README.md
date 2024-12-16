![Book Translator](https://raw.githubusercontent.com/KazKozDev/researchify/main/banner.jpg)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![Flask](https://img.shields.io/badge/flask-2.0%2B-lightgrey)
![FAISS](https://img.shields.io/badge/FAISS-1.7%2B-orange)
![Ollama](https://img.shields.io/badge/ollama-1.0.0-brightgreen)
![Gemma](https://img.shields.io/badge/gemma-2.9B-purple)
![RAG](https://img.shields.io/badge/RAG-enabled-success)
![arXiv](https://img.shields.io/badge/arXiv-API-red)
![OpenAlex](https://img.shields.io/badge/OpenAlex-API-green)

# 🔬 Researchify - Scientific Research Assistant

A specialized chatbot designed to streamline scientific research by helping academics and researchers find, analyze, and understand scientific papers. The system serves as an intelligent research assistant that can search through academic databases, process scholarly articles, analyze citation patterns, and engage in natural conversations about research topics.

This project aims to solve common challenges in academic research:
- Time-consuming literature search and analysis
- Complex paper interpretation and summarization
- Citation impact assessment
- Research trend identification
- Cross-format document processing

Through natural language conversation, researchers can:
- Search for relevant papers using plain language queries
- Get paper summaries and key findings
- Analyze citation patterns and impact
- Process and extract information from various document formats
- Explore research trends and connections

![Book Translator](https://raw.githubusercontent.com/KazKozDev/researchify/main/demo.png)

## ✨ Features

- **Conversational Interface**: Natural language interaction for research queries and paper discussions
- **Smart Search**: 
  - Vector similarity search using FAISS and sentence transformers
  - Complex query parsing with field specifications
  - Hybrid retrieval combining semantic and metadata filters
- **Document Processing**:
  - Multi-format support (PDF, DOCX, XLSX, TXT)
  - Automatic text extraction and analysis
  - Metadata extraction and processing
- **Research Analysis**:
  - Paper content analysis and summarization
  - Citation impact evaluation
  - Geographic and temporal citation patterns
- **RAG System**:
  - Context-aware response generation
  - Document chunking with overlap
  - Efficient vector storage and retrieval

## 🚀 Getting Started

### Prerequisites

```bash
python 3.8+
ollama
PyMuPDF
flask
numpy
faiss-cpu
sentence-transformers
pandas
pypdf2
python-docx
chardet
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/researchify.git
cd researchify
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the Ollama server with Gemma-2B model:
```bash
ollama run gemma2
```

4. Run the application:
```bash
python app.py
```

## 🔧 Configuration

```bash
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=16777216  # 16MB
PAPER_ANALYSIS_CACHE_SIZE=100
MODEL_NAME=gemma2
VECTOR_STORE_PATH=vector_store
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
```

## 🏗️ Architecture

The system consists of several interconnected components:

```
├── Core Components
│   ├── Vector Store (FAISS + Sentence Transformers)
│   ├── RAG System
│   └── Query Processing
├── Document Handlers
│   ├── PDF Processor
│   ├── Word Processor
│   ├── Excel Processor
│   └── Text Processor
├── Analysis Modules
│   ├── Paper Analyzer
│   ├── Citation Analyzer
│   └── Impact Assessor
└── API Layer
    ├── Flask Server
    ├── Chat Interface
    └── Research Endpoints
```

## 📊 Technical Details

### Vector Search System

- FAISS similarity search implementation
- Sentence Transformer embeddings (all-MiniLM-L6-v2)
- Complex query support:
  - Field-specific search (title, abstract, category)
  - Boolean operations
  - Metadata filtering
- Thread-safe concurrent access
- Persistent index storage

### Document Processing

Specialized handlers for each format:
- **PDF Processing**:
  - PyPDF2-based text extraction
  - Encryption detection
  - Metadata parsing
- **Word Documents**:
  - Full text extraction
  - Core properties retrieval
- **Excel/CSV**:
  - Data preview generation
  - Statistical summaries
- **Text Files**:
  - Encoding detection
  - Format preservation

### RAG Implementation

- Chunk-based document processing
- Configurable overlap for context preservation
- Hybrid search combining:
  - Vector similarity
  - Category filtering
  - Date-based filtering
- Response generation with context integration

### Analysis Capabilities

Paper analysis includes:
- Content extraction and summarization
- Methodology identification
- Results analysis
- Limitations assessment
- Future work extraction

Citation analysis provides:
- Citation counts and trends
- Geographic distribution
- Research impact metrics
- Venue analysis
- Temporal patterns

## 🤝 Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- arXiv API for paper access
- OpenAlex for citation data
- FAISS for vector search capabilities
- Sentence Transformers for embeddings
- Ollama and Gemma-2B for LLM support

---
<div align="center">
From Bcn with ❤️ by KazKozDev
