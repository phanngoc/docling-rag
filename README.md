# Document Conversion and RAG System

This project provides a document processing and Retrieval Augmented Generation (RAG) system for extracting, chunking, and querying information from various document formats.

## Features

- Convert PDF, HTML, Markdown, and image documents
- Extract text, tables, and images from documents
- Process and chunk documents using various strategies (Character, Recursive, Semantic)
- Store document chunks in a vector database (Qdrant)
- Query document content using natural language with RAG
- Visualize and export document elements
- Web interface for document upload and interactive Q&A

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key (for LLM capabilities)

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd doc-convert
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Qdrant server (required for vector storage):
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

### Option 2: Docker Deployment

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd doc-convert
   ```

2. Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Run the application stack:
   ```bash
   docker-compose up
   ```

## Usage

### Document Conversion

Convert PDF documents to Markdown using the `test1.py` script:

```bash
python test1.py
```

This will:
1. Download a sample PDF from arXiv
2. Convert it to Markdown format
3. Save the output to `./output/` directory

### Figure Export

Extract figures and tables from PDF documents using the `figure-export.ipynb` notebook:

1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook figure-export.ipynb
   ```
2. Run the cells to extract and save images from the PDF

### RAG Web Interface

Use the Streamlit web application to process documents and ask questions:

1. Start the application:
   ```bash
   streamlit run rag_streamlit_app.py
   ```

2. Access the web interface at http://localhost:8501

3. Enter your OpenAI API key (if not set in .env)

4. Input a document URL to process

5. Ask questions about the document content

## Chunking Strategies

The system supports multiple chunking strategies:

- **Character Chunking**: Divides text based on a fixed number of characters
- **Recursive Character Chunking**: Divides text recursively until a condition is met
- **Document Specific Chunking**: Respects document structure (paragraphs, sections)
- **Semantic Chunking**: Groups text by semantic relationships
- **Token-based Chunking**: Divides text based on token count

## Development

### Project Structure

- `rag_streamlit_app.py`: Main Streamlit web application
- `test1.py`: PDF to Markdown converter script
- `test2.py` and `test2.ipynb`: Document chunking and vector search examples
- `figure-export.ipynb`: Extract figures and tables from PDFs
- `requirements.txt`: Python dependencies
- `Dockerfile` and `docker-compose.yml`: Containerization configuration

### Adding New Features

1. **New Document Formats**:
   - Extend the `DocumentConverter` class with additional format handlers
   - Add the format to `allowed_formats` in converter initialization

2. **Custom Chunking Strategies**:
   - Create a new chunker class extending base chunkers
   - Implement the `chunk` method

3. **Improving RAG**:
   - Modify the LangChain components in `setup_langchain_rag` function
   - Adjust parameters in the retriever or document chain

## Troubleshooting

- **Memory Issues**: For large documents, increase Docker memory limits or process documents in smaller batches
- **PDF Conversion Issues**: Check PDF permissions or try alternative URLs
- **Qdrant Connectivity**: Ensure Qdrant server is running and accessible

## License

[Specify License]