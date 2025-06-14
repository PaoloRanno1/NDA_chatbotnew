# 📋 NDA Analyzer Chatbot

A sophisticated Streamlit web application for analyzing Non-Disclosure Agreements (NDAs) using AI-powered document analysis and conversational interface.

## 🌟 Features

- **📄 Document Analysis**: Upload and analyze PDF NDAs
- **💬 Interactive Chat**: Natural language conversation about NDA content
- **⚖️ Legal Compliance**: Specialized analysis for private equity firm requirements
- **🔍 Smart Q&A**: RAG-powered question answering with source citations
- **📊 Conversation Memory**: Maintains context across chat sessions
- **🎯 Intent Classification**: Automatically routes queries to appropriate analysis methods

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd nda-analyzer-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 🔧 Configuration

### OpenAI API Key

You can provide your OpenAI API key in several ways:

1. **Through the Streamlit interface** (recommended for development)
   - Enter your API key in the sidebar when the app starts

2. **Environment variable**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Streamlit secrets** (recommended for deployment)
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

### Model Selection

The application supports multiple OpenAI models:
- **gpt-4o** (default, recommended)
- **gpt-4**
- **gpt-3.5-turbo**

## 📋 Usage

### 1. Document Upload
- Use the sidebar to upload a PDF NDA document
- Click "Initialize Analyzer" to load the document

### 2. Quick Actions
- **📄 Summary**: Get a comprehensive document overview
- **⚖️ Legal Analysis**: Perform detailed compliance analysis
- **🗑️ Clear Chat**: Reset conversation history

### 3. Interactive Chat
- Ask questions about the NDA in natural language
- Get contextual responses with source citations
- Maintain conversation context across multiple queries

### 4. Example Questions
- "What are the main parties involved in this NDA?"
- "What are the confidentiality obligations?"
- "How long does this agreement last?"
- "Does this NDA comply with our firm's requirements?"

## 🏗️ Architecture

### Core Components

1. **EnhancedNDAAnalyzer**: Main analysis engine with:
   - Document processing and vectorization
   - Intent classification
   - Specialized prompts for different analysis types
   - Conversation memory management

2. **Streamlit Interface**: Web application providing:
   - File upload and management
   - Interactive chat interface
   - Real-time analysis feedback
   - Configuration controls

3. **RAG Pipeline**: Retrieval-Augmented Generation for:
   - Document chunking and embedding
   - Semantic search over document content
   - Source-cited responses

### Analysis Types

- **SUMMARY**: Basic document overview
- **LEGAL_ANALYSIS**: Detailed compliance checking
- **QUESTION**: Specific Q&A with document content
- **GENERAL**: Conversational responses

## 🛠️ Development

### Project Structure
```
nda-analyzer-chatbot/
├── app.py                 # Main Streamlit application
├── NDA_chatbot.py        # Core analyzer class
├── requirements.txt      # Python dependencies
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── .gitignore           # Git ignore rules
└── README.md           # This file
```

### Key Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM orchestration and document processing
- **openai**: OpenAI API integration
- **chromadb**: Vector database for document embeddings
- **pypdf**: PDF document processing

## 🚀 Deployment

### Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Configure secrets (add your OpenAI API key)
   - Deploy!

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t nda-analyzer .
docker run -p 8501:8501 nda-analyzer
```

### Environment Variables

For production deployment, set these environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key

## 🔒 Security Considerations

- **API Keys**: Never commit API keys to version control
- **Document Privacy**: Uploaded documents are processed locally and not stored permanently
- **Memory Management**: Chat history is cleared when the session ends

## 📝 Legal Compliance Analysis

The application includes specialized analysis for private equity firm requirements:

### Critical Exclusions (Flagged as HIGH PRIORITY)
- Broad liability clauses
- Overly broad non-solicitation terms
- Information retention restrictions
- Penalty clauses
- Non-compete clauses
- IP transfer provisions

### Mandatory Inclusions (Required)
- Investment disclaimers
- Broad disclosee definitions
- Electronic data retention exceptions
- Regulatory compliance retention rights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:

1. Check the [Issues](../../issues) page
2. Review the troubleshooting section below
3. Create a new issue with detailed information

### Common Issues

**"No module named 'NDA_chatbot'"**
- Ensure `NDA_chatbot.py` is in the same directory as `app.py`
- Check that all imports are correct

**"OpenAI API Error"**
- Verify your API key is correct and has sufficient credits
- Check your internet connection

**"Document loading failed"**
- Ensure the uploaded file is a valid PDF
- Check file size (must be under 200MB)

## 🔮 Future Enhancements

- [ ] Multi-document comparison
- [ ] Batch processing capabilities
- [ ] Advanced visualizations
- [ ] Export analysis reports
- [ ] Integration with document management systems
- [ ] Multi-language support
