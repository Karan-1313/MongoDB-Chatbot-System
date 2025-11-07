# MongoDB Chatbot System

An intelligent chatbot system that uses MongoDB Atlas Vector Search and OpenAI to answer questions based on your documents (PDFs, text files).

## 🚀 Features

- **📄 Document Processing**: Automatically processes PDFs and text files
- **🔍 Vector Search**: Uses MongoDB Atlas Vector Search for semantic document retrieval
- **🤖 AI-Powered**: Leverages OpenAI GPT-4 for intelligent responses
- **📊 Source Attribution**: Cites sources for every answer
- **⚡ Fast API**: RESTful API built with FastAPI
- **📈 Monitoring**: Built-in logging, metrics, and performance monitoring
- **🔄 Retry Logic**: Automatic retry with exponential backoff for API calls
- **🛡️ Error Handling**: Comprehensive error handling and validation

## 📋 Prerequisites

- Python 3.8+
- MongoDB Atlas account (free tier works!)
- OpenAI API account

## 🔧 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd mongodb-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your credentials:
   # - MONGODB_URI (from MongoDB Atlas)
   # - OPENAI_API_KEY (from OpenAI Platform)
   ```

4. **Set up MongoDB Atlas Vector Search Index:**
   
   Create a search index in MongoDB Atlas with this configuration:
   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 3072,
         "similarity": "cosine"
       },
       {
         "type": "filter",
         "path": "metadata.source"
       }
     ]
   }
   ```
   Name it `vector_index` (or update `VECTOR_INDEX_NAME` in `.env`)

## 🎯 Quick Start

### 1. Load Documents

```bash
# Create a documents folder
mkdir documents

# Add your PDF files to the documents folder

# Load them into MongoDB
python scripts/load_docs.py documents/
```

### 2. Start the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### 3. Test the API

**Option 1: Interactive API Docs**
- Open http://localhost:8000/docs
- Try the POST `/api/v1/chat` endpoint

**Option 2: Command Line**
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

**Option 3: Python**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={"question": "What is this document about?"}
)

print(response.json()["answer"])
```

## 📁 Project Structure

```
mongodb-chatbot/
├── src/
│   ├── api/              # FastAPI routes and models
│   ├── core/             # Configuration, logging, monitoring
│   ├── database/         # MongoDB connection and operations
│   ├── graph/            # LangGraph workflow orchestration
│   ├── nodes/            # Workflow nodes (retrieval, reasoning)
│   └── tools/            # Utilities (embeddings, vector store, text processing)
├── scripts/              # Document loading scripts
├── logs/                 # Application logs (auto-generated)
├── .env                  # Your configuration (DO NOT COMMIT!)
├── .env.example          # Configuration template
├── requirements.txt      # Python dependencies
├── main.py              # Application entry point
└── COMMANDS.md          # Quick command reference
```

## 🔑 Environment Variables

### Required
- `MONGODB_URI` - MongoDB Atlas connection string
- `OPENAI_API_KEY` - OpenAI API key

### Optional
- `MONGODB_DATABASE` - Database name (default: `chatbot_db`)
- `MONGODB_COLLECTION` - Collection name (default: `documents`)
- `EMBEDDING_MODEL` - OpenAI embedding model (default: `text-embedding-3-large`)
- `CHAT_MODEL` - OpenAI chat model (default: `gpt-4`)
- `API_HOST` - Server host (default: `0.0.0.0`)
- `API_PORT` - Server port (default: `8000`)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `VECTOR_INDEX_NAME` - MongoDB vector index name (default: `vector_index`)
- `SIMILARITY_THRESHOLD` - Similarity threshold (default: `0.7`)
- `MAX_RETRIEVED_DOCS` - Max documents to retrieve (default: `5`)

See `.env.example` for complete configuration options.

## 📡 API Endpoints

### Chat
**POST** `/api/v1/chat`
```json
{
  "question": "What is this about?",
  "max_tokens": 500,
  "session_id": "optional-session-id"
}
```

### Health Check
**GET** `/health` - Check system health

### Status
**GET** `/status` - Detailed system status

### Metrics
**GET** `/metrics` - Performance metrics

### Documentation
**GET** `/docs` - Interactive API documentation (Swagger UI)

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
isort src/ tests/
```

### Linting
```bash
flake8 src/ tests/
```

## 📊 Monitoring

The system includes comprehensive monitoring:
- **Logs**: Check `logs/` folder for detailed logs
  - `app.log` - All application logs
  - `error.log` - Error logs only
  - `performance.log` - Performance metrics
- **Metrics**: Visit `/metrics` endpoint for real-time metrics
- **Health**: Visit `/health` for system health status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [OpenAI](https://openai.com/)
- Uses [MongoDB Atlas](https://www.mongodb.com/atlas)
- Orchestrated with [LangGraph](https://github.com/langchain-ai/langgraph)

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Note**: Never commit your `.env` file or any files containing API keys or credentials!