# Quick Command Reference

## Setup (One-time)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit .env file with your credentials
# Add your MONGODB_URI and OPENAI_API_KEY
```

## Adding PDFs

```bash
# Create a folder for your documents
mkdir documents

# Copy your PDF files into the documents folder
# Then load them:
python scripts/load_docs.py documents/

# Or load from any other folder:
python scripts/load_docs.py /path/to/your/pdfs/
```

## Running the API

```bash
# Start the server
python main.py

# Server will run on http://localhost:8000
```

## Testing the API

### Method 1: Browser (Easiest)
```
Open: http://localhost:8000/docs
Click: POST /api/v1/chat → Try it out
Enter your question and click Execute
```

### Method 2: Command Line
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'
```

### Method 3: Python Script
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={"question": "What is this about?"}
)

print(response.json()["answer"])
```

## Useful Endpoints

```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status

# Performance metrics
curl http://localhost:8000/metrics

# API documentation
Open: http://localhost:8000/docs
```

## Common Tasks

### Add more documents
```bash
python scripts/load_docs.py new_documents/
```

### Check logs
```bash
# View application logs
cat logs/app.log

# View error logs
cat logs/error.log

# View performance logs
cat logs/performance.log
```

### Stop the server
```
Press Ctrl+C in the terminal where the server is running
```

## Troubleshooting

### Check if MongoDB is connected
```bash
curl http://localhost:8000/health
```

### View detailed status
```bash
curl http://localhost:8000/status
```

### Check environment variables
```bash
python -c "from src.core.config import validate_required_env_vars; validate_required_env_vars()"
```

## Example Workflow

```bash
# 1. Setup (first time only)
pip install -r requirements.txt

# 2. Add your credentials to .env file
# Edit .env and add MONGODB_URI and OPENAI_API_KEY

# 3. Add PDFs
mkdir documents
# Copy your PDFs to documents/
python scripts/load_docs.py documents/

# 4. Start server
python main.py

# 5. Test in browser
# Open http://localhost:8000/docs
# Try the /api/v1/chat endpoint

# 6. Ask questions!
```

That's it! 🚀
