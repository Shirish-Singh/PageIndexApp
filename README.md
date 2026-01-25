# PageIndex Document QA

A free, open-source web application for intelligent document Q&A using tree-structured indexing. Upload PDFs, build semantic indexes, and ask questions - powered by free LLM APIs.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **PDF Document Indexing** - Automatically builds a hierarchical tree structure from your documents
- **Intelligent Q&A** - Ask natural language questions and get accurate answers with citations
- **Multiple LLM Providers** - Works with Groq (free), OpenAI, or local Ollama models
- **Save & Load Indexes** - Export document indexes as JSON to skip re-processing
- **No Vector Database** - Uses PageIndex's innovative tree-based approach instead of embeddings
- **Privacy Focused** - Your documents stay local; only text chunks are sent to the LLM

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pageindex-document-qa.git
cd pageindex-document-qa
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free API key

Get a free Groq API key at [console.groq.com](https://console.groq.com) (no credit card required).

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API key
```

### 5. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Deploy to Streamlit Cloud

You can deploy this app for free on Streamlit Cloud:

### 1. Fork/Push to GitHub

Push your code to a GitHub repository.

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository and `app.py`

### 3. Configure Secrets

In Streamlit Cloud dashboard, go to **Settings > Secrets** and add:

```toml
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Deploy!

Click "Deploy" and your app will be live in minutes.

## Usage

### Basic Workflow

1. **Upload PDF** - Drag and drop or click to upload your document
2. **Wait for Indexing** - The app builds a tree-structured index (may take a minute)
3. **Ask Questions** - Type your questions in the chat interface
4. **Download Index** - Save the index JSON to skip processing next time

### Loading Saved Indexes

1. Go to the **"Load Saved Index"** tab
2. Upload a previously downloaded `*_index.json` file
3. Start asking questions immediately - no processing needed!

### Changing LLM Provider

Use the sidebar to switch between providers:

| Provider | Cost | Setup |
|----------|------|-------|
| **Groq** | Free | Get key at [console.groq.com](https://console.groq.com) |
| **OpenAI** | Paid | Get key at [platform.openai.com](https://platform.openai.com) |
| **Ollama** | Free | Install [Ollama](https://ollama.ai) locally |

## Configuration

### Environment Variables

Create a `.env` file with your settings:

```env
# LLM Provider: groq, openai, or ollama
LLM_PROVIDER=groq

# Model to use (provider-specific)
LLM_MODEL=llama-3.1-8b-instant

# API Keys (add the one for your provider)
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key

# Ollama settings (if using local models)
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### Available Models

**Groq (Free)**
- `llama-3.1-8b-instant` - Fast, good for most documents
- `llama-3.3-70b-versatile` - More capable, slower rate limits
- `mixtral-8x7b-32768` - Good for longer contexts

**OpenAI (Paid)**
- `gpt-4o-mini` - Fast and affordable
- `gpt-4o` - Most capable

**Ollama (Local)**
- `llama3.2` - Good balance of speed/quality
- `mistral` - Fast and efficient

## Project Structure

```
PageIndexApp/
├── app.py              # Main Streamlit application
├── llm_provider.py     # LLM provider abstraction layer
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment configuration
├── PageIndex/          # PageIndex library (cloned from GitHub)
│   └── pageindex/
│       ├── page_index.py
│       └── utils.py
└── README.md
```

## How It Works

This app uses [PageIndex](https://github.com/VectifyAI/PageIndex), a novel approach to document understanding that:

1. **Builds a Tree Structure** - Analyzes document layout to create a hierarchical index
2. **Preserves Context** - Maintains relationships between sections, unlike flat chunking
3. **No Embeddings Needed** - Works without vector databases or embedding models
4. **LLM-Powered Reasoning** - Uses the LLM to navigate the tree and find relevant sections

## Disclaimer

**This software is provided "as is", without warranty of any kind.**

- This is a free, open-source project with **no support provided**
- Use at your own risk - not intended for production or commercial use
- Document content is sent to third-party LLM APIs (Groq/OpenAI) for processing
- Do not upload sensitive, confidential, or personal documents
- API rate limits and availability depend on your chosen provider
- The authors are not responsible for any data loss, costs, or damages

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PageIndex](https://github.com/VectifyAI/PageIndex) by VectifyAI - The core indexing technology
- [Streamlit](https://streamlit.io) - The web framework
- [Groq](https://groq.com) - Free LLM API access

---

**Made with minimal mass by the open source community**
