# End-to-end Chatbot Generative AI

An end-to-end chatbot project based on generative AI, utilizing the latest LangChain framework and vector database technology.

## Features

- Conversation management based on LangChain framework
- Text embedding using Sentence Transformers
- Chroma vector database for efficient similarity search
- Flask Web interface
- PDF document processing capabilities

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/End-to-end-Chatbot-Generative-AI.git
cd End-to-end-Chatbot-Generative-AI
```

2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Environment setup
- Copy `.env.example` to `.env`
- Fill in the OpenAI API key

## Usage

1. Start the application
```bash
python app.py
```

2. Access the Web Interface
- Open your browser and visit `http://localhost:8080`

## Project Structure

```
.
├── app.py              # Main Flask application file
├── src/               # Source code directory
├── static/            # Static resources
├── templates/         # HTML templates
├── data/              # Data files directory
├── db/                # Database files
└── requirements.txt   # Project dependencies
```

## Requirements

- Python 3.8+
- Supported OS: Linux, macOS, Windows
