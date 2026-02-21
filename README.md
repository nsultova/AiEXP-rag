
##### Disclaimer

This is a proof-of-concept codebase built for educational purposes. It implements a minimal RAG (Retrieval-Augmented Generation) pipeline in the form of a personal librarian, and is intentionally kept simple to make the internals easy to follow and experiment with.
The repository includes a working baseline along with exercises designed to help you understand how the pieces fit together — and to break things intentionally so you can learn from it.


It is not production-ready. Error handling, security, and scalability are out of scope by design.

###### Techstack
* langchain
* FastAPI
* ChromaDB
* SSR w. jinja2

**Models**
* Embedding: https://huggingface.co/intfloat/e5-small-v2
* LLM: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

**!!!** If you change the embedding-model you NEED to reingest everything

##### Architecture

```
ai-librarian/
├── data/               # stores vector DB and uploaded files
├── templates/          # HTML files (Jinja2)
├── static/             # .scss and .css files
├── src/
│   ├── __init__.py
│   ├── app.py          # webserver
│   ├── config.py       # general config params
│   ├── ingest.py       # loaders & splitters
│   ├── vector.py       # database interface
│   ├── rag.py          # actual rag/llm
|   ├── metadata.py     # preprocessing documents
|   ├── reset_db.py     # cleanup script for ChromaDB
|   ├── utils.py        # shared text-processing utilities
│   └── test_suite.py   # various tests for different modules  
└── requirements.txt
```

**Dataflow**
`upload → ingest → embed → store → retrieve → generate`

 ![Dataflow](/img/basic_RAG_white_background.png)

#### For Archlinux

```
uv venv ai-librarian-env
source ./ai-librarian-env/bin/activate
uv pip install -e .
```

**Note:** Install ollama systemwide
`sudo pacman -S ollama`

* make sure ollama is enabled `sudo systemctl start ollama`
* grab the model `ollama pull llama3.2`
* start Ollama: `ollama run llama3.2` (or whichever model you configured)
* run the server: `python -m src.app`
* go to `http://localhost:8000`



#### RUN

Always from the project root (`ai-librarian/`), never from inside `src/`:

```
python -m src.app          # run the server
python -m src.test_models  # run all tests
python -m src.test_models rag  # run one test
```


#### Various tests

`python -m src.test_models` - run all tests
`python -m src.test_models rag`  - run one test


`curl http://localhost:8000/library` — should show your books with chapters populated
`curl "http://localhost:8000/debug/search?q=what+happens+in+the+first+chapter"` — inspect whether retrieved chunks are semantically relevant and whether metadata looks clean

**Test UI-DEMO**
Determine if it's your cache that must be cleaned or sth more serious

`python -m http.server 8080`

navigate to
`http://127.0.0.1:8080/templates/preview.html`

