# NLP-Project-1

To run locally (but first of all, please, change the /frontend/cfg.py contents to `ROOT = 'http://127.0.0.1:8000/'`) :
```bash
>>> uvicorn backend.main:app --reload
>>> streamlit run app.py --server.fileWatcherType none
```

To run with docker:
```bash
>>> docker-compose up
```
