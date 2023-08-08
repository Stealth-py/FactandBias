# Fact&Bias

To make a conda environment

```
conda create --name ugrip
conda activate ugrip
conda install pip
pip install -r requirements.txt
```

To run locally (but first of all, please, change the /frontend/cfg.py contents to `ROOT = 'http://127.0.0.1:8000/'`) :
```bash
>>> uvicorn backend.main:app --reload
>>> streamlit run app.py --server.fileWatcherType none
```

To run with docker:
```bash
>>> docker-compose up
```
After that the frontend will be available via `http://localhost:8501/` and the backend is available via `http://localhost:8000/docs`.
Keep in mind that the backend requires some time to download the model weights
