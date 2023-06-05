from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/parse")
async def parse(url: str):
    return {"message": "Hello World"}

