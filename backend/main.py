from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import insert
from sqlalchemy.orm import Session
from .database import SessionLocal
from typing import List
import uvicorn
from .scrape.scraping import extract_website
from .models import Article, Results
from .schemas import Article as ART, Results as RES
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

from memcache import async_memcache as aeromemcached

app = FastAPI()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/parse")
@cache(expire=60 * 60 * 24)
async def parse(url: str, db: Session = Depends(get_db)) -> List[RES]:
    try:
        result = extract_website(url)
    except Exception as e:
        print(e)
        return {}
    base_url, links = list(result.items())[0]
    articles = [
        Article(base_url=base_url,
                url=links[link]['processed_data'].get('source', ''),
                raw_txt=links[link].get('raw_html', ''),
                txt=links[link]['processed_data'].get('raw_text', ''),
                authors=links[link]['processed_data'].get('author', ''),
                date_created=links[link]['processed_data'].get('date', '')
                )
        for link in links
    ]
    results = []
    for a in articles:
        db.add(a)
        db.commit()
        r = Results(
            factuality_results={"Factuality": {"0": "Factual", "1": "Not Factual"},
             "Scores": {"0": 0.8124814628, "1": 0.1875185372}},
            bias_results={"Bias": {"0": "Left", "1": "Center", "2": "Right"},
             "Scores": {"0": 0.1792511051, "1": 0.0271034325, "2": 0.7936454624}},
            url_id=a.id
        )
        results.append(r)
        # db.add(r)
        # db.commit()
    # db.bulk_save_objects(
    #    articles,
    # )
    # db.commit()
    # db.refresh(articles)
    return results


@app.get("/db")
@cache(expire=60)
async def parse(db: Session = Depends(get_db)):
    return {"data": db.query(Article).all()}


@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-inmemorycache")
    print("Started")


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
