from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import insert
from sqlalchemy.orm import Session
from .database import SessionLocal
from typing import List, Any
import uvicorn
from .scrape.scraping import extract_website
from .models import Article, Results
from .schemas import Article as ART, Results as RES
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from time import time
from more_itertools import chunked
from .coder import ORJsonCoder
#import models.dummy_model_util as dmu
#from memcache import async_memcache as aeromemcached

from .inference_models.inference import ModelInference

# factmodel = ModelInference(model_path="models/sbert-factuality/checkpoint-497",
#                            tokenizer_path="sentence-transformers/all-mpnet-base-v2",
#                            quantize=False, use_gpu=True)
biasmodel = ModelInference(model_path="theArif/mbzuai-political-bias-bert",
                           tokenizer_path="theArif/mbzuai-political-bias-bert", quantize=False, use_gpu=True)
factmodel = biasmodel
app = FastAPI()

def request_key_builder(
    func,
    namespace: str = "",
    *,
    request = None,
    response = None,
    #*args,
    **kwargs,
):
    res = ":".join([
        namespace,
        request.method.lower(),
        request.url.path,
        repr(sorted(request.query_params.items()))
    ])
    print(res)
    return res



# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/parse")
@cache(expire=60 * 60 * 24, key_builder=request_key_builder)
async def parse(url: str, db: Session = Depends(get_db)) -> Any:
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

    print("Dumping Results")
    results = []
    txts = []
    cur = time()
    for a in articles[:1]:
        db.add(a)
        db.commit()
        # biasresults = dmu.get_inference_results(a.txt, task = "bias")
        # factresults = dmu.get_inference_results(a.txt, task = "fact")
        txts.append(a.txt)
    preds_factuality = []
    preds_bias = []    
    for chunk in chunked(txts, 64):
        biasresults = biasmodel.predict(chunk)
        factresults = factmodel.predict(chunk)
        preds_bias.extend(biasresults)
        preds_factuality.extend(factresults)
    for factresults, biasresults in zip(preds_factuality, preds_bias):
        r = Results(
            factuality_results={"Factuality": {"0": "Less Factual", "1": "Mixed Factuality", "2": "Highly Factual"},
             "Scores": {"0": factresults[0], "1": factresults[1], "2": factresults[2]}},
            bias_results={"Bias": {"0": "Left", "1": "Center", "2": "Right"},
             "Scores": {"0": biasresults[0], "1": biasresults[1], "2": biasresults[2]}},
            url_id=a.id
        )

        results.append(r)
    end = time()

    print({"Factuality": {"0": "Less Factual", "1": "Mixed Factuality", "2": "Highly Factual"},
             "Scores": {"0": factresults[0], "1": factresults[1], "2": factresults[2]}})

    print("Time to run: ", end - cur)
    print(results)
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
