import datetime

from pydantic import BaseModel

class Article(BaseModel):
    id: int
    url: str
    base_url: str
    raw_txt: str

    authors: str
    # When the website published it
    date_created: datetime.datetime
    # When we added it to the DB
    date_added: datetime.datetime


class Results(BaseModel):
    id: int
    url_id: int
    factuality_results: dict
    bias_results: dict
    