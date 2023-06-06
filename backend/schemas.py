import datetime
from typing import Optional
from pydantic import BaseModel


class Article(BaseModel):
    #id: Optional[int]
    url: str
    base_url: str
    raw_txt: str
    txt: str

    authors: Optional[str]
    # When the website published it
    date_created: str
    # When we added it to the DB
    #date_added: Optional[datetime.datetime]

    class Config:
        orm_mode = True


class Results(BaseModel):
    #id: int
    url_id: int
    factuality_results: dict
    bias_results: dict

    class Config:
        orm_mode = True
