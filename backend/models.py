from sqlalchemy import Text, Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy_utils import JSONType

import datetime
from .database import Base


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True)
    # Source url
    base_url = Column(String, unique=False, index=True)
    raw_txt = Column(Text)
    txt = Column(Text)

    authors = Column(String, unique=False)
    # When the website published it
    date_created = Column(String, unique=False)
    # When we added it to the DB
    date_added = Column(DateTime, default=datetime.datetime.utcnow)


class Results(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    url_id = Column(Integer,  ForeignKey("articles.id"))

    # Source url
    factuality_results = Column(JSONType)
    bias_results = Column(JSONType)


