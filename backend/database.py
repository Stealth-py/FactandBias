from sqlalchemy import create_engine
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

from sqlalchemy.orm import registry

mapper_registry = registry()

metadata_obj = sa.MetaData()



articles = sa.Table(
    'articles',
    metadata_obj,
    sa.Column('id', sa.Integer, primary_key=True),
    sa.Column('url', sa.String, unique=True),
    sa.Column('base_url', sa.String),
    sa.Column('raw_txt', sa.String),
    sa.Column('txt', sa.String),
    sa.Column('authors', sa.String),
    sa.Column('date_created', sa.String),
    sa.Column('date_added', sa.DateTime),
)

# # Create the profile table
metadata_obj.create_all(engine)
#mapper_registry.map_declaratively(Article)
