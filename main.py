from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
import pandas as pd

import torch

from config import PINECONE_API_KEY


model_id = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id)

nlp = pipeline(
    'ner',
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy='max',
    device='cpu'
)

retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")

api_key = PINECONE_API_KEY
pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name="medium-data",
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(    
        cloud="aws",
        region="us-east-1"
    )
)

idx = pc.Index('medium-data')

df = load_dataset(
    "fabiochiu/medium-articles",
    data_files="medium_articles.csv",
    split="train"
).to_pandas()

df = df.iloc[0:1000]




