from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
import pandas as pd
import torch
from tqdm.auto import tqdm

from config import PINECONE_API_KEY

def extract_entities(list_of_text):
    entities = []
    for doc in list_of_text:
        entities.append([item['word'] for item in nlp(doc)])
    return entities

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
df = df.iloc[0:30]
df = df.dropna()
df['text_extended'] = df['title']+"."+df['text'].str[:1000]

# batch_size = 64
batch_size = 10
for i in range(0,len(df),batch_size):
    i_end = min(i+batch_size,len(df))
    df_batch = df.iloc[i:i_end].copy()

    emb = retriever.encode(df_batch['text_extended'].tolist()).tolist()

    entities = extract_entities(df_batch['text_extended'].tolist())

    df_batch['named_entity'] = [list(set(entity)) for entity in entities]

    # df_batch = df_batch.drop('text_extended', axis=1)
    df_batch = df_batch.drop('text', axis=1)
    # axis = 1 -> for column wise dropping
    metadata = df_batch.to_dict(orient='records') # to convert to dictionary
    # metadata in pinecone needs to be in dictionary format
    ids = [f"{j}" for j in range(i,i_end)]

    vectors_to_upsert = list(zip(ids,emb,metadata))
    idx.upsert(vectors=vectors_to_upsert)

