from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
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


