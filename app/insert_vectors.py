from datetime import datetime
import pandas as pd
from timescale_vector.client import uuid_from_time
from database.vectorstore import VectorStore


def prepare_record(row):
    """Function to prepare records to be inserted to the vector database, after creating the embeddings."""

    # Creating question answer pair strings
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"

    # Creating embedding for the content
    content_embedding = vector_store.get_embedding(content)

    # Creating a series from the required data
    record_series = pd.Series(
        {
            'id': str(uuid_from_time(datetime.now())),
            'metadata': {
                'category': row['category'],
                'created_at': datetime.now().isoformat()
            },
            'content': content,
            'embedding': content_embedding
        }
    )

    return record_series
    
# Initializing vectorstore
vector_store = VectorStore()

# Loading data to pandas dataframe
df = pd.read_csv('../data/faq_dataset.csv', delimiter=';')

# Applying record preparation funciton to each record
df = df.apply(prepare_record, axis=1)

# Creating Table, Index and Inserting data to table
vector_store.create_tables()
vector_store.create_index()
vector_store.upsert(df)