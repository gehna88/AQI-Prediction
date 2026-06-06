"""
mongo_store.py — MongoDB Atlas helper replacing Hopsworks Feature Store.
Drop-in replacement: store_df() and read_df() match fg.insert() / fg.read().
"""

import os
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING, UpdateOne

MONGO_URI = os.getenv("MONGODB_URI", "")
DB_NAME   = "aqi_predictor"
COL_NAME  = "features"


def _col():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
    col = client[DB_NAME][COL_NAME]
    col.create_index([("timestamp", ASCENDING)], unique=True)
    return col


def store_df(df: pd.DataFrame) -> int:
    """Upsert DataFrame rows — timestamp is the unique key."""
    col     = _col()
    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        if hasattr(rec.get("timestamp"), "isoformat"):
            rec["timestamp"] = rec["timestamp"].isoformat()
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else v)
               for k, v in rec.items()}
        records.append(rec)

    ops    = [UpdateOne({"timestamp": r["timestamp"]}, {"$set": r}, upsert=True)
              for r in records]
    result = col.bulk_write(ops, ordered=False)
    n      = result.upserted_count + result.modified_count
    print(f"  MongoDB: {n} rows upserted ({len(records)} sent)")
    return n


def read_df() -> pd.DataFrame:
    """Read all rows sorted by timestamp."""
    col  = _col()
    docs = list(col.find({}, {"_id": 0}).sort("timestamp", ASCENDING))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def read_latest(n_hours: int = 169) -> pd.DataFrame:
    """Read the last n_hours rows for lag feature computation."""
    col   = _col()
    total = col.count_documents({})
    skip  = max(0, total - n_hours)
    docs  = list(col.find({}, {"_id": 0})
                    .sort("timestamp", ASCENDING)
                    .skip(skip)
                    .limit(n_hours))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)