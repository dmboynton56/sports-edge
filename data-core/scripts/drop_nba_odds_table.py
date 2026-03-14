#!/usr/bin/env python3
"""Drop raw_nba_odds table to recreate with correct schema."""

import os
import sys
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

project_id = "learned-pier-478122-p7"
table_id = f"{project_id}.sports_edge_raw.raw_nba_odds"

client = bigquery.Client(project=project_id)
client.delete_table(table_id, not_found_ok=True)
print(f"Dropped table {table_id}")
