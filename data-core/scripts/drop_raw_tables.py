from google.cloud import bigquery
import os

def drop_tables():
    project = "learned-pier-478122-p7"
    client = bigquery.Client(project=project)
    tables = [
        "sports_edge_raw.raw_schedules",
        "sports_edge_raw.raw_pbp",
        "sports_edge_raw.raw_team_stats"
    ]
    for table_id in tables:
        full_id = f"{project}.{table_id}"
        print(f"Dropping {full_id}...")
        client.delete_table(full_id, not_found_ok=True)
    print("Done.")

if __name__ == "__main__":
    drop_tables()
