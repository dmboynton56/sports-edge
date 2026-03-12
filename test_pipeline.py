import sys
import os

# Ensure the src folder is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_builder import TrainingDatasetBuilder

def run_test():
    print("Initializing TrainingDatasetBuilder with API fallback...")
    builder = TrainingDatasetBuilder(use_api_fallback=True)
    
    # We will build a small dataset and output to a test file
    df = builder.build_dataset(limit=10, output_path="data/processed/test_training_dataset.csv")
    
    if df is not None and not df.empty:
        print(f"Success! Dataset built with shape: {df.shape}")
        print("Sample of processed data:")
        print(df.head())
    else:
        print("Failed to build dataset or dataset is empty.")

if __name__ == "__main__":
    run_test()
