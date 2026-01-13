import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.td_predictor import TDScorerPredictor

def main():
    predictor = TDScorerPredictor(model_version='v1')
    # Train on recent seasons
    seasons = [2021, 2022, 2023, 2024]
    predictor.train(seasons)
    print("Training complete.")

if __name__ == "__main__":
    main()
