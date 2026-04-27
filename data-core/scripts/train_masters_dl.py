#!/usr/bin/env python3
import os
import sys
import pandas as pd
import joblib
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.dl_course_fit import CourseFitTrainer

def main():
    print("Loading feature store for PyTorch Course Fit training...")
    df = pd.read_csv('notebooks/cache/pga_feature_store_event_level.csv')
    
    trainer = CourseFitTrainer(models_dir='models/saved')
    dataset, num_p, num_c, num_cont = trainer.prepare_data(df)
    
    # Train for 20 epochs
    trainer.train(dataset, num_players=num_p, num_courses=num_c, num_continuous=num_cont, epochs=20)
    
    print("PyTorch Course Fit model trained and saved successfully.")

if __name__ == "__main__":
    main()
