import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import joblib

class PGACourseFitNN(nn.Module):
    """
    A PyTorch Neural Network that learns complex non-linear interactions 
    between a player's style and course/environmental conditions.
    """
    def __init__(self, num_players: int, num_courses: int, num_continuous_features: int, 
                 embedding_dim: int = 16, hidden_dims: list = [64, 32]):
        super(PGACourseFitNN, self).__init__()
        
        # 1. Entity Embeddings for Categorical Variables
        # This maps a discrete Player ID to a dense vector representing their "style"
        self.player_embedding = nn.Embedding(num_embeddings=num_players, embedding_dim=embedding_dim)
        
        # This maps a discrete Course ID to a dense vector representing its "difficulty/layout"
        self.course_embedding = nn.Embedding(num_embeddings=num_courses, embedding_dim=embedding_dim)
        
        # 2. Continuous Feature Processing
        # We will feed in things like wind_speed, recent_form_sg, etc.
        self.continuous_bn = nn.BatchNorm1d(num_continuous_features)
        
        # 3. Deep Feed-Forward Network
        # The input to the first hidden layer is: Player Embedding + Course Embedding + Continuous Features
        input_dim = (embedding_dim * 2) + num_continuous_features
        
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2)) # Prevent overfitting
            input_dim = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        
        # 4. Output Layer
        # Predicts a single continuous value: Expected Strokes Gained
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, player_idx, course_idx, continuous_features):
        # Extract embeddings
        p_emb = self.player_embedding(player_idx)
        c_emb = self.course_embedding(course_idx)
        
        # Normalize continuous features
        cont_x = self.continuous_bn(continuous_features)
        
        # Concatenate everything together into one big feature vector per row
        x = torch.cat([p_emb, c_emb, cont_x], dim=1)
        
        # Pass through the Multi-Layer Perceptron (MLP)
        x = self.mlp(x)
        
        # Predict Expected SG
        out = self.output_layer(x)
        return out.squeeze() # Return a 1D tensor

class CourseFitTrainer:
    """
    Handles the data preparation, training loop, and evaluation for the PyTorch model,
    specifically targeting the RTX 5070 GPU.
    """
    def __init__(self, models_dir: str = "data-core/models/saved/"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        # Automatically detect and use the RTX 5070 if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initialized PyTorch Trainer using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target_sg_per_round'):
        """
        Converts a Pandas DataFrame into PyTorch Tensors.
        Requires 'name' (player) and 'course_id' (or 'tournament') as categorical features.
        """
        print("Preparing data for PyTorch...")
        
        # Drop rows missing the target
        df = df.dropna(subset=[target_col]).copy()
        
        # 1. Encode Categoricals to Integers (0 to N-1)
        # We save these mappings so we can encode live players later!
        player_mapping = {name: idx for idx, name in enumerate(df['name'].unique())}
        course_mapping = {course: idx for idx, course in enumerate(df['tournament'].unique())}
        
        df['player_idx'] = df['name'].map(player_mapping)
        df['course_idx'] = df['tournament'].map(course_mapping)
        
        joblib.dump(player_mapping, os.path.join(self.models_dir, 'player_mapping.joblib'))
        joblib.dump(course_mapping, os.path.join(self.models_dir, 'course_mapping.joblib'))
        
        # 2. Extract Continuous Features
        meta_cols = ['season', 'start', 'end', 'tournament', 'location', 'name', 'position_str', 'position_num', 'dataset_split', 'player_idx', 'course_idx']
        target_cols = ['target_sg_total', 'target_sg_per_round', 'target_made_cut', 'target_top10', 'target_top20', 'target_win']
        cols_to_drop = [c for c in meta_cols + target_cols if c in df.columns]
        
        continuous_cols = [col for col in df.columns if col not in cols_to_drop]
        df[continuous_cols] = df[continuous_cols].fillna(0) # Simple imputation
        
        # 3. Convert to PyTorch Tensors
        # Categoricals must be LongTensors (integers)
        player_tensor = torch.tensor(df['player_idx'].values, dtype=torch.long)
        course_tensor = torch.tensor(df['course_idx'].values, dtype=torch.long)
        
        # Continuous must be FloatTensors
        continuous_tensor = torch.tensor(df[continuous_cols].values, dtype=torch.float32)
        
        # Target must be FloatTensor
        target_tensor = torch.tensor(df[target_col].values, dtype=torch.float32)
        
        # 4. Create DataLoader for batching
        dataset = TensorDataset(player_tensor, course_tensor, continuous_tensor, target_tensor)
        
        return dataset, len(player_mapping), len(course_mapping), len(continuous_cols)

    def train(self, dataset, num_players: int, num_courses: int, num_continuous: int, epochs: int = 20, batch_size: int = 512):
        """
        Executes the training loop on the GPU.
        """
        # Create DataLoaders (split into 80/20 train/val)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Pin memory speeds up CPU to GPU data transfer!
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # Initialize the Model and move it to the RTX 5070
        model = PGACourseFitNN(num_players, num_courses, num_continuous).to(self.device)
        
        # Loss function (Mean Squared Error for Regression) and Optimizer (AdamW)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        print(f"\nStarting PyTorch Training for {epochs} epochs on {self.device}...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train() # Set to training mode (enables dropout/batchnorm)
            train_loss = 0.0
            
            # The Training Loop
            for batch_idx, (p_idx, c_idx, cont_x, targets) in enumerate(train_loader):
                # 1. Move data to GPU
                p_idx, c_idx, cont_x, targets = p_idx.to(self.device), c_idx.to(self.device), cont_x.to(self.device), targets.to(self.device)
                
                # 2. Forward pass (predict)
                predictions = model(p_idx, c_idx, cont_x)
                
                # 3. Calculate loss
                loss = criterion(predictions, targets)
                
                # 4. Backward pass (calculate gradients)
                optimizer.zero_grad()
                loss.backward()
                
                # 5. Update weights
                optimizer.step()
                
                train_loss += loss.item() * p_idx.size(0)
                
            train_loss /= len(train_dataset)
            
            # The Validation Loop
            model.eval() # Set to evaluation mode
            val_loss = 0.0
            with torch.no_grad(): # Don't track gradients (saves memory/time)
                for p_idx, c_idx, cont_x, targets in val_loader:
                    p_idx, c_idx, cont_x, targets = p_idx.to(self.device), c_idx.to(self.device), cont_x.to(self.device), targets.to(self.device)
                    predictions = model(p_idx, c_idx, cont_x)
                    loss = criterion(predictions, targets)
                    val_loss += loss.item() * p_idx.size(0)
            
            val_loss /= len(val_dataset)
            val_rmse = np.sqrt(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(self.models_dir, 'pytorch_course_fit.pth'))
                
        print(f"Training Complete. Best Validation RMSE: {np.sqrt(best_val_loss):.4f}")
        print(f"Model weights saved to {os.path.join(self.models_dir, 'pytorch_course_fit.pth')}")

if __name__ == "__main__":
    # Quick test execution
    print("Loading feature store for PyTorch training...")
    df = pd.read_csv('data-core/notebooks/cache/pga_feature_store_event_level.csv')
    
    trainer = CourseFitTrainer()
    dataset, num_p, num_c, num_cont = trainer.prepare_data(df)
    
    # Train for 10 epochs as a test run
    trainer.train(dataset, num_players=num_p, num_courses=num_c, num_continuous=num_cont, epochs=10)
