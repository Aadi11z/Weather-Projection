import torch
import torch.nn as nn
import torch.optim as optim
from models import CNNLSTM_Downscaler
from data_loader import get_dataloaders
import os

def train_model(gcm_path, era5_path, gcm_name='GCM', epochs=50, batch_size=8, lr=1e-3, seq_length=14, patience=7):
    """
    Train a CNN-LSTM downscaler for a given GCM input dataset.

    Args:
        gcm_path (str): Path to the coarse GCM input CSV (3x3 grid).
        era5_path (str): Path to the ERA5 target CSV (17x17 grid).
        gcm_name (str): Name of the GCM (used for checkpoint naming and logs).
        epochs (int): Maximum training epochs.
        batch_size (int): Batch size for DataLoaders.
        lr (float): Initial learning rate.
        seq_length (int): Sliding window length in days.
        patience (int): Early stopping patience (epochs without improvement).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    if not os.path.exists(era5_path):
        print(f"ERA5 target dataset at {era5_path} is missing.")
        print("Please ensure the data generation script has completed creating the spatial matrix before training.")
        return

    # 1. Load Data
    print("Initializing Data Loaders...")
    train_loader, val_loader = get_dataloaders(
        gcm_path, era5_path,
        batch_size=batch_size, seq_length=seq_length,
        gcm_name=gcm_name
    )
    
    # 2. Instantiate Model
    print("Instantiating CNN-LSTM Model...")
    model = CNNLSTM_Downscaler(in_channels=5, hidden_channels=64, out_channels=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    # 3. Loss & Optimizer
    # HuberLoss is ideal for climate arrays as it penalizes extreme outliers less aggressively than MSE, 
    # preventing unstable gradients from sudden storm/extreme heat pixels.
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning Rate Scheduler: halves LR when val loss plateaus for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 4. Training Loop
    checkpoint_path = f"best_downscaler_{gcm_name}.pth"
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            # outputs shape: (Batch, Seq_Len, 5, 17, 17)
            outputs = model(x)
            
            # Compute loss over the entire spatiotemporal output sequence
            loss = criterion(outputs, y)
            
            loss.backward()
            # Gradient clipping prevents exploding gradients with normalized features
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.1e}")
        
        # 6. Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # 7. Checkpointing + Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f">> Saved new improved {gcm_name} model checkpoint!")
        else:
            epochs_no_improve += 1
            print(f"   No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best Val Loss: {best_val_loss:.4f}")
                break

if __name__ == "__main__":
    # ─── GCM CONFIGURATION ────────────────────────────────────
    # Each entry maps a GCM name to its historical training CSV.
    # The model trains one downscaler per GCM, all against the same ERA5 ground truth.
    GCM_CONFIGS = {
        'MIROC6':  'MIROC6_UAE_Spatial_Input_1950_2014.csv',
        'MPI-ESM': 'MPI_ESM_UAE_Spatial_Historical_1950_2014.csv',
        # 'EC-Earth3': 'EC_Earth3_UAE_Spatial_Historical_1950_2014.csv',  # uncomment when available
    }

    ERA5_TARGET = "UAE_ERA5_Spatial_Baseline_1950_2014.csv"

    print("=" * 60)
    print("  UAE MULTI-GCM CLIMATE DOWNSCALING TRAINER")
    print("=" * 60)

    for gcm_name, gcm_path in GCM_CONFIGS.items():
        print(f"\n{'─' * 60}")
        print(f"  Training downscaler for: {gcm_name}")
        print(f"  Input: {gcm_path}")
        print(f"{'─' * 60}")

        if not os.path.exists(gcm_path):
            print(f"  SKIP: {gcm_path} not found. Run the {gcm_name} fetch script first.")
            continue

        train_model(gcm_path, ERA5_TARGET, gcm_name=gcm_name)
        print(f"\n  {gcm_name} training complete.\n")

    print("=" * 60)
    print("  ALL TRAINING COMPLETE")
    print("=" * 60)
