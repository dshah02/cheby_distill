import os
import torch
import random
from model import STU, get_polynomial_spectral_filters
from gen_lds import random_LDS

# Set device to CPU
device = torch.device("cpu")
torch_dtype = torch.float32

# Create a configuration class to store model parameters
class ModelConfig:
    def __init__(self):
        self.seq_len = 8192
        self.num_filters = 24
        self.torch_dtype = torch.float32
        self.use_tensordot = False
        self.dim = 1
        self.use_flash_fft = False

# Initialize the config
config = ModelConfig()

# Model dimensions
d_h = 10
d_in = 1
d_out = 1

# Get polynomial spectral filters
spectral_filters = get_polynomial_spectral_filters(
    seq_len=config.seq_len,
    k=config.num_filters,
    device=device,
    dtype=torch_dtype,
)

def load_model_pair(lds_path, stu_path):
    try:
        # Initialize LDS
        lds = random_LDS(d_h, d_in, d_out, device=device, dtype=torch_dtype)
        lds.load_state_dict(torch.load(lds_path, map_location=device))
        lds.eval()
        
        # Initialize STU
        stu = STU(config, spectral_filters).to(device=device, dtype=torch_dtype)
        stu.load_state_dict(torch.load(stu_path, map_location=device))
        stu.eval()
        
        return lds, stu
    except Exception as e:
        print(f"Error loading models from {lds_path} and {stu_path}: {str(e)}")
        return None, None

def main():
    # Get all model IDs from the LDS directory
    lds_dir = "./models/gen_lds_trained"
    stu_dir = "./models/gen_stu_trained"
    
    if not os.path.exists(lds_dir) or not os.path.exists(stu_dir):
        print("Model directories not found!")
        return
    
    # Get all model IDs
    model_ids = [f.split('.')[0] for f in os.listdir(lds_dir) if f.endswith('.pt')]
    
    # Load each model pair
    loaded_pairs = []
    for model_id in model_ids:
        lds_path = os.path.join(lds_dir, f"{model_id}.pt")
        stu_path = os.path.join(stu_dir, f"{model_id}.pt")
        
        if not os.path.exists(stu_path):
            print(f"STU model not found for ID {model_id}")
            continue
            
        lds, stu = load_model_pair(lds_path, stu_path)
        if lds is not None and stu is not None:
            loaded_pairs.append((model_id, lds, stu))
            print(f"Successfully loaded model pair {model_id}")
    
    print(f"\nLoaded {len(loaded_pairs)} model pairs successfully")
    return loaded_pairs

if __name__ == "__main__":
    loaded_pairs = main() 