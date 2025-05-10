import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    STU,
    get_polynomial_spectral_filters,
)

from lds import random_LDS

torch.set_float32_matmul_precision("high")

# SEED = 1746
# np.random.seed(SEED)
# torch.manual_seed(SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a configuration class to store model parameters
class ModelConfig:
    def __init__(self):
        self.seq_len = 1024
        self.num_filters = 24
        self.torch_dtype = torch.float32
        self.use_tensordot = False
        self.dim = 1
        self.use_flash_fft = False

# Initialize the config
config = ModelConfig()

seq_len = config.seq_len
num_filters = config.num_filters
torch_dtype = config.torch_dtype
use_tensordot = config.use_tensordot
dim = config.dim

d_h = 1
d_in = 1
d_out = 1

bsz = 16
step_cnt = 2000

lds = random_LDS(d_h, d_in, d_out, device= device, dtype = torch_dtype).to(dtype = torch_dtype, device = device)

# Get polynomial spectral filters
spectral_filters = get_polynomial_spectral_filters(
    seq_len=seq_len,
    k=num_filters,
    device=device,
    dtype=torch_dtype,
)


import random
random_id = random.randint(100000, 999999)
print('model id', random_id)

def train_stu(lds, steps, verbose=True):
    model = STU(config, spectral_filters).to(device = device, dtype = torch_dtype)
    lr = 1
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    model.train()

    for step in range(steps):
        inputs = torch.randn(bsz * seq_len, d_in).to(device = device, dtype = torch_dtype)
        
        # Use torch.no_grad() to avoid storing gradient information for LDS
        with torch.no_grad():
            targets = lds.generate_trajectory(inputs)

        inputs = inputs.reshape(bsz, seq_len, d_in).to(device).type(torch_dtype)
        targets = targets.reshape(bsz, seq_len, d_out).to(device).type(torch_dtype)
        outputs = model.forward(inputs)
        # print(outputs, targets)
        loss = F.mse_loss(outputs, targets)
        print(f"LOSS {random_id}:", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model, loss

stu = train_stu(lds, step_cnt)


lds_path = f'./models/lds_trained/{random_id}.pt'
torch.save(lds.state_dict(), lds_path)

# Save the STU state dict
stu_model, final_loss = stu  # Unpack the model and loss from train_stu return value
stu_path = f'./models/stu_trained/{random_id}.pt'
torch.save(stu_model.state_dict(), stu_path)
