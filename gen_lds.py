
import torch
from torch import nn

class LDS(nn.Module):
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        dtype: torch.dtype = torch.float32,
        bsz_dim = 896,
        device = torch.device('cuda')
    ):
        """
        state_dim: dimension of LDS hidden state h_t.
        input_dim: dimension of input x_t.
        output_dim: dimension of output.
        kx: AR order (number of taps).
        """
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        self.bsz_dim = bsz_dim
        self.cache_enabled = False
        self.device = device

    
        A_init = torch.randn(state_dim, state_dim, dtype=dtype)
        u, s, v = torch.svd(A_init)   
        A_normalized = A_init / (s[0] + 1e-8)
        self.A = nn.Parameter(A_normalized).to(dtype)
        self.B = nn.Parameter((torch.randn(input_dim, state_dim) / input_dim).to(dtype))
        self.C = nn.Parameter((torch.randn(state_dim, output_dim) / state_dim).to(dtype))

        self.h0 = nn.Parameter(torch.zeros(state_dim, dtype=dtype)).to(device)

        # We'll maintain the hidden state 'h' for recurrent generation.
        self.h = self.h0.unsqueeze(0).expand(bsz_dim, -1).clone().to(device)

    def reset_state(self, batch_size=896):
        """
        Resets the hidden state and AR buffer.
        The hidden state 'h' is set to h0, replicated along the batch dimension.
        """
        self.cache_enabled = False
        self.h = self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(self.device)
        
    def next_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Optimized single-step update:
          1) h_{t+1} = A * h_t + x_t @ B.
          2) lds_out = h_{t+1} @ C.
          3) Update the AR buffer and compute AR output in one optimized step if kx > 0.
        Returns final_out: shape [bsz, output_dim].
        """
        self.h = self.h @ self.A + x_t.matmul(self.B)
        lds_out = self.h.matmul(self.C)
        return lds_out
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LDS model.
    
        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
    
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, output_dim].
        """
        batch_size, seq_len, _ = inputs.size()
        # Reset the hidden state and AR buffer for a new sequence.
        if not self.cache_enabled:
            self.reset_state(batch_size)
        if seq_len == 1:
            y_t = self.next_step(inputs.squeeze(1))
            return y_t.unsqueeze(1)  # shape => [batch_size, 1, output_dim]
        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            y_t = self.next_step(x_t)
            outputs.append(y_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)
    
    def generate_trajectory(self, us, h0=None):
        _, d_u = us.shape
        assert d_u == self.input_dim, (d_u, self.input_dim)
        
        if h0 is not None:
            h_t = h0
        else:
            h_t = self.h0
            
        A = self.A
        obs = []
        
        for u in us:
            h_t =h_t @ A + (u @ self.B)
            o_t = h_t @ self.C
            obs.append(o_t)
        
        obs = torch.stack(obs, dim=0)
        return obs

def random_LDS(d_h: int, d_o: int, d_u: int, lower_bound: float = 0, device = torch.device('cpu'), dtype = torch.float32):
  """
  makes a random LDS with hidden state dimension d_h, observation dimension d_o, and control dimension d_u.
  `lower_bound` is a float in [0, 1] specifying the minimum absolute value for entries in A.
  Each entry in A will be in [lower_bound, 1] multiplied by +/-1 with equal probability.
  """
  # Create LDS instance
  lds = LDS(state_dim=d_h, input_dim=d_u, output_dim=d_o, device = device, bsz_dim = 1, dtype=dtype)
  
  # Override the A parameter with custom initialization
  A_init = torch.randn(d_h, d_h, dtype=dtype, device=device)
  u, s, v = torch.svd(A_init)
  scaling_factor = (torch.rand(1, device=device) * (1 - lower_bound) + lower_bound) / (s[0] + 1e-8)
  A_normalized = A_init * scaling_factor
  lds.A = nn.Parameter(A_normalized)
  
  # Initialize other parameters randomly
  lds.B = nn.Parameter(torch.randn(d_u, d_h).to(dtype = dtype, device = device) / d_u)
  lds.C = nn.Parameter(torch.randn(d_h, d_o).to(dtype = dtype, device = device)/ d_h)
  lds.h0 = nn.Parameter(torch.zeros(d_h).to(dtype = dtype, device = device))
  
  return lds