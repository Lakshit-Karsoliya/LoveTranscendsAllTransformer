import torch 
import torch.nn as nn
import math


class Attention(nn.Module):
    """
    It can be done inside the same attention module as above but don't know why I chose this.
    """
    def __init__(self, emb_dim, n_heads=8, RoPE=False, RoPE_PrecomputedAngles=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)
        self.RoPE = RoPE
        self.RoPE_PrecomputedAngles = RoPE_PrecomputedAngles
        if self.RoPE:
            sin = torch.sin(self.RoPE_PrecomputedAngles)
            cos = torch.cos(self.RoPE_PrecomputedAngles)
            self.sin = sin[None, None, :, :]
            self.cos = cos[None, None, :, :]


    def apply_RoPE(self, input_tensor):
        bs, num_head, seq_len, head_dim = input_tensor.shape 
        sin = self.sin[:, :, :seq_len, :] 
        cos = self.cos[:, :, :seq_len, :]

        input_tensor = input_tensor.view(bs, num_head, seq_len, head_dim // 2, 2)
        t1 = input_tensor[..., 0]
        t2 = input_tensor[..., 1]
        rotated_embedding = torch.stack(
            [t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1
        )
        return rotated_embedding.view(bs, num_head, seq_len, head_dim)
    
    def create_causal_mask(self, seq_len, device):
        """
        Returns a boolean causal mask of shape [1, 1, seq_len, seq_len]
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        return mask[None, None, :, :]  # shape -> [1, 1, seq_len, seq_len]

    def forward(self, q, k, v,mask=False):
        bs, q_seq_len, _ = q.shape
        _, k_seq_len,_ = k.shape
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        q = q.reshape(bs, q_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bs, k_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bs, k_seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.RoPE:
            q = self.apply_RoPE(q)
            k = self.apply_RoPE(k)
        if mask:
            causal_mask = self.create_causal_mask(k_seq_len, q.device)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask:
            attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn = torch.matmul(attn_probs, v)
        attn = attn.transpose(1, 2).reshape(bs, q_seq_len, self.emb_dim).contiguous()
        return attn
