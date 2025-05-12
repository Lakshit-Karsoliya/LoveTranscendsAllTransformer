import torch 
import torch.nn as nn 
from Attention import Attention

class Encoder(nn.Module):
    '''
    based on prenorm where normalizaiton applied earlier as compared to generic transformer in Attention all you need
    '''
    def __init__(self,emb_dim,hidden_dim,n_heads,RoPE=False,RoPE_Precomputed_Angles=None):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.attn = Attention(emb_dim,n_heads=n_heads,RoPE=RoPE,RoPE_PrecomputedAngles=RoPE_Precomputed_Angles)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )
    def forward(self,input_tensor):
        #input_tensro --> shape [bs,seq_len,emb_dim]
        skip = self.layer_norm1(input_tensor)
        x = self.attn(q=skip,k=skip,v=skip,mask=False)
        x = x+skip
        skip2 = self.layer_norm2(x)
        x = self.mlp(skip2)
        x = x+skip2

        return x
