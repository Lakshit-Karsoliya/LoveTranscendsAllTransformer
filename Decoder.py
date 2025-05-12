import torch.nn as nn
from Attention import Attention

class Decoder(nn.Module):
    """
        Using prenorm and able to ingest RoPE 
        May be i add SwiGLU cuz I am worthy
    """
    def __init__(self,emb_dim,hidden_dim,n_heads,RoPE=False,RoPE_Precomputed_Angles=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.mha = Attention(emb_dim,n_heads=n_heads,RoPE=RoPE,RoPE_PrecomputedAngles=RoPE_Precomputed_Angles)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mmha = Attention(emb_dim,n_heads=n_heads,RoPE=RoPE,RoPE_PrecomputedAngles=RoPE_Precomputed_Angles)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self,input_tensor,encoder_output=None):
        # input_tensor --> [bs,seq_len,emb_dim]
        skip1 = input_tensor
        x = self.norm1(input_tensor)
        x = self.mha(q=x,k=x,v=x,mask=True)
        x = x+skip1 
        
        skip2 = x 
        x = self.norm2(x)
        if encoder_output is None:
            x = self.mmha(q=x,k=x,v=x,mask=True)
        else:
            x = self.mmha(q=x,k=encoder_output,v=encoder_output,mask=False)
        x = self.mlp(x)
        x = x + skip2 
        return x 



