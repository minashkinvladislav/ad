import torch
import torch.nn as nn
from torch.nn import functional as F

#------------------TRANSFORMER ARCHITECTURE-------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout, batch_first=True)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones((3 * block_size, 3 * block_size), dtype=torch.bool))
        )
    def forward (self, x):
        B, T, C = x.shape
        causal_mask = self.causal_mask[:T, :T]
        norm_x = self.ln1(x)
        x = x + self.sa(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            need_weights=False,
        )[0]
        x = x + self.ffwd(self.ln2(x))
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, n_embd, n_head, block_size, n_layer, dropout, n_steps, n_states, n_actions):
        super().__init__()
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.timestep_emb = nn.Embedding(n_steps, n_embd)
        self.state_emb = nn.Embedding(n_states, n_embd)
        self.action_emb = nn.Embedding(n_actions, n_embd)
        self.return_emb = nn.Embedding(2, n_embd)
        self.action_head = nn.Linear(n_embd, n_actions)

    def forward(self, states, actions, rewards, timesteps, targets=None):
        B, T = states.shape # (Batch, Time (context))
        time_emb = self.timestep_emb(timesteps)
        act_emb = self.action_emb(actions) + time_emb
        state_emb = self.state_emb(states) + time_emb
        returns_emb = self.return_emb(rewards) + time_emb # (B, T, Embed)
        # [batch_size, seq_len * 3, emb_dim], (s_0, a_0, r_0, s_1, a_1, r_1, ...)
        sequence = (
            torch.stack([state_emb, act_emb, returns_emb], dim=1) # (B, M, T, Embed)
            .permute(0, 2, 1, 3) # (B, T, M, Embed)
            .reshape(B, 3 * T, time_emb.shape[2])
        ) # (B, Modality * T, N_embed)
        logits = self.blocks(sequence)
        # make predictions only based on state
        logits = self.action_head(logits[:, 0::3]) # (B,T,C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
