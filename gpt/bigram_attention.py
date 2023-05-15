
from typing import Iterator
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads, n_embed, block_size, dropout=0.3):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # run multiple attentions in parallel and concat the outputs
        outputs = [h(x) for h in self.heads]
        out = torch.cat(outputs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):

    def __init__(self, head_size, n_embed, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, self.head_size, bias=False)
        self.query = nn.Linear(n_embed, self.head_size, bias=False)
        self.value = nn.Linear(n_embed, self.head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # q: "what am I looking for?"
        # k: "what do I contain?"
        # if q, k align (dot product), then keep lots of it

        # let's do a single Head

        # Think of x as private to the token
        k = self.key(x) # [B, T, 16]
        q = self.query(x) # [B, T, 16]
        v = self.value(x) # [B, T, 16]

        # Compute attention scores ("affinities")
        
        # wei is the "affinities" -> based on the data, it measures how the features from the history should affect the current token
        # Setting wei = torch.zeros((T, T)) would avoid self-attention, and just average over the full history
        # E.g. I'm a vowel in position 8, show me all the constenants up to position 4
        # the scaling keeps the values ~ 0 to prevent softmax being too peaky.
        wei = q @ k.transpose(-2, -1) * self.head_size ** 0.5 # [B, T, 16] @ [B, 16, T] -> [B, T, T]
        

        # prevent looking into the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [B, T, T]
        wei = F.softmax(wei, dim=-1) # [B, T, T]
        wei = self.dropout(wei)
        out = wei @ v # [B, T, T] @ [B, T, 16] -> [B, T, 16]
        return out
    
class FeedForward(nn.Module):
    """ A single linear layer followed by non-linearity """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
    
    def forward(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        out = self.gamma * xhat + self.beta
        return out
    
    def parameters(self):
        return [self.gamma, self.beta]



class Block(nn.Module):
    """ Transformer block: communication followed by computation"""
    def __init__(self, n_embed, block_size, n_heads, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embed, block_size, dropout)
        self.ff = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed) # normalize the channels
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x)) # eack token thinks individually about the communicated data
        return x

        
class BigramAttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_layer, n_heads, dropout, device):
        super().__init__()
        self.device = device
        self.n_embed = n_embed # H: 32
        self.block_size = block_size # T:
        self.n_heads = n_heads
        self.dropout = dropout

        # Generate an embedding vector for token 12 (e.g. for character "H")
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # Generate an embedding vector for the position in the sequence
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # self.attention = Head(n_embed, n_embed, block_size)
        # head_size = 4
        # num_heads = n_embed // head_size
        # self.sa_heads = MultiHeadAttention(
        #     head_size, num_heads, n_embed, block_size)
        # self.ffwd = FeedForward(n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, block_size, dropout=dropout, n_heads=n_heads) for _ in range(n_layer)]
            
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.ln_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_embeds = self.token_embedding_table(idx) # [B, T, C]
        pos_embeds = self.position_embedding_table(
            torch.arange(T, device=self.device)) # [T, C]

        x = tok_embeds + pos_embeds # [B, T, C]
        x = self.blocks(x)
        x = self.ln_f(x)
    
        logits = self.ln_head(x) # (B, T, vocab_size)

        # [B, T, C] -> [B*T, C] for F.cross_entropy
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx: (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # if the history is too big, truncate so we have enough positional embeddings for it
            idx_cond = idx[:, -self.block_size:]
            # get the predictions for the context
            logits, loss = self(idx_cond)
            # look at the last timestep
            logits = logits[:, -1, :] # [B, C]
            # softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # [B, C]
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled idx to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # [B, T+1]
        return idx

