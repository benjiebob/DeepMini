
import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # For token 43, the 43rd row of token_embedding_table contains the logits for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # [B, T, C]
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
            # get the predictions for the context
            logits, loss = self(idx)
            # look at the last timestep
            logits = logits[:, -1, :] # [B, C]
            # softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # [B, C]
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled idx to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # [B, T+1]
        return idx

