import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from bigram import BigramLanguageModel
from bigram_attention import BigramAttentionLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
max_iters = 5000
batch_size = 64
block_size = 256
eval_iters = 500
learning_rate = 3e-4
n_embed = 384
num_heads = 6
num_layers = 6
dropout = 0.2

def load_words():
    input_dir = "/Users/benbiggs/Documents/code/DeepMini/data/harrypotter"
    word_list = []
    books = sorted(os.listdir(input_dir))
    for fn in books:
        path = os.path.join(input_dir, fn)
        with open(path, "r") as f:
            data = f.read()
            word_list.append(data)

    # combine all words
    words = "\n".join(word_list)
    return words

def get_batch(data, batch_size=4, block_size=8, verbose=False):  
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    for b in range(batch_size): # batch dimension
        if verbose:
            print (x[b])
            print (y[b])
        for t in range(block_size): # time dimension
            context = x[b, :t+1]
            target = y[b, t]
            if verbose:
                print (f"input is {context.tolist()}, target is {target}")
    
    return x.to(device), y.to(device)

def get_tokenizer(chars):
    # tokenize the input text
    # maps, str -> integers
    # google uses, sentence piece (google), subword unit level
    # tiktoken (openai), this is what gpt uses
    stoi = { ch:i for i,ch in enumerate(chars)}
    itos = { i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return encode, decode

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for name, split in [['train', train_data], ['val', val_data]]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean()
    model.train()
    return out




def main():
    # load harry potter
    text = load_words()
    # print first 1000 characters
    print (text[:1000])
    # get the unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print (''.join(chars))
    print (vocab_size)

    encode, decode = get_tokenizer(chars)

    print (encode("hello there!"))
    print (decode(encode("hello there!")))

    # encode the entire harry potter dataset
    data = torch.tensor(encode(text), dtype=torch.long)
    print (data.shape, data.dtype)
    print (data[:1000])

    # split up into train and val
    # val is the last few paragraphs of the book
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # block_size / context length is the max length of the context
    # we get the transformer used to seeing contexts of 1 -> block_size
    # target is always the next character
    # we can never show the transformer more than block_size as input
    xb, yb = get_batch(
        train_data, batch_size=1, block_size=8, 
        verbose=True)

    print (xb.shape) # [4, 8]
    print (yb.shape) # [4, 8]

    # model = BigramLanguageModel(vocab_size)
    model = BigramAttentionLanguageModel(
        vocab_size, block_size, n_embed, num_layers, num_heads, dropout, device)
    model = model.to(device)

    # logits, loss = model(xb, yb)
    # print (logits.shape) # [4, 8, 92]
    # print (loss)

    # create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print (f"step {iter}: train loss {losses['train']}, val_loss {losses['val']}")

        # sample a batch
        xb, yb = get_batch(train_data)

        # evaluate
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print (loss.item())
    
      # test the generation model
    test_idx = torch.zeros((1, 1), dtype=torch.long, device=device) # [B, T] of 0 (new line character)
    test_pred_idx = model.generate(test_idx, max_new_tokens=100)
    test_chars = decode(test_pred_idx[0].tolist())
    print (test_chars)



    

   


    




if __name__ == "__main__":
    main()