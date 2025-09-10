'''
step 26, loss: 6.494344711303711, dt: 126.28ms, tok/sec: 129746.79
step 27, loss: 6.382092475891113, dt: 127.75ms, tok/sec: 128252.01
step 28, loss: 6.304886817932129, dt: 126.72ms, tok/sec: 129291.03
step 29, loss: 6.213507175445557, dt: 126.67ms, tok/sec: 129348.22
step 30, loss: 6.203151226043701, dt: 127.10ms, tok/sec: 128903.72
step 31, loss: 6.227931976318359, dt: 127.20ms, tok/sec: 128806.11
step 32, loss: 6.189470291137695, dt: 126.55ms, tok/sec: 129462.02
step 33, loss: 6.312307357788086, dt: 127.51ms, tok/sec: 128496.38
step 34, loss: 6.409172534942627, dt: 126.98ms, tok/sec: 129028.61
step 35, loss: 6.276366233825684, dt: 127.33ms, tok/sec: 128676.83
step 36, loss: 6.224465847015381, dt: 127.49ms, tok/sec: 128512.00
step 37, loss: 6.225424289703369, dt: 126.89ms, tok/sec: 129120.49
step 38, loss: 6.226751327514648, dt: 126.66ms, tok/sec: 129353.09
step 39, loss: 6.044832229614258, dt: 127.57ms, tok/sec: 128432.50
step 40, loss: 6.213140964508057, dt: 126.61ms, tok/sec: 129402.29
step 41, loss: 5.982927322387695, dt: 126.93ms, tok/sec: 129077.81
step 42, loss: 6.102788925170898, dt: 127.26ms, tok/sec: 128740.23
step 43, loss: 6.002697944641113, dt: 127.52ms, tok/sec: 128482.44
'''
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#-------------------------------------------------------------------------
'''
Causal: In the context of GPT-2, "causal" means that the attention mechanism is restricted to only 
consider previous tokens in the sequence. This is crucial for autoregressive models like GPT-2, which 
generate text one token at a time. The model should not have access to future tokens during training 
or inference.
'''
class CausalSelfAttention(nn.Module): # multi head attention
    #-------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class MLP(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Block(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
    '''
    This implementation of GPT follows the same pattern of GPT2. When exploring the
      open weights from GPT2 128M model we would find something like this:
            transformer.wte.weight torch.Size([50257, 768])
            transformer.wpe.weight torch.Size([1024, 768])
            transformer.h.0.ln_1.weight torch.Size([768])
            transformer.h.0.ln_1.bias torch.Size([768])
            transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
            transformer.h.0.attn.c_attn.bias torch.Size([2304])
            transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
            transformer.h.0.attn.c_proj.bias torch.Size([768])
            transformer.h.0.ln_2.weight torch.Size([768])
            transformer.h.0.ln_2.bias torch.Size([768])
            transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
            transformer.h.0.mlp.c_fc.bias torch.Size([3072])
            transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
            transformer.h.0.mlp.c_proj.bias torch.Size([768])

            [...] LAYERS FROM 0 TO 11
            
            transformer.h.11.mlp.c_proj.weight torch.Size([3072, 768])
            transformer.h.11.mlp.c_proj.bias torch.Size([768])
            transformer.ln_f.weight torch.Size([768])
            transformer.ln_f.bias torch.Size([768])
            lm_head.weight torch.Size([50257, 768])

    -------------------

    First we should define some basic hyper parameters: 
        - 50257 is the token vocab size
        - 768 is the embedding dimension
        - 1024 is the block size

    -------------------

    As we can see GPT is composed of: 
        - transformer, which will be explained below
        - lm_head (language model head - LinearLayer) in_features = 50257 (vocab_size), out_features = 768 (embedding dimension), bias  = False
    
    -------------------

    The transformer is composed of:
        - wte (word token embeddings   - Embedding) with num_embeddings = 50257 (vocab_size) and embeddings_dim = 768 (embedding dimension)
        - wpe (word position embedding - Embedding) with num_embeddings = 1024 (block size) and embeddings_dim = 768 (embedding dimension)
        - h (hidden layers, from 0 to 11 to be explained below - ModuleList[Block])
        - ln_f (layer normalization final - LayerNorm) ln_f.weight normalized_shape = 768 (embedding dimension) ln_f.bias normalized_shape = 768 (embedding dimension)

    -------------------

    The h (hidden layers - ModuleList[Block]) is composed of 12 hidden layers of the following components:
        - ln_1 (layer normalization LayerNorm), ln_1.weight normalized_shape = 768, ln_1.bias normalized_shape = 768
        - attn (CausalSelfAttention - note that this is CAUSAL)
        - ln_2 (layer normalization LayerNorm), ln_2.weight normalized_shape = 768, ln_2.bias normalized_shape = 768
        - mlp (MLP Multi Layer Perceptron)

    -------------------

    The CausalSelfAttention
        - c_attn (attention  - linear layer) in_features = 768 (embedding dimension) out_features = 2304 (3 * 768 embedding dimension), bias = 2304 (3 * 768 embedding dimension) 
        - c_proj (projection - linear layer) in_features = 768 (embedding dimension), out_features = 768 (embedding dimension), bias = 768 (embedding dimension) 

    -------------------

    The MLP (Multi Layer Perceptron)
        - c_fc (fully connected - linear layer) in_features = 768 (embedding dimension), out_features = 3072 (4 * 768), bias = 3072 (4 * 768)
        - c_proj (projection    - linear layer) in_features = 3072 (4 * 768), out_features = 768 (embedding dimension), bias = 768 (embedding dimension)

    '''
class GPT(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    '''
    The decorator below (@classmethod) is used to define a method that is bound to the 
    class and not the instance of the class. This means that when you call a class method, 
    it receives the class as its first argument, which is conventionally named cls. This 
    is similar to how instance methods receive the instance as their first argument, 
    conventionally named self.
    They are often used for methods that need to access or modify  class-level attributes, 
    rather than instance-level attributes.
    For instance:
        class MyClass:
            class_attribute = 0
            ...
            @classmethod
            def increment_class_attribute(cls):
                cls.class_attribute += 1
    '''
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------


import tiktoken
#-------------------------------------------------------------------------
class DataLoaderLite:
    #-------------------------------------------------------------------------
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda' 
        print('using cuda acceleration')
    # elif torch.backends.mps.is_built():
    #     device = 'mps'
    #     print('using mps acceleration')
    else:
        device = 'cpu'
        print('using cpu')


    return device
#-------------------------------------------------------------------------
device = get_device()

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024)

torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

import time
#-------------------------------------------------------------------------
# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f"step {i:4d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
#-------------------------------------------------------------------------

import sys; sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#-------------------------------------------------------------------------
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
#-------------------------------------------------------------------------
