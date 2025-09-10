'''
step   17 | loss: 6.578763 | norm: 0.8878 | dt: 136.34ms | tok/sec: 120166.47
step   18 | loss: 6.534562 | norm: 1.1942 | dt: 135.12ms | tok/sec: 121254.19
step   19 | loss: 6.346401 | norm: 0.8581 | dt: 134.66ms | tok/sec: 121671.32
step   20 | loss: 6.447803 | norm: 1.5113 | dt: 136.34ms | tok/sec: 120171.93
step   21 | loss: 6.170336 | norm: 1.3388 | dt: 135.41ms | tok/sec: 120993.30
step   22 | loss: 6.289427 | norm: 3.2838 | dt: 135.67ms | tok/sec: 120761.54
step   23 | loss: 6.169595 | norm: 1.1067 | dt: 136.74ms | tok/sec: 119818.03
step   24 | loss: 6.129137 | norm: 1.0382 | dt: 134.48ms | tok/sec: 121834.84
step   25 | loss: 6.344614 | norm: 1.4192 | dt: 134.74ms | tok/sec: 121594.47
step   26 | loss: 6.447644 | norm: 1.6790 | dt: 135.61ms | tok/sec: 120815.26
step   27 | loss: 6.311741 | norm: 1.1721 | dt: 128.96ms | tok/sec: 127046.55
step   28 | loss: 6.203239 | norm: 1.0882 | dt: 124.54ms | tok/sec: 131552.76
step   29 | loss: 6.128735 | norm: 1.1578 | dt: 123.03ms | tok/sec: 133170.83
step   30 | loss: 6.092750 | norm: 0.7807 | dt: 124.44ms | tok/sec: 131664.92
step   31 | loss: 6.093057 | norm: 0.7099 | dt: 123.56ms | tok/sec: 132595.25
step   32 | loss: 6.056186 | norm: 1.0320 | dt: 124.40ms | tok/sec: 131705.80
step   33 | loss: 6.206657 | norm: 1.3467 | dt: 124.01ms | tok/sec: 132120.06
step   34 | loss: 6.282577 | norm: 0.9424 | dt: 123.90ms | tok/sec: 132232.18
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

        # key, query, value projections for all heads, but in a batch, instead of 3 linear layers for q,v,k we have
        #   just 1 with enough bandwith for the 3 of them
        #  concatenated attention - https://arxiv.org/pdf/1706.03762 - page 4 - Multi-Head Attention - right image
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # (768, 768 * 3) -> (768, 2304)

        # output projection
        #  concatenated projection - https://arxiv.org/pdf/1706.03762 - page 4 - Multi-Head Attention - right image
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
	
        self.c_proj.NANOGPT_SCALE_INIT = 1
	
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        mask = torch.tril( #triangular matrix
            torch.ones(config.block_size, config.block_size) # ones matrix of size (block_size, block_size) -> (1024, 1024)
        ).view(1, 1, config.block_size, config.block_size) # creates a view with dim (1,1,1024,1024) basically adds 2 more dummy dims

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", mask)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) B = 5, T = 8, C = 768

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(input = x) # [5, 8, 2304]

        # all [5, 8, 768]
        q, k, v = qkv.split(split_size = self.n_embd, dim=2) # split_size = 768, but the output is 2304, so 3 splits

        # the view splits the last dimension (768) and doesn't alter the first one, as their
        #   numbers were unchanged. But in general this is a simple reorganization as the number
        #   of elements didn't change.
        #   [5, 8, 12, 64] |||  from [5, 8, 768] -> view [5, 8, 12, 64] -> [5, 12, 8, 64]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 


        #--------------------------------------------------
        # Replace this by flash attention - start

        # # multiply Q (query) by K (key), and traspose the second matrix K, then apply the scaling factor
        # # attention (materializes the large (T,T) matrix for all the queries and keys)
        # # att [5, 12, 8, 8] <- q [5, 12, 8, 64] ,  k.trans [5, 12, 64, 8]
        # att = (q @ k.transpose(-2, -1)) * ( 1.0 / math.sqrt(k.size(-1)) )

        # # apply mask - don't let it peek into the future tokens
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        # att = F.softmax(att, dim=-1)

        # # y [5, 12, 8, 64] , att [5, 12, 8, 8]  , v [5, 12, 8, 64]
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


        # # .contiguous(): This ensures that the tensor's memory layout is contiguous.
        # # y [5, 8, 768]   |   y.trans.cont [5, 8, 12, 64] -> y.view [5, 8, 768] | B = 5, T = 8, C = 768
                

        # Replace this by flash attention - end
        #--------------------------------------------------
        '''
        The old code above was challenging for torch.compile to optimize. Flash attention is a 'kernel
        fusion' operation. Flash attention can be up to 7.6X faster than the old implementation
        '''

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
	
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection  - [5, 8, 768]
        y = self.c_proj(y)

        return y
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class MLP(nn.Module):
    #-------------------------------------------------------------------------
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) # context fully connected
        self.gelu    = nn.GELU(approximate='tanh') # GELU activation will use the tanh approximation for its computation, which can be faster than the exact computation.
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) # context projection
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
    # note you can define the values here in any order
    block_size : int = 1024 # max sequence length
    vocab_size : int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer    : int = 12 # number of layers
    n_head     : int = 12 # number of heads
    n_embd     : int = 768 # embedding dimension
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
    def __init__(self, gpt_config : GPTConfig):
        super().__init__()

        self.config = gpt_config

        # below we need to define the following:
        #   transformer and lm_head
        #
        # and transformer is composed of: 
        #   wte, wpe, h, ln_f
        #--------
        # transformer dict - define wte, wpe, h, ln_f

        # h - hidden layers. composed of a list of Block objects
        block_list = [ Block(gpt_config) 
                       for _ in range(gpt_config.n_layer) ]

        # the transformer dictionary 
        transf_dict = dict(
                wte  = nn.Embedding(gpt_config.vocab_size, gpt_config.n_embd),
                wpe  = nn.Embedding(gpt_config.block_size, gpt_config.n_embd),
                h    = nn.ModuleList(block_list),
                ln_f = nn.LayerNorm(gpt_config.n_embd) )
        #--------
        # define transformer
        self.transformer = nn.ModuleDict( transf_dict )
        #--------
        #define lm_head
        self.lm_head = nn.Linear(gpt_config.n_embd, gpt_config.vocab_size, bias=False)
        #--------

        '''
        An explanation is necessary for the weight sharing below: transformer.wte (token embedding)
        is used to convert input token IDs into dense vectors. lm_head is used to convert the final 
        hidden states back into token probabilities for the output. By sharing weights, the model 
        ensures that the same vector space is used for both input and output tokens. This symmetry 
        can help the model learn more effectively, as the same representation is used consistently 
        throughout the network. Plus, research and empirical results have shown that weight sharing 
        can lead to better performance in language models
        '''
	
        # weight sharing scheme - transformer token embeddings take the weights of the language model head weights
        self.transformer.wte.weight = self.lm_head.weight
	

        # init params
        self.apply(self._init_weights)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _init_weights(self, module):
        # just trying to make sure things are initialized correctly. 
        #   Linear layers should be std = 0.02 and a mean of 0, except if they have a flag NANOGPT_SCALE_INIT
        #     this flag is reserved for accumulation residual layers in this case they should be 
        #     0.02 * 1 / sqrt(2 * self.config.n_layer) -> 0.02 * (2 * self.config.n_layer) ** -0.5
        #     this helps to control the growth of activations in the forward pass (for res nets)
        #     This will typically affect the C_PROJ (linear) of MLP and CausalSelfAttention. You can
        #     confirm these are the 2 last res nets of these components.
        # Note that the values around 0.02 are a mere simplification of the Xavier initialization.
        #   for many parameters as in d_model = 768, we would have 1/sqrt(768) = 0.03608439182
        #   or if we were using d_model as 1600, we would have 1/sqrt(1600) = 0.025

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
        #  create a 1-dim tensor 
        pos = torch.arange(start = 0, end = T, dtype=torch.long, device=idx.device) # shape (T)

        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        
        x = tok_emb + pos_emb # residual connection / skip connection
        
        # forward the blocks of the transformer hidden layers
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)

        # linear layer
        logits = self.lm_head(x) # (B, T, vocab_size)


        loss = None
        if targets is not None:

            # for logits.view(-1, logits.size(-1)):
            #    -1 means guess the size, while we define the size of the last dim as being 5027
            #    since in one example logits is [4, 32, 50257], that means [ 4 * 32 , 5027 ] -> [ 128, 5027 ]
            # for targets.view(-1)
            #     we are amalgamating targets into a single dim, therefore [128], targets shape is [4, 32]

            loss = F.cross_entropy(input = logits.view(-1, logits.size(-1)) , target = targets.view(-1))


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

        #-------
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        #-------

        #-------
        # select pre config of the model type
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        print('-------')
        print('config_args.items()')
        for k,v in config_args.items():
            print(f"    k: {k} - v: {v}")
        print('-------')    
        #-------

        #
        # create a from-scratch initialized minGPT model
        gpt_config = GPTConfig(**config_args) # the values from config can be passed in any order

        print(f'config: {gpt_config}')

        #-------------------
        # create our own model and its state dictionary
        model = GPT(gpt_config)

        # we are going to set the values to state_dict, but since this is a reference and any updates here will
        #   reflect at model
        state_dict = model.state_dict() # from nn.Module        
        state_dict_keys = state_dict.keys()

     

        print('-------')  
        print('state_dict_keys')
        for i in state_dict_keys:
            print(f"    k: {i}")
        print('-------')


        # filter out '.attn.bias' from state_dict_keys
        state_dict_keys = [
            k 
            for k in state_dict_keys 
            if not k.endswith('.attn.bias')
        ] # discard this mask / buffer, not a param


        #-----------
        removed_items = sorted(
                list( 
                set( state_dict.keys() )
                .symmetric_difference( 
                    set(state_dict_keys) 
                ) 
            )
        )
        print('-------')  
        print('removed_items')
        for i in removed_items:
            print(f"    i: {i}")
        print('-------')
        #-----------
        #-------------------

        #-------------------
        # Let's grab the model from HuggingFace and then filter out what we don't want
        # init a huggingface/transformers model
        model_hugginface = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict_huggingface = model_hugginface.state_dict()


        # copy while ensuring all of the parameters are aligned and match in names and shapes
        state_dict_keys_huggingface = state_dict_huggingface.keys()

        #-------
        # ignore ".attn.masked_bias" it's just a buffer, and ".attn.bias" is a mask (buffer)
        state_dict_keys_huggingface = [
            k 
            for k in state_dict_keys_huggingface 
            if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')
        ]
        #-------

        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        #-------------------

        #-------------------
        # copy phase
        # at this point the dict state from hugging face and our dict state should be identical
        assert len(state_dict_keys_huggingface) == len(state_dict_keys), f"mismatched keys: {len(state_dict_keys_huggingface)} != {len(state_dict_keys)}"


        for k in state_dict_keys_huggingface:

            if any( k.endswith(w) for w in transposed ): # let's transpose what needs to transposed
                # special treatment for the Conv1D weights we need to transpose
                #   e.g.: state_dict_huggingface[k].shape ---> torch.Size([768, 2304])
                #     state_dict_huggingface[k].shape[::-1] -> torch.Size([2304, 768])
                #     state_dict[k].shape -------------------> torch.Size([2304, 768])
                assert state_dict_huggingface[k].shape[::-1] == state_dict[k].shape

                with torch.no_grad():
                    transposed_component = state_dict_huggingface[k].t()
                    state_dict[k].copy_( transposed_component )
            else:
                # vanilla copy over the other parameters
                assert state_dict_huggingface[k].shape == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_huggingface[k])

            #----
            

        #-------------------







        return model
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------


import tiktoken
#-------------------------------------------------------------------------
class DataLoaderLite:
    #-------------------------------------------------------------------------
    def __init__(self, B, T):
        # B - batch size, T - sequence length
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        # **** DELETE THIS CHANGE ***
        # file_path = './models/10_nano_gpt/current/'
        # with open(f'{file_path}input.txt', 'r') as f:
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
        '''
        We are going to read and use sequentially everything in the file. we will use the curr pos
        to keep track where we left. In the line below we are reading from tokens, starting from where
        we left off last, and then adding the range of B(batch size) + T(sequence len) + 1 as it's non-inclusive
        '''
        buf = self.tokens[self.current_position : self.current_position+B*T+1]

        # 1 token offset. Then reshape it to [B, T] as 'buf' is a 1 dim tensor
        x = (buf[:-1]).view(B, T) # inputs - from idx 0 to the end minus the last token
        y = (buf[1:]).view(B, T) # targets - from the idx 1 to the last token

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
    elif torch.backends.mps.is_built():
        device = 'mps'
        print('using mps acceleration')
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
# train_loader = DataLoaderLite(B=4, T=32)

'''
The 'high' precision setting allows float32 matrix multiplications to use TensorFloat32, which 
has 10 mantissa bits explicitly stored, or to treat each float32 number as the sum of two 
bfloat16 numbers, which provides approximately 16 mantissa bits with 14 bits explicitly stored. 
If the appropriate fast algorithms are not available, the computations fall back to using the 
'highest' precision setting, which employs the full float32 datatype with 24 mantissa bits.

This configuration is particularly beneficial for CUDA devices, where TensorFloat32 operations 
can be significantly faster than traditional float32 operations. The 'high' precision setting 
leverages these faster operations to improve performance without a substantial loss in precision, 
making it a useful option for many machine learning and scientific computing applications.
'''
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# compile - this greatly increase the performance
model = torch.compile(model)

import time
#-------------------------------------------------------------------------
# optimize!

# using hyperparameters from GPT3 - literally following what they published in their paper.
#  this routine uses similar values for the parameters, but not quite the same
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

print(f'train_loader.B * train_loader.T: {train_loader.B * train_loader.T}')
for i in range(50):
    t0 = time.time()

    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    '''
    Cast all operatoins within this block (and device, typically a CUDA enabled device) to bfloat16.
    bfloat16 is particularly desiable because it offers a good balance between precision and large
    improvements in performance (for CUDA devices).
    Note: For your specific CUDA card, check the estimated bfloat16 operations and possible other floating
    points types, as NVIDIA often updates their types and performance estimates.
    '''
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # forward
        logits, loss = model(x, y)
    loss.backward()

    '''
    using hyperparameters from GPT3 - literally following what they published in their paper.
     this routine uses similar values for the parameters, but not quite the same    
    Calculate the global norm of the parameters - every gradient from the parameters are squared,
      sum all up, then take the square root 
    In the example below we define that the max_norm is no bigger than 1.0
    One of the reasons to use this, is in the case of bad data batches we would get a high loss, 
      which leads to a high gradient. This is a case of 'model shock'.
    Ideally we want the learning process to be more on the smooth side and not too peaky which can
     lead to instabilities and low convergence
    '''
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize() # wait for the GPU to finish work

    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f"step {i:4d} | loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
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
        logits = model(x) # (B, T, vocab_size) - [5, 8, 50257]
        
        # take the logits at the last position. Consider that the original logit shape is [5, 8, 50257]
        #  if you take only the last position from the second dim, as in logits[:, -1, :], what you are
        #  effectively doing is [5, 50257] 
        logits_last = logits[:, -1, :] # (B, vocab_size) - [5, 50257] 

        # get the probabilities
        probs = F.softmax(input = logits_last, dim=-1)

        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(input = probs, k = 50, dim=-1)

        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(input = topk_probs, num_samples = 1) # (B, 1)

        # gather the corresponding indices
        xcol = torch.gather(input = topk_indices, dim = -1, index = ix) # (B, 1)

        # append to the sequence
        x = torch.cat((x, xcol), dim=1)
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# print the generated text
for i in range(num_return_sequences): # num_return_sequences = 5
    tokens = x[i, :max_length].tolist() # max_length - 30
    decoded = enc.decode(tokens)
    print(">", decoded)
#-------------------------------------------------------------------------
