import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
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
    def configure_optimizers(self, weight_decay, learning_rate, device_type):

        #-----------------
        # start with all of the candidate parameters (that require grad)
        '''
        these are params like:
            transformer.wte.weight torch.Size([50304, 768])
            transformer.wpe.weight torch.Size([1024, 768])
            transformer.h.0.ln_1.weight torch.Size([768])
            transformer.h.0.ln_1.bias torch.Size([768])
            transformer.h.0.attn.c_attn.weight torch.Size([2304, 768])
        '''
        param_dict = { param_name: param 
                      for param_name, param in self.named_parameters()
                      if param.requires_grad 
        }
        
        # param_dict = {param_name: param 
        #               for param_name, param in param_dict.items()
        #               if param.requires_grad
        # }


        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [ param 
                        for param_name, param in param_dict.items() 
                        if param.dim() >= 2
        ]

        nodecay_params = [param 
                          for param_name, param in param_dict.items() 
                          if param.dim() < 2
        ]

        optim_groups = [
            {
                'params': decay_params, 
                'weight_decay': weight_decay
            },
            {
                'params': nodecay_params, 
                'weight_decay': 0.0
            }
        ]
        #-----------------
        num_decay_params = sum( p.numel() for p in decay_params )
        num_nodecay_params = sum( p.numel() for p in nodecay_params )
        if master_process: # we only want to print to console if this is the master process

            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        #-----------------

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters

        use_fused = fused_available and device_type == "cuda"
        if master_process: # we only want to print to console if this is the master process
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------


import tiktoken
import numpy as np
#-------------------------------------------------------------------------
# remember that at this point the dataset was processed by an independent script that
#   has downloaded the data, tokenized it, and stored as a large shard of numpy tensors
def load_tokens(filename):
    numpy_tensor = np.load(filename)
    numpy_tensor = numpy_tensor.astype(np.int32)
    pytorch_tensor = torch.tensor(numpy_tensor, dtype=torch.long)
    return pytorch_tensor
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class DataLoaderLite:
    #-------------------------------------------------------------------------
    # process_rank=ddp_rank -> unique identifier for each process across all nodes and GPUs
    # num_processes=ddp_world_size -> total number of processes participating in the distributed training.
    #   If you have 2 nodes, each with 4 GPUs, the world size will always be 8.
    def __init__(self, B, T, process_rank, num_processes, split):
        # B - batch size, T - sequence length
        self.B = B
        self.T = T
        self.process_rank = process_rank   # ddp_rank
        self.num_processes = num_processes # world size

        #--------

        # at init load tokens from disk and store them in memory
        # **** DELETE THIS CHANGE ***
        # file_path = './models/10_nano_gpt/current/'
        # with open(f'{file_path}input.txt', 'r') as f:
        #with open('input.txt', 'r') as f:
        #    text = f.read()	
	
	
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B" 
        shards = os.listdir(data_root) # ['edufineweb_train_000008.npy', 'edufineweb_train_000020.npy' ...

        shards = [ filename 
                  for filename in shards 
                  if split in filename # does the filename contains 'train' or 'val'
        ]

        shards = sorted(shards) # sort the shards. important in the case of documents being split between shards

        shards = [ os.path.join(data_root, filename)  # ['edu_fineweb10B/edufineweb_train_000001.npy', 'edu_fineweb10B/edufineweb_train_000002.npy', ...
                  for filename in shards ]


        self.shards = shards
	
        assert len(shards) > 0, f"no shards found for split {split}"
	
	
        if master_process:  # we only want to print to console if this is the master process
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def reset(self):

        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(filename = self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
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
        #   note that the 'pagination' has to jump by an amount equivalent to the 'world size' 
        self.current_position += B * T * self.num_processes # world size

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # originally this was 0, but with DDP this is its zero. You can double check at the constructor

            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(filename = self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank

        return x, y

    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------


# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss
#-------------------------------------------------------------------------
def get_most_likely_row(tokens, mask, logits):
    # tokens -> the list of all sentences options. The len is the len of the longest sentence, the excess is padded with zeroes
    # mask -> list of ones and zeroes, where the beginning is filled with 0s representing the len of ctx_tokens and the end filled with 1s representing the len of end_tokens
    # logits -> model's prediction (from 4 options)

    # the idea in this function is to statistically compare the options with what was genereted from
    #   logits. The option that most closely matches the answer from logits is the one selected.

    '''
    print(f'******** at get_most_likely_row')
    print(f'tokens shape: {tokens.shape}')
    print(f'mask shape: {mask.shape}')
    print(f'logits shape: {logits.shape}')
    print('-------')
      Results: 
        tokens shape: torch.Size([4, 20])
        mask shape:   torch.Size([4, 20])
        logits shape: torch.Size([4, 20, 50304])

    And the next probing commands and their results:
        print(f'logits:\n {            logits[-1, -2, : ]}')  # [ 0.9023, -0.2373, -0.2871,  ...,  0.1504,  0.5273,  0.6250]
        print(f'shift_logits:\n {shift_logits[-1, -1 , : ]}') # [ 0.9023, -0.2373, -0.2871,  ...,  0.1504,  0.5273,  0.6250]

    As we can see shift_logits is just skipping the last token prediction made on logits.
    And shift_tokens is skipping the first token.

    Remember that the result of the model is 1 new token prediction from what was provided
     so what we should do (and we do) is to clip the last token from logits, now logits
     will have 19 tokens, but it will be missing the first token. So we clip it from the
     tokens list, so they align. 
    '''

    # evaluate the autoregressive loss at all positions
    shift_logits = ( logits[ ... ,   : -1 , : ] ).contiguous() # [4, 19, 50304]
    shift_tokens = ( tokens[ ... , 1 :        ] ).contiguous() # [4, 19]
    #  alignment ------------------^----^


    # the line below uses -1 as to infer/figure out the dimensions, and we give only one
    #   fixed dimension, shift_logits.size(-1) which is 50304, so final shape [ 4 * 19,  50304]
    #   [76, 50304]
    flat_shift_logits = shift_logits.view( -1, shift_logits.size(-1) ) # from 3d to 2d tensor [76, 50304]
    flat_shift_tokens = shift_tokens.view( -1 ) # from 2d to 1d tensor [ 76 ]


    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # [76]

    # create a view, make it [4 , -1] -> [4, 19] 
    shift_losses = shift_losses.view(tokens.size(0), -1)


    
    # now get the average loss just for the completion region (where mask == 1), in each row
    #  the mask shiftting is the same idea as from the tokens, we skip the first token to align with model's
    # [4, 19]
    shift_mask = ( mask[ ... , 1: ] ).contiguous() # we must shift mask, so we start at the last prompt token
    
    # e.g.: [1, 2, 3, 4, 5, 6] * [0,0,0, 1, 1, 1] = [1*0, 2*0, 3*0, 4*1, 5*1, 6*1] = [0, 0, 0, 4, 5, 6]
    masked_shift_losses = shift_losses * shift_mask # [4, 19]
    
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)   # [4]
    avg_loss = sum_loss / shift_mask.sum(dim=1) # [4]

    # masked_shift_losses shape: torch.Size([4, 19])
    # sum_loss shape: torch.Size([4])
    # avg_loss shape: torch.Size([4])



   
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    #   the data should be something like this:
    #   avg_loss: tensor([10.8125, 11.0000, 10.8750, 10.8750]
    #   pred_norm: 0 (idx)
    pred_norm = avg_loss.argmin().item()


    return pred_norm
#-------------------------------------------------------------------------

# ***************************************
# DDP simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# ***************************************

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

# using torch run, the application would return values ranging from 0 to N depending
#  of which GPU number/ID this script is being executed.
ddp = int(os.environ.get(key = 'RANK', default = -1)) != -1 # is this a ddp run?

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"

    # backend='nccl' argument specifies that the NVIDIA Collective Communications Library (NCCL) 
    #   will be used as the communication backend. NCCL is optimized for GPU-to-GPU communication 
    #   and is particularly well-suited for distributed training on NVIDIA GPUs
    init_process_group(backend='nccl') # from torch.distributed

    '''
    RANK - global rank of the current process in the distributed training setup.
      It uniquely identifies each process across all nodes and GPUs in the distributed 
      system.
      If you have 2 nodes, each with 4 GPUs, there will be 8 processes in total. 
      The global ranks will range from 0 to 7.

    LOCAL_RANK - local rank of the current process on a specific node.
      It identifies which GPU (or device) the process is assigned to on the local 
      machine (node).
      On a node with 4 GPUs, the local ranks will range from 0 to 3.

    WORLD_SIZE - total number of processes participating in the distributed training.
      It defines the size of the distributed system and is used for communication and 
      synchronization between processes.
      If you have 2 nodes, each with 4 GPUs, the world size will be 8.
      Note that WORLD_SIZE is a constant across all processes. If this is running
      on NODE 2 GPU 3, WORLD_SIZE would still be 8.
    '''
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])



    device = f'cuda:{ddp_local_rank}'

    torch.cuda.set_device(device)

    master_process = ddp_rank == 0 # IMPORTANT - this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True # IMPORTANT

print('--------------------------')
print(f'ddp_rank: {ddp_rank}')
print(f'ddp_local_rank: {ddp_local_rank}')
print(f'ddp_world_size: {ddp_world_size}')
print('--------------------------')
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
import time

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
enc = tiktoken.get_encoding("gpt2")

#------------------------------
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
# B = 64 # micro batch size
B = 16 # micro batch size
T = 1024 # sequence length
# B = 4
# T = 32 # sequence length


# ddp_world_size is 1 on a single system. If you have 2 nodes, each with 4 GPUs, the world 
#   size will always be 8.
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"


grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process: # we only want to print to console if this is the master process
    print('\n\n-------------------------')

    print(f'B: {B}\t T: {T}\tddp_world_size: {ddp_world_size}')
    print(f'total_batch_size: {total_batch_size}\t(B * T): {(B * T)}\t(B * T * ddp_world_size):{(B * T * ddp_world_size)}')
    print(f'grad_accum_steps = total_batch_size // (B * T * ddp_world_size): {grad_accum_steps}\n')


    # print(f"total desired batch size: {total_batch_size}")
    # print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    print('-------------------------\n\n')

train_loader      = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
validation_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
#------------------------------



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

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)

# compile - this greatly increase the performance
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if torch.cuda.is_available() and use_compile:
    model = torch.compile(model)

#-----
# dpp check and model conversion
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
#-----

max_lr = 6e-4
min_lr = max_lr * 0.1 # 0.000059999999999999995
# warmup_steps = 715
# max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
warmup_steps = 10
max_steps = 50

max_steps_minus_warmup_steps = max_steps - warmup_steps
max_lr_minus_min_lr = max_lr - min_lr

'''
With a LR scheduler we acknowledge that are different stages during the training process. Initially
  the model is in an almost useless state where it lears more about what tokens are not used. Then after
  it warms up we can make faster (macro) corrections with a larger LR, but once it has achieved a 
  certain level of stability we want to gradually fine tune its learning rate to something smaller
  so corrections are smoother (micro)
'''
#-------------------------------------------------------------------------
def get_lr(it:int): # it -> steps from the training process

    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps: # usually 10
        # 6e-4 * (0 + 1)  / 10  = 6e-4 * 1  / 10 = 6e-4    / 10 = 5.9999999999999995e-05
        # 6e-4 * (1 + 1)  / 10  = 6e-4 * 2  / 10 = 0.0012  / 10 = 0.00011999999999999999
        # ...
        # 6e-4 * (9 + 1)  / 10  = 6e-4 * 10 / 10 = 0.00599 / 10 = 0.0006
        return max_lr * (it+1) / warmup_steps
    
    # 2) if it > lr_decay_iters, return min learning rate
    if it >= max_steps:
        return min_lr # 0.000059999999999999995
    

    # 3) in between, use cosine decay down to min learning rate
    #   (10 - 10) / (50 - 10) = 0 / 40 = 0 
    #   (11 - 10) / (50 - 10) = 1 / 40 = 0.025
    #   ...  -> 0.05 , 0.075 , 0.1, 0.125 , 0.15, ... 0.95 , 0.975
    decay_ratio = (it - warmup_steps) / (max_steps_minus_warmup_steps)

    assert 0 <= decay_ratio <= 1

    # 0.5 * ( 1.0 + math.cos( math.pi * 0.0 ) ) = 0.5 * ( 1.0 + math.cos( 0 ) ) = 
    #     0.5 * ( 1.0 + 1.0 ) = 0.5 * 2 = 1
    # 0.5 * ( 1.0 + math.cos( math.pi * 0.025 ) ) = 0.5 * ( 1.0 + math.cos( 0.07853981633974483 ) ) = 
    #     0.5 * ( 1.0 + 0.996917333733128 ) = 0.5 *  1.996917333733128 = 0.998458666866564
    # 
    # so from the start the values are:
    #   1.0 , 0.998458666866564 , 0.9938441702975689, 0.9861849601988383, ... , 0.001541333133436018
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0

    # min_lr + coeff * (max_lr - min_lr) = 5.9999999999999995e-05 + coeff * ( 0.0005399999999999999 )
    # 5.9999999999999995e-05 + 1.0 * ( 0.0005399999999999999 )               = 0.0005999999999999998
    # 5.9999999999999995e-05 + 0.998458666866564 * ( 0.0005399999999999999 ) = 0.0005991676801079444
    # 
    # from the start the values are:
    #   0.0005999999999999998 , 0.0005991676801079444 , 0.0005966758519606872 , 0.0005925398785073725 , 
    #   ... , 0.00006083231989205545
    return min_lr + coeff * (max_lr_minus_min_lr)
#-------------------------------------------------------------------------



# optimize!

# using hyperparameters from GPT3 - literally following what they published in their paper.
#  this routine uses similar values for the parameters, but not quite the same
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

print(f'train_loader.B * train_loader.T: {train_loader.B * train_loader.T}')

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

#--------------------
with open(log_file, "w") as f: # open for writing to clear the file - not implemented yet
    pass
#--------------------


#-------------------------------------------------------------------------
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    #-------------------------------------------------------------------------
    # once in a while evaluate our validation loss - we don't do the backward pass here
    if step % 250 == 0 or last_step:

        # set model to eval, this changes the behavior of certain layers, layers like Dropout 
        #   and BatchNorm have distinct behaviors depending on the mode. Dropout is disabled 
        #   (no activations are dropped), and BatchNorm uses the running statistics computed 
        #   during training instead of recalculating them
        model.eval()
        
        # remember that this is the VALIDATION not the TRAINING loader. And that for edufineweb we
        #   only have 1 file as validation
        validation_loader.reset() 

        #-----------------
        with torch.no_grad(): # disable gradient computation
            val_loss_accum = 0.0
            val_loss_steps = 20

            for _ in range(val_loss_steps):
                
                x, y = validation_loader.next_batch()
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        #-----------------

        if ddp:
            ''' this is a distributed communication operation where multiple processes (often 
            running on different GPUs or machines) need to share and synchronize data. 
            specifically, this operation reduces the loss_accum tensor across all 
            participating processes by applying the specified reduction operation, in this 
            case, dist.ReduceOp.AVG (average). '''
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
	    
	#-------------------------------------------------------------------------
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
	    #-------------------------------------------------------------------------
            if step > 0 and (step % 500 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
	    #-------------------------------------------------------------------------
	#-------------------------------------------------------------------------

    #-------------------------------------------------------------------------
    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
	
	    #-------------------------------------
        ''' from hellaswag.py - this will yield one json line from hellaswag examples. In this case val
        from "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl" ''' 
        for i, example in enumerate(iterate_examples("val")): # this can be better implemented, we read all lines and just execute something on a few (considering a large pool of GPUs or nodes)

            ''' only process examples where i % ddp_world_size == ddp_rank. So take ddp_world_size = 8
            and ddp_rank = 3
            i = 0   ->   0 % 8 != 3   ->   0 != 3 = True
            i = 1   ->   1 % 8 != 3   ->   1 != 3 = True
            i = 2   ->   2 % 8 != 3   ->   2 != 3 = True
            i = 3   ->   3 % 8 != 3   ->   3 != 3 = False
            ...
            i = 7   ->   7 % 8 != 3   ->   7 != 3 = True
            i = 8   ->   8 % 8 != 3   ->   0 != 3 = True
            i = 9   ->   9 % 8 != 3   ->   1 != 3 = True
            '''
            if i % ddp_world_size != ddp_rank: # we read a line and throw it away here. improve this
                continue
		
            # render the example into tokens and labels
            # the returned data from render_example is:
            #   data -> the list of options encoded (list of 4 options) with a leading space
            #   tokens -> the list of all sentences options. The len is the len of the longest sentence, the excess is padded with zeroes
            #   mask -> list of ones and zeroes, where the beginning is filled with 0s representing the len of ctx_tokens and the end filled with 1s representing the len of end_tokens
            #   label -> the idx of the correct answer
            _, tokens, mask, label = render_example(example) # from hellaswag.py

            tokens = tokens.to(device)
            mask = mask.to(device)
	    
            #-------------
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens) # send the 4 sentences options

                # print(f'tokens shape:\n{tokens.shape}') # torch.Size([4, 20])
                # print(f'logits shape:\n{logits.shape}') # torch.Size([4, 20, 50304])

                pred_norm = get_most_likely_row(tokens, mask, logits) # idx of the most likely answer
            #-------------
		
            num_total += 1 # num of tests
            num_correct_norm += int(pred_norm == label) # if the predicted idx is the same as the answer add 1 to the score
	    #-------------------------------------
	    

        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device) # num of tests

            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)

            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)

            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total


        if master_process:
            print('\n\n')
            print('------------------')
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            print('------------------')
            print('\n')

            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")


    #-------------------------------------------------------------------------



    # once in a while generate from the model (except step 0, which is noise)
    #-------------------------------------------------------------------------
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()

        num_return_sequences = 4
        max_length = 32

        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        x_gen = tokens.to(device)

        sample_rng = torch.Generator(device=device) # creates a random number generator that is tied to a specific device
        sample_rng.manual_seed(42 + ddp_rank)
	
	
        #-------------------------------------------------------------------------
        while x_gen.size(1) < max_length:

            #--------------
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x_gen) # (B, T, vocab_size)

                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)

                # get the probabilities
                probs = F.softmax(logits, dim=-1)

                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)

                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

                # append to the sequence
                x_gen = torch.cat((x_gen, xcol), dim=1)
            #--------------
	    #-------------------------------------------------------------------------

	    #-------------------------------------------------------------------------		
        # print the generated text
        for i in range(num_return_sequences):
            tokens = x_gen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
	    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    
    
    # do one step of the optimization
    model.train()
    
    optimizer.zero_grad()
    loss_accum = 0.0
    #-------------------------------------
    for micro_step in range(grad_accum_steps): # note grad_accum_steps = total_batch_size // (B * T * ddp_world_size) = 524288 // (16 * 1024 * 1) = 32

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)


        '''
        Cast all operatoins within this block (and device, typically a CUDA enabled device) to bfloat16.
        bfloat16 is particularly desiable because it offers a good balance between precision and large
        improvements in performance (for CUDA devices).
        Note: For your specific CUDA card, check the estimated bfloat16 operations and possible other floating
        points types, as NVIDIA often updates their types and performance estimates.
        '''
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # forward
            logits, loss = model(x, y)

        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps

        '''
        detach() is used to create a tensor that shares the same data as the original 
        tensor but is detached from the computation graph. 
        prevents the loss_accum tensor from contributing to the gradient computation
        Gradient Accumulation: In scenarios where the total batch size is too large to 
        fit into memory (e.g., due to GPU memory constraints), the training process is 
        split into smaller "micro-batches." The losses from these micro-batches are 
        accumulated over several iterations before performing a single optimization 
        step. The variable loss_accum serves as a running total to accumulate the loss 
        values from each micro-batch.
        '''
        loss_accum += loss.detach()

        if ddp:
            '''
            micro_step can be, in the range of 0 to 31, and if we use the same example, grad_accum_steps = 32
              therefore at micro_step = 0 -> (micro_step == grad_accum_steps - 1) -> (0 ==  32 - 1) -> 0 == 31 -> False
              and micro_step = 31 -> (31 ==  32 - 1) -> (31 == 31) -> True

            model.require_backward_grad_sync is part of pytorch's DDP module. it is a 
              boolean flag that determines whether gradient synchronization across 
              processes should occur during the backward pass
            '''
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        loss.backward()
    #-------------------------------------

    
    if ddp:
        ''' this is a distributed communication operation where multiple processes (often 
          running on different GPUs or machines) need to share and synchronize data. 
          specifically, this operation reduces the loss_accum tensor across all 
          participating processes by applying the specified reduction operation, in this 
          case, dist.ReduceOp.AVG (average).
        '''
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

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
    
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:

        if 'lr' not in param_group:
            param_summary = f'*** NOTE - LR IS NOT PRESENT in param_group'
        else:
            param_summary = f"param_group['lr']: {param_group['lr']:.4f} - new LR: {lr:.4f}"

        param_group['lr'] = lr
    
    optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize() # wait for the GPU to finish work

    t1 = time.time()
    dt = t1 - t0 # time difference in seconds

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process: # we only want to print to console if this is the master process
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | {param_summary}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
#-------------------------------------------------------------------------
if ddp:
    destroy_process_group()
