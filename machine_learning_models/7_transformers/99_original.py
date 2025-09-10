import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

#-------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):

    """
    the MultiHeadAttention class encapsulates the multi-head attention mechanism commonly used in transformer models. 
    It takes care of splitting the input into multiple attention heads, applying attention to each head, and then combining 
    the results. By doing so, the model can capture various relationships in the input data at different scales, improving 
    the expressive ability of the model 
    """

    #-------------------------------------------------------------------------
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model    = d_model              # Model's dimension
        self.num_heads  = num_heads            # Number of attention heads to split the input into
        self.dim_key    = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.weight_query   = nn.Linear(d_model, d_model) # Query transformation
        self.weight_key     = nn.Linear(d_model, d_model) # Key transformation
        self.weight_value   = nn.Linear(d_model, d_model) # Value transformation
        self.weight_output  = nn.Linear(d_model, d_model) # Output transformation
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # Calculate attention scores - the attention scores are calculated by taking the dot product of queries and keys
        #   and then scaling by the square root of the key dimension
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_key)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        
        attn_probs = torch.softmax(attn_scores, dim=-1) # Softmax is applied to obtain attention probabilities
        
        
        output = torch.matmul(attn_probs, value) # Multiply by values to obtain the final output
        return output
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------  
    def split_heads(self, x):
        # Reshape the input x into the shape (batch_size, num_heads, seq_length, dim_key). It enables the model to 
        #   process multiple attention heads concurrently, allowing for parallel computation

        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.dim_key).transpose(1, 2)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------    
    def combine_heads(self, x):
        # After applying attention to each head separately, this method combines the results back into a single tensor 
        #   of shape (batch_size, seq_length, d_model). This prepares the result for further processing
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    #-------------------------------------------------------------------------
    #------------------------------------------------------------------------- 
    def forward(self, query, key, value, mask=None):
        # 1 - Apply Linear Transformations: The queries, keys, and values are first passed through linear transformations 
        #   using the weights defined in the initialization
        # 2 - Split Heads: The transformed query, key, value are split into multiple heads using the split_heads method
        query   = self.split_heads( self.weight_query(query) )
        key     = self.split_heads( self.weight_key(key)     )
        value   = self.split_heads( self.weight_value(value) )
        
        # Apply Scaled Dot-Product Attention. The scaled_dot_product_attention method is called on the split heads
        attn_output = self.scaled_dot_product_attention(query, key, value, mask)
        
        # Combine Heads - The results from each head are combined back into a single tensor using the combine_heads method
        output = self.weight_output(self.combine_heads(attn_output))
        return output
    #-------------------------------------------------------------------------
#------------------------------------------------------------------------- 
#-------------------------------------------------------------------------
class PositionWiseFeedForward(nn.Module):
    """
    the PositionWiseFeedForward class defines a position-wise feed-forward neural network that consists of two linear 
    layers with a ReLU activation function in between. In the context of transformer models, this feed-forward network 
    is applied to each position separately and identically. It helps in transforming the features learned by the 
    attention mechanisms within the transformer, acting as an additional processing step for the attention outputs
    """
    #-------------------------------------------------------------------------
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        # notes: 
        # d_model: Dimensionality of the model's input and output
        # d_ff: Dimensionality of the inner layer in the feed-forward network
        
        # self.fc1 and self.fc2 - two fully connected (linear) layers with input and output dimensions as defined by d_model and d_ff
        self.fc1  = nn.Linear( d_model, d_ff )
        self.fc2  = nn.Linear( d_ff, d_model )
        self.relu = nn.ReLU() # activation function, which introduces non-linearity between the two linear layers
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        # x - input to the feed-forward network

        # self.fc1(x): The input is first passed through the first linear layer (fc1).
        # self.relu(...): The output of fc1 is then passed through a ReLU activation function. ReLU replaces 
        #   all negative values with zeros, introducing non-linearity into the model.
        # self.fc2(...): The activated output is then passed through the second linear layer (fc2), producing 
        #   the final output.        
        return self.fc2(self.relu(self.fc1(x)))
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    PositionalEncoding class adds information about the position of tokens within the sequence. Since the 
    transformer model lacks inherent knowledge of the order of tokens (due to its self-attention mechanism), 
    this class helps the model to consider the position of tokens in the sequence. The sinusoidal functions 
    used are chosen to allow the model to easily learn to attend to relative positions, as they produce a 
    unique and smooth encoding for each position in the sequence. The class is implemented as a nn.Module, 
    allowing it to be used as a layer within the model. The forward method adds the positional encodings to 
    the input tensor x, which is expected to have shape (batch_size, seq_length, d_model).    
    """
    #-------------------------------------------------------------------------
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # d_model: The dimension of the model's input
        # max_seq_length: The maximum length of the sequence for which positional encodings are pre-computed
        
        pe = torch.zeros(max_seq_length, d_model) # tensor filled with zeros, which will be populated with positional encodings
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # tensor containing the position indices for each position in the sequence
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # term used to scale the position indices in a specific way
        
        # the sine function is applied to the even indices and the cosine function to the odd indices of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe is registered as a buffer, which means it will be part of the module's state but will not be considered a trainable parameter
        self.register_buffer('pe', pe.unsqueeze(0))
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, x):
        # The forward method simply adds the positional encodings to the input x. It uses the first x.size(1) 
        #   elements of pe to ensure that the positional encodings match the actual sequence length of x
        return x + self.pe[:, :x.size(1)]
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """
    The EncoderLayer class defines a single layer of the transformer's encoder. It encapsulates a multi-head 
    self-attention mechanism followed by position-wise feed-forward neural network, with residual connections, 
    layer normalization, and dropout applied as appropriate. These components together allow the encoder to 
    capture complex relationships in the input data and transform them into a useful representation for 
    downstream tasks. Typically, multiple such encoder layers are stacked to form the complete encoder part 
    of a transformer model    
    """
    #-------------------------------------------------------------------------
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        # d_model: The dimensionality of the input.
        # num_heads: The number of attention heads in the multi-head attention.
        # d_ff: The dimensionality of the inner layer in the position-wise feed-forward network.
        # dropout: The dropout rate used for regularization

        self.self_attn    = MultiHeadAttention(d_model, num_heads) # Multi-head attention mechanism
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff) # Position-wise feed-forward neural network
        self.norm1   = nn.LayerNorm(d_model) # self.norm1 and self.norm2: Layer normalization, applied to smooth the layer's input
        self.norm2   = nn.LayerNorm(d_model) # see above
        self.dropout = nn.Dropout(dropout)   # Dropout layer, used to prevent overfitting by randomly setting some activations to zero during training
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------        
    def forward(self, x, mask):

        # x: The input to the encoder layer.
        # mask: Optional mask to ignore certain parts of the input.

        attn_output = self.self_attn( x, x, x, mask )             # Self-Attention: The input x is passed through the multi-head self-attention mechanism
        x           = self.norm1( x + self.dropout(attn_output) ) # Add & Normalize (after Attention): The attention output is added to the original input (residual connection), followed by dropout and normalization using norm1
        ff_output   = self.feed_forward( x )                      # Feed-Forward Network: The output from the previous step is passed through the position-wise feed-forward network
        x           = self.norm2( x + self.dropout(ff_output) )   # Add & Normalize (after Feed-Forward): Similar to step 2, the feed-forward output is added to the input of this stage (residual connection), followed by dropout and normalization using norm2

        return x # Output: The processed tensor is returned as the output of the encoder layer
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """
    The DecoderLayer class defines a single layer of the transformer's decoder. It consists of a multi-head 
    self-attention mechanism, a multi-head cross-attention mechanism (that attends to the encoder's output), 
    a position-wise feed-forward neural network, and the corresponding residual connections, layer 
    normalization, and dropout layers. This combination enables the decoder to generate meaningful outputs 
    based on the encoder's representations, taking into account both the target sequence and the source 
    sequence. As with the encoder, multiple decoder layers are typically stacked to form the complete 
    decoder part of a transformer model.    
    """
    #-------------------------------------------------------------------------
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        # d_model: The dimensionality of the input.
        # num_heads: The number of attention heads in the multi-head attention.
        # d_ff: The dimensionality of the inner layer in the feed-forward network.
        # dropout: The dropout rate for regularization.

        self.self_attn    = MultiHeadAttention(d_model, num_heads) # Multi-head self-attention mechanism for the target sequence
        self.cross_attn   = MultiHeadAttention(d_model, num_heads) # Multi-head attention mechanism that attends to the encoder's output
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff) # Position-wise feed-forward neural network
        self.norm1        = nn.LayerNorm(d_model)                  # Layer normalization components
        self.norm2        = nn.LayerNorm(d_model)                  # same as above
        self.norm3        = nn.LayerNorm(d_model)                  # same as above
        self.dropout      = nn.Dropout(dropout)                    # Dropout layer for regularization
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        x: The input to the decoder layer.
        enc_output: The output from the corresponding encoder (used in the cross-attention step).
        src_mask: Source mask to ignore certain parts of the encoder's output.
        tgt_mask: Target mask to ignore certain parts of the decoder's input
        """
        attn_output = self.self_attn( x, x, x, tgt_mask )                    # Self-Attention on Target Sequence: The input x is processed through a self-attention mechanism.
        x           = self.norm1( x + self.dropout(attn_output) )            # Add & Normalize (after Self-Attention): The output from self-attention is added to the original x, followed by dropout and normalization using norm1.
        attn_output = self.cross_attn( x, enc_output, enc_output, src_mask ) # Cross-Attention with Encoder Output: The normalized output from the previous step is processed through a cross-attention mechanism that attends to the encoder's output enc_output.
        x           = self.norm2( x + self.dropout(attn_output) )            # Add & Normalize (after Cross-Attention): The output from cross-attention is added to the input of this stage, followed by dropout and normalization using norm2.
        ff_output   = self.feed_forward( x)                                  # Feed-Forward Network: The output from the previous step is passed through the feed-forward network.
        x           = self.norm3( x + self.dropout(ff_output) )              # Add & Normalize (after Feed-Forward): The feed-forward output is added to the input of this stage, followed by dropout and normalization using norm3.

        return x # Output: The processed tensor is returned as the output of the decoder layer.
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------
class Transformer(nn.Module):
    """
    The Transformer class brings together the various components of a Transformer model, including the 
      embeddings, positional encoding, encoder layers, and decoder layers. It provides a convenient 
      interface for training and inference, encapsulating the complexities of multi-head attention, 
      feed-forward networks, and layer normalization.
    This implementation follows the standard Transformer architecture, making it suitable for 
      sequence-to-sequence tasks like machine translation, text summarization, etc. The inclusion of 
      masking ensures that the model adheres to the causal dependencies within sequences, ignoring 
      padding tokens and preventing information leakage from future tokens.
    These sequential steps empower the Transformer model to efficiently process input sequences and 
      produce corresponding output sequences    
    """
    #-------------------------------------------------------------------------
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        """
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: The dimensionality of the model's embeddings.
        num_heads: Number of attention heads in the multi-head attention mechanism.
        num_layers: Number of layers for both the encoder and the decoder.
        d_ff: Dimensionality of the inner layer in the feed-forward network.
        max_seq_length: Maximum sequence length for positional encoding.
        dropout: Dropout rate for regularization.        
        """

        self.encoder_embedding   = nn.Embedding( src_vocab_size, d_model )        # Embedding layer for the source sequence.
        self.decoder_embedding   = nn.Embedding( tgt_vocab_size, d_model )        # Embedding layer for the target sequence
        self.positional_encoding = PositionalEncoding( d_model, max_seq_length )  # Positional encoding component

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) # A list of encoder layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) # A list of decoder layers

        self.fc      = nn.Linear( d_model, tgt_vocab_size ) # Final fully connected (linear) layer mapping to target vocabulary size
        self.dropout = nn.Dropout( dropout ) # Dropout layer
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def generate_mask(self, src, tgt):
        """
        This method is used to create masks for the source and target sequences, ensuring that padding 
        tokens are ignored and that future tokens are not visible during training for the target sequence
        """
        src_mask    = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask    = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length  = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask    = tgt_mask & nopeak_mask

        return src_mask, tgt_mask
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def forward(self, src, tgt):
        """
        Input Embedding and Positional Encoding: The source and target sequences are first embedded using 
          their respective embedding layers and then added to their positional encodings.
        Encoder Layers: The source sequence is passed through the encoder layers, with the final encoder 
          output representing the processed source sequence.
        Decoder Layers: The target sequence and the encoder's output are passed through the decoder layers, 
          resulting in the decoder's output.
        Final Linear Layer: The decoder's output is mapped to the target vocabulary size using a fully 
          connected (linear) layer.        
        """
        src_mask, tgt_mask  = self.generate_mask(src, tgt)
        src_embedded        = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded        = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output # output is a tensor representing the model's predictions for the target sequence
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------   
#-------------------------------------------------------------------------   
def sample_data_preparation():
    """
    Initialize a transformer model and generate random source and target sequences that can be fed 
      into the model. The chosen hyperparameters determine the specific structure and properties 
      of the transformer. This setup could be part of a larger script where the model is trained 
      and evaluated on actual sequence-to-sequence tasks, such as machine translation or text 
      summarization    
    """
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512 # Dimensionality of the model's embeddings, set to 512
    num_heads = 8
    num_layers = 6 # Number of layers for both the encoder and the decoder
    d_ff = 2048 # Dimension of the inner layer in the feed-forward network
    max_seq_length = 100
    dropout = 0.1

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

    # Generate random sample data
    #  Random integers between 1 and tgt_vocab_size, representing a batch of target sequences with shape (64, max_seq_length)
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    return {
        'transformer'       : transformer,
        'src_data'          : src_data,
        'tgt_data'          : tgt_data,
        'tgt_vocab_size'    : tgt_vocab_size,
        'src_vocab_size'    : src_vocab_size,
        'max_seq_length'    : max_seq_length
    }
#-------------------------------------------------------------------------  
#-------------------------------------------------------------------------  
def execute_part1():
    result_sample_data_preparation =  sample_data_preparation()
    return {
        'transformer'   : result_sample_data_preparation['transformer'],
        'src_data'      : result_sample_data_preparation['src_data'],
        'tgt_data'      : result_sample_data_preparation['tgt_data'],
        'tgt_vocab_size': result_sample_data_preparation['tgt_vocab_size'],
        'src_vocab_size': result_sample_data_preparation['src_vocab_size'],
        'max_seq_length': result_sample_data_preparation['max_seq_length']
    }
#-------------------------------------------------------------------------  
#-------------------------------------------------------------------------  
def training(transformer, src_data, tgt_data, tgt_vocab_size):
    """
    This code snippet trains the transformer model on randomly generated source and target sequences 
    for 100 epochs. It uses the Adam optimizer and the cross-entropy loss function. The loss is 
    printed for each epoch, allowing you to monitor the training progress. In a real-world scenario, 
    you would replace the random source and target sequences with actual data from your task, such 
    as machine translation
    """
    # Defines the loss function as cross-entropy loss. The ignore_index argument is set to 0, meaning 
    #   the loss will not consider targets with an index of 0 (typically reserved for padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Defines the optimizer as Adam with a learning rate of 0.0001 and specific beta values
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Sets the transformer model to training mode, enabling behaviors like dropout that only apply during training
    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad() # Clears the gradients from the previous iteration

        # Passes the source data and the target data (excluding the last token in each sequence) through the 
        #   transformer. This is common in sequence-to-sequence tasks where the target is shifted by one token
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    return {
        'criterion': criterion
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part2(transformer, src_data, tgt_data, tgt_vocab_size):
    result_training = training(transformer=transformer, src_data = src_data, tgt_data = tgt_data, tgt_vocab_size = tgt_vocab_size)
    return {
        'criterion': result_training['criterion']
    }
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def model_performance_evaluation(transformer, src_vocab_size, tgt_vocab_size, max_seq_length, criterion):
    """
    This code snippet evaluates the transformer model on a randomly generated validation dataset, 
    computes the validation loss, and prints it. In a real-world scenario, the random validation 
    data should be replaced with actual validation data from the task you are working on. The 
    validation loss can give you an indication of how well your model is performing on unseen 
    data, which is a critical measure of the model's generalization ability
    """


    # Puts the transformer model in evaluation mode. This is important because it turns off certain 
    # behaviors like dropout that are only used during training
    transformer.eval()

    # Generate random sample validation data
    # Random integers between 1 and src_vocab_size, representing a batch of validation source sequences 
    #   with shape (64, max_seq_length)
    val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    # Random integers between 1 and tgt_vocab_size, representing a batch of validation target sequences 
    #   with shape (64, max_seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    # Disables gradient computation, as we don't need to compute gradients during validation. This can reduce memory 
    #   consumption and speed up computations.
    with torch.no_grad():

        # Passes the validation source data and the validation target data (excluding the last token in each sequence) 
        #   through the transformer
        val_output = transformer(val_src_data, val_tgt_data[:, :-1])

        # Computes the loss between the model's predictions and the validation target data (excluding the first token 
        #   in each sequence). The loss is calculated by reshaping the data into one-dimensional tensors and using 
        #   the previously defined cross-entropy loss function
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {val_loss.item()}")
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
def execute_part3(transformer, src_vocab_size, tgt_vocab_size, max_seq_length, criterion):
    model_performance_evaluation(
        transformer=transformer,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_seq_length=max_seq_length,
        criterion=criterion
    )
#-------------------------------------------------------------------------    



result_part1 = execute_part1()
result_part2 = execute_part2(
    transformer     = result_part1['transformer'], 
    src_data        = result_part1['src_data'], 
    tgt_data        = result_part1['tgt_data'], 
    tgt_vocab_size  = result_part1['tgt_vocab_size']
)
execute_part3(
    transformer     = result_part1['transformer'], 
    src_vocab_size  = result_part1['src_vocab_size'], 
    tgt_vocab_size  = result_part1['tgt_vocab_size'], 
    max_seq_length  = result_part1['max_seq_length'],
    criterion       = result_part2['criterion']
)