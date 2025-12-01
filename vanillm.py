import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import shared

class VanillmModel(nn.Module):
    def __init__(self, block_size, vocab_size, embedding_dim, n_heads):

        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads

        self.token_embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        '''
        In the embedding layer, every integer token ID is used as an index to look up a corresponding vector of size embedding_dim from a vocab_size × embedding_dim matrix.
        For an input batch like [[1,2,3], [4,5,6]], the embedding layer returns a tensor of shape [2,3,embedding_dim], where each integer token ID has been replaced by its embedding vector.
        
        Input batch:

        [[1,2,3],
        [4,5,6]]

        Output:

        [
        [ E[1], E[2], E[3] ],
        [ E[4], E[5], E[6] ]
        ]

        Where each E[i] is a vector of 32 floats.

        '''

        self.positional_embedding_table = nn.Embedding(num_embeddings=self.block_size, embedding_dim=self.embedding_dim)

        '''
        This is another lookup table, but with shape [block_size x embedding_dim].
        Unlike the token_embedding_table that associates a vector to each token ID,
        the positional_embedding_table associates a vector to each POSITION in the sequence.
        '''

        self.transformer = Transformer(self.embedding_dim, self.block_size, self.n_heads).to(shared.device)

        self.final_linear_layer = nn.Linear(embedding_dim, vocab_size)


    def forward(self, idx):
        # idx is expected to be [B, T] (batch, time)
        B, T = idx.shape

        # ----- token + positional embeddings -----
        embedded_tensor = self.token_embedding_table(idx)  # [B, T, C]

        # 🔧 FIX 1: use actual T from input, not self.block_size
        positions = torch.arange(T, device=idx.device)     # [T]
        pos_tensor = self.positional_embedding_table(positions)  # [T, C]

        # broadcast [T, C] over batch → [B, T, C]
        pos_embedded_tensor = embedded_tensor + pos_tensor  # [B, T, C]

        # ----- transformer -----
        x = self.transformer(pos_embedded_tensor)          # [B, T, C]

        # ----- final linear layer → logits -----
        logits = self.final_linear_layer(x)                # [B, T, vocab_size]

        # 🔧 FIX 2: for training, return raw logits (no softmax here)
        return logits

        # if you ever want probabilities for inference ONLY:
        # probs = self.final_soft_max(logits)
        # return probs

class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_dim, block_size):

        super().__init__()

        # n_heads is the number of attention heads in Multi-Head Attention

        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.head_dim = head_dim

        # Three liner layers that take in input the vector of embedding_dim dimension for each token (embedded vector + position) and compute a vector of head_dim dimension
        # The weight matrix inside of each linear layer has a shape embedding_dim x head_dim

        self.linear_Q = nn.Linear(self.embedding_dim, self.head_dim, bias=False)
        self.linear_K = nn.Linear(self.embedding_dim, self.head_dim, bias=False)
        self.linear_V = nn.Linear(self.embedding_dim, self.head_dim, bias=False)

        # The casual mask is a lower-triangular matrix that prevents attention to future position, it will be used and explained in the forward pass
        # 🔧 FIX: create it once with max size (block_size x block_size), but DON'T hardcode device here
        causal_mask = torch.ones(self.block_size, self.block_size)
        causal_mask = causal_mask.tril()
        self.register_buffer("causal_mask", causal_mask)  # so it moves with .to(device)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        # x: [B, T, embedding_dim]
        B, T, C = x.shape

        Q = self.linear_Q(x)   # [B, T, head_dim]
        K = self.linear_K(x)   # [B, T, head_dim]
        V = self.linear_V(x)   # [B, T, head_dim]

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, T, T]

        # Scaling the scores for stabilizing the training
        attention_scores = attention_scores * 1/math.sqrt(self.head_dim)

        # 🔧 FIX: slice causal mask to current sequence length and ensure it's on the same device as x
        causal_mask = self.causal_mask[:T, :T].to(x.device)      # [T, T]

        # Put all the zero indexes of the causal mask of the attention score to a large negative number (this is for the softmax)
        attention_scores = attention_scores.masked_fill(causal_mask == 0, -1e10)

        attention_scores_probs = self.softmax(attention_scores)

        output = torch.matmul(attention_scores_probs, V)         # [B, T, head_dim]

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads, block_size):

        super().__init__()

        # n_heads is the number of attention heads in Multi-Head Attention

        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.n_heads = n_heads
        self.head_dim = embedding_dim//n_heads

        # Create n_heads SelfAttentionHead modules and store it in a ModuleList
        self.self_attention_heads = nn.ModuleList([SelfAttentionHead(self.embedding_dim, self.head_dim, self.block_size) for i in range(self.n_heads)])
        self.final_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        head_outputs = []
        for i in range(0, self.n_heads):
            head_outputs.append(self.self_attention_heads[i](x))

        # Concatenate all the outputs of the Self attention layers On the last dimension (the head dimension)
        multihead_output = torch.cat(head_outputs, dim=-1)

        # Pass the result through the final linear layer to unify all the results of the attention into a single unified transformation
        output = self.final_layer(multihead_output)

        return output

class FeedForward(nn.Module):
    
    def __init__(self, embedding_dim):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(self.embedding_dim, 4*self.embedding_dim),
            nn.ReLU(),
            nn.Linear(4*self.embedding_dim, embedding_dim)
        )

    def forward(self, x):
        output = self.net(x)
        return output
    
class Transformer(nn.Module):
    
    def __init__(self, embedding_dim, block_size, n_heads):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.n_heads = n_heads


        self.layer_norm_1 = nn.LayerNorm(self.embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(self.embedding_dim)
        self.multi_head_attention = MultiHeadAttention(self.embedding_dim, self.n_heads, self.block_size)
        self.feed_forward_network = FeedForward(self.embedding_dim)

    def forward(self, x):
        x_normed = self.layer_norm_1(x)
        attention_output = self.multi_head_attention(x_normed)
        x = x + attention_output
        x_normed = self.layer_norm_2(x)
        ff_output = self.feed_forward_network(x_normed)
        x = x + ff_output

        return x