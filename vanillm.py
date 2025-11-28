import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F

#This is my training data
poem_txt = 'Sempre caro mi fu quest’ermo colle, E questa siepe, che da tanta parte Dell’ultimo orizzonte il guardo esclude. Ma sedendo e mirando, interminati Spazi di là da quella, e sovrumani Silenzi, e profondissima quiete Io nel pensier mi fingo; ove per poco Il cor non si spaura. E come il vento Odo stormir tra queste piante, io quello Infinito silenzio a questa voce Vo comparando: e mi sovvien l’eterno, E le morte stagioni, e la presente E viva, e il suon di lei. Così tra questa Immensità s’annega il pensier mio: E il naufragar m’è dolce in questo mare.'

class Tokenizer:

    def __init__(self, training_txt):

        self.tokens_dict = self.create_tokens_dict(training_txt)


    def create_tokens_dict(self, training_txt):

        tokens_dict = {}
        token_id = 0

        for ch in training_txt:
            if ch not in tokens_dict.keys():
                tokens_dict[ch] = token_id
                token_id += 1
        
        return tokens_dict
    
    def txt_2_tokens(self, txt):
        tokens = []
        for ch in txt:
            tokens.append(self.tokens_dict[ch])
        return tokens

    def tokens_2_txt(self, tokens):
        txt = []
        for tk in tokens:
            for key in self.tokens_dict.keys():
                if self.tokens_dict[key] == tk:
                    txt.append(key)
        return txt
    
def list_2_batch(list, batch_size, block_size):

    batch_list = []
    tmp_batch = []
    i = 0 #index for block size
    j = 0 #index for batch size

    while i+block_size<len(list):
        tmp_batch.append(list[i:i+block_size])
        i+=1
        if len(tmp_batch) > batch_size -1:
            batch_list.append(tmp_batch)
            tmp_batch = []

    return batch_list


    
tokenizer = Tokenizer(poem_txt)
tokens = tokenizer.txt_2_tokens(poem_txt)
back_to_txt = tokenizer.tokens_2_txt(tokens)

# Define how many tokens the model can look at once
# So the prediction will be based only on these characters
block_size = 10
print("block size: " + str(block_size))


vocab_size = len(tokenizer.tokens_dict)
print("vocab size: " + str(vocab_size))

# Define the dimension of the embedding vector for each token
embedding_dim = 32
print("edmbedding dimension: " + str(embedding_dim))

batch_size = 4
print("batch size: " + str(batch_size))

n_heads = 4
print("number of attention heads: " + str(n_heads))


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

        self.transformer = Transformer(self.embedding_dim, self.block_size, self.n_heads).to('cuda')

        self.final_linear_layer = nn.Linear(embedding_dim, vocab_size)


    def forward(self, x):
        # Computes the embedding vector of each token
        # So the output will be a tensor of block_size elements, each of them is a tensor of embedding_dim elements
        embedded_tensor = self.token_embedding_table(x)
        pos = torch.arange(block_size, device='cuda')
        pos_tensor = self.positional_embedding_table(pos)

        pos_embedded_tensor = embedded_tensor + pos_tensor

        transformer_output = self.transformer(pos_embedded_tensor)

        output = self.final_linear_layer(transformer_output)

        return output



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
        self.causal_mask = torch.ones(block_size, block_size).to('cuda')
        self.causal_mask = self.causal_mask.tril()

        self.softmax = nn.Softmax()

    def forward(self, x):

        Q = self.linear_Q(x)
        K = self.linear_K(x)
        V = self.linear_V(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scaling the scores for stabilizing the training
        attention_scores = attention_scores * 1/math.sqrt(self.head_dim)

        # Put all the zero indexes of the causal mask of the attention score to a large negative number (this is for the softmax)
        attention_scores = attention_scores.masked_fill(self.causal_mask == 0, -1e10)

        attention_scores_probs = self.softmax(attention_scores)

        output = torch.matmul(attention_scores_probs, V)

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
        multihead_output = torch.cat(head_outputs, dim=2)

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
        x += attention_output
        x_normed = self.layer_norm_2(x)
        ff_output = self.feed_forward_network(x_normed)
        x += ff_output

        return x



vanillallm_model = VanillmModel(block_size, vocab_size, embedding_dim, n_heads).to('cuda')

print(vanillallm_model.eval())

batch_list = list_2_batch(tokens, batch_size, block_size)


for batch in batch_list:


    # Convert list → tensor [T]
    batch_tensor = torch.tensor(batch, dtype=torch.long, device='cuda')

    # Add batch dimension → [1, T]
    # Because every operation does a down dimension
    batch_tensor = batch_tensor.unsqueeze(0)

    vanillallm_model(batch_tensor)

    
    # Input and target
    #x_batch = batch_tensor[:, :-1]   # [1, T-1]
    #y_batch = batch_tensor[:, 1:]    # [1, T-1]

    #print(batch_tensor.shape)

    # Forward pass
    #logits = vanillallm_model(x_batch)   # [B=1, T-1, vocab_size]

    """
    # Flatten dynamically
    B, T, V = logits.shape
    logits_flat = logits.view(B * T, V)
    y_flat = y_batch.view(B * T)

    # Loss
    loss = F.cross_entropy(logits_flat, y_flat)

    # Backprop
    optim.zero_grad()
    loss.backward()
    optim.step()
    """


