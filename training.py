import utils as u
import vanillm as v
import torch
import random
import torch.nn.functional as F
import shared

if __name__ == "__main__":

    tokenizer = u.Tokenizer(shared.poem_txt)
    tokens = tokenizer.txt_2_tokens(shared.poem_txt)
    back_to_txt = tokenizer.tokens_2_txt(tokens)

    # Define how many tokens the model can look at once
    # So the prediction will be based only on these characters
    print("block size: " + str(shared.block_size))

    vocab_size = len(tokenizer.tokens_dict)
    print("vocab size: " + str(vocab_size))

    # Define the dimension of the embedding vector for each token
    print("edmbedding dimension: " + str(shared.embedding_dim))

    print("batch size: " + str(shared.batch_size))

    print("number of attention heads: " + str(shared.n_heads))


    batch_list = u.list_2_batch(tokens, shared.batch_size, shared.block_size)

    model = v.VanillmModel(shared.block_size, vocab_size, shared.embedding_dim, shared.n_heads).to(shared.device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


    for epoch in range(shared.n_epochs):

        random.shuffle(batch_list)

        total_loss = 0

        for batch in batch_list:
            batch_tensor = torch.tensor(batch, dtype=torch.long, device=shared.device)  # [B, T]

            # standard next-token prediction: x -> y shifted by 1
            x_batch = batch_tensor[:, :-1]   # [B, T-1]
            y_batch = batch_tensor[:,  1:]   # [B, T-1]

            logits = model(x_batch)          # [B, T-1, V]

            B, T, V = logits.shape

            logits_flat = logits.reshape(B * T, V)
            y_flat      = y_batch.reshape(B * T)

            loss = F.cross_entropy(logits_flat, y_flat)

            total_loss += loss.item()
            

            optim.zero_grad()
            loss.backward()
            optim.step()

        
        avg_loss = total_loss / len(batch_list)
        print(f"epoch {epoch+1}/{shared.n_epochs}  loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "vanillm.pth")
    print("Model saved successfully as vanillm.pth")