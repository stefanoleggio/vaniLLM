import torch
import utils as u
import vanillm as v
import sys
import time
import shared

def generate_characters(model, tokenizer, input_char, n_gen_char, device=shared.device):
    input_tokens = tokenizer.txt_2_tokens(input_char)  # list of ints

    model.eval()
    generated_tokens = []

    block_size = model.block_size  # use model's block size

    with torch.no_grad():
        for _ in range(n_gen_char):
            # [1, T]
            input_tokens_tensor = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)

            logits = model(input_tokens_tensor)          # [1, T, V]
            last_token_logits = logits[:, -1, :]         # [1, V]
            last_token_probs = torch.softmax(last_token_logits, dim=-1)  # [1, V]

            next_token = torch.argmax(last_token_probs, dim=-1)  # [1]
            token_id = next_token.item()

            generated_tokens.append(token_id)
            input_tokens.append(token_id)

            if len(input_tokens) > block_size-1:
                input_tokens = input_tokens[1:]

    return tokenizer.tokens_2_txt(generated_tokens)

if __name__ == "__main__":


    tokenizer = u.Tokenizer(shared.poem_txt)
    vocab_size = len(tokenizer.tokens_dict)

    model = v.VanillmModel(shared.block_size, vocab_size, shared.embedding_dim, shared.n_heads).to(shared.device)
    state_dict = torch.load("vanillm.pth", map_location=shared.device, weights_only=True)
    model.load_state_dict(state_dict)

    while True:
        try:
            n_gen_char = int(input('\nHow many characters do you want to generate?\n'))
            break
        except:
            print("Insert an integer!")
        
    
while True:
    input_string = input("\nWrite a sentence of the poem and vanillm will complete it!\n")

    input_list = list(input_string)
    generated_output_list = generate_characters(model, tokenizer, input_list, n_gen_char)

    # Move cursor UP one line (back to where user typed)
    sys.stdout.write("\033[F")
    # Clear that line
    sys.stdout.write("\033[K")

    # Print the user input in normal color
    sys.stdout.write(input_string)
    sys.stdout.flush()

    # Now set YELLOW for the generated text
    sys.stdout.write("\033[33m")   # 33m = yellow
    sys.stdout.flush()

    # Print generated characters in yellow on the SAME line
    for ch in generated_output_list:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(0.05)

    # Reset color back to normal and go to new line
    sys.stdout.write("\033[0m\n")
    sys.stdout.flush()