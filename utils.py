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

