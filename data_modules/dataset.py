from torch.utils.data import Dataset

class CodeDataset_unsup(Dataset):
    def __init__(self, args, df, tokenizer):
        
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        self.max_len = args.max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        sample = self.df.code.values[index]
        
        tokens = self.tokenizer.tokenize(sample)
        tokens = ["[CLS]"] + tokens
        if len(tokens) >= self.max_len:
            tokens  = tokens[:512-1]
        tokens.append("[SEP]")

        input_ids  = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        return {
                'input_ids': input_ids,
                'attention_mask': attention_mask, 
                'token_type_ids': token_type_ids
                }

class CodeDataset_sup(Dataset):
    def __init__(self, args, df, tokenizer):
        
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        self.max_len = args.max_len

    def __len__(self):
        return len(self.df)

    def make_input(self, sample):
        
        tokens = self.tokenizer.tokenize(sample)
        tokens = ["[CLS]"] + tokens
        if len(tokens) >= self.max_len:
            tokens  = tokens[:512-1]
        tokens.append("[SEP]")

        input_ids  = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        return {
                'input_ids': input_ids,
                'attention_mask': attention_mask, 
                'token_type_ids': token_type_ids
                }

    def __getitem__(self, index):

        sample   = self.df.code.values[index]
        positive = self.df.positive.values[index]
        negative = self.df.negative.values[index]

        sample   = self.make_input(sample)
        positive = self.make_input(positive)
        negative = self.make_input(negative)

        return {
                'sample': sample,
                'positive': positive,
                'negative': negative
                }

class CodeDataset_eval(Dataset):
    def __init__(self, args, df, tokenizer):
        
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        self.max_len = args.max_len

    def __len__(self):
        return len(self.df)

    def make_input(self, sample):
        
        tokens = self.tokenizer.tokenize(sample)
        tokens = ["[CLS]"] + tokens
        if len(tokens) >= self.max_len:
            tokens  = tokens[:512-1]
        tokens.append("[SEP]")

        input_ids  = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        return {
                'input_ids': input_ids,
                'attention_mask': attention_mask, 
                'token_type_ids': token_type_ids
                }

    def __getitem__(self, index):
        
        sample1 = self.df.code1.values[index]
        sample2 = self.df.code2.values[index]
        label   = self.df.similar.values[index]

        input1 = self.make_input(sample1)
        input2 = self.make_input(sample2)

        return {
                'input1': input1,
                'input2': input2,
                'label': label
                }