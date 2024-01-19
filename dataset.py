
import torch
import tokenizer
import datasets

# Create dataset classs using Dataset library
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.ds = dataset

        # Tokeniser function
        self.tk_q = tokenizer.Tokenizer('questions')
        self.tk_a = tokenizer.Tokenizer('answers')

    def __len__(self):
        return len(self.ds) # Find dataset length

    # Retrieve data from index, tokenise and add BOS and EOS tokens 
    def __getitem__(self, idx):
        instruction = self.ds[idx]['instruction']
        input = self.ds[idx]['input']
        sql = self.ds[idx]['response']
        encode_input = instruction + input # create combined input 
        e_text = self.tk_q.encode(encode_input) # tokenise input 
        d_input = [self.tk_a.sp.bos_id()] + self.tk_a.encode(sql) #tokenise decoder data + bos
        d_target = (self.tk_a.encode(sql)) + [self.tk_a.sp.eos_id()] #tokenise decoder data + eos
        return { 'd_input': torch.tensor(d_input), 'd_target': torch.tensor(d_target), 'e_text': torch.tensor(e_text) }

    # Pad tokenised data to the same length  
    def collate_fn(self, batch):
        e_text_pad = torch.nn.utils.rnn.pad_sequence([item['e_text'] for item in batch], batch_first=True, padding_value=self.tk_q.sp.pad_id())
        d_input_pad = torch.nn.utils.rnn.pad_sequence([item['d_input'] for item in batch], batch_first=True, padding_value=self.tk_a.sp.pad_id())
        d_target_pad = torch.nn.utils.rnn.pad_sequence([item['d_target'] for item in batch], batch_first=True, padding_value=self.tk_a.sp.pad_id())
        
        return { 'd_input': d_input_pad, 'd_target': d_target_pad, 'e_text': e_text_pad }

# Generate training, validation, and test datasets 
def dataset_generator(dataset):
    ds = datasets.load_dataset(dataset)['train']

    split = ds.train_test_split(test_size=0.2, shuffle=False)
    train_data = split['train']
    combo_data = split['test']
    split2 = combo_data.train_test_split(test_size = 0.5, shuffle = False)
    test_data = split2['train']
    val_data = split2['test']

    
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)
    val_dataset = Dataset(val_data)
    
    return train_dataset, val_dataset, test_dataset

