import torch
from decoder import Transformer
from tokenizer import Tokenizer

import constants

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Instantiate transformer model
model = Transformer(
    vocab_size=constants.VOCAB_SIZE,
    embedding_dim=constants.EMBEDDING_DIM,
    ff_dim=constants.FF_DIM,
    num_heads=constants.NUM_HEADS,
    num_layers=constants.NUM_LAYERS,
    dropout=constants.DROPOUT,
    with_encoder=constants.WITH_ENCODER,
)

model = model.to(device)

text_tokenizer = Tokenizer('questions')
sql_tokenizer = Tokenizer('answers')
# model.load_weights('model.pt')


# Function to generate SQL from text 
def inference(text):
    tokenised_text = text_tokenizer.encode(text)
    encoder_input = torch.tensor([tokenised_text]).to(device)
    max_sequence_length = 40

    # Initialise decoding process
    next_token_id = sql_tokenizer.sp.bos_id()
    iterable_sql = [next_token_id]
    break_count = 0

    # Generate tokens until EOS token produced or max sequence length reached 
    while (next_token_id != sql_tokenizer.sp.eos_id()) & (break_count <= max_sequence_length):
        iterable_decoder_tensor = torch.tensor(iterable_sql).view(1,-1)
        probability_matrix = model(encoder_input=encoder_input, decoder_input=iterable_decoder_tensor)
        probability_vector = probability_matrix[0, -1, :] 
        next_token_id = (torch.multinomial(probability_vector, 1)) # predict next token 
        iterable_sql = iterable_sql + [next_token_id.item()]
        break_count += 1

    return sql_tokenizer.decode(iterable_sql) # decode tokens to produce SQL 





