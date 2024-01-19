import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import dataset
import constants
import tokenizer
from decoder import Transformer
import wandb
import tqdm
wand = True

##### cuda magic ####
def getDevice():
  is_cuda = torch.cuda.is_available()
  return "cuda:0" if is_cuda else "cpu"


#initiate wanb #
if wand == True:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Transformer",
        
        # track hyperparameters and run metadata
        config= {
        "learning_rate": constants.LEARNING_RATE,
        "dimensions": constants.EMBEDDING_DIM,
        "vocab_size": constants.VOCAB_SIZE,
        "epochs": constants.EPOCHS,
        "num_heads" : constants.NUM_HEADS,
        "num_layers" : constants.NUM_LAYERS,
        "ff_dim" : constants.FF_DIM,
        "dropout" : constants.DROPOUT,
        "batch_size" : constants.BATCH_SIZE
        }
    )


device = getDevice()

#load dataset
ds_train, ds_val, ds_test = dataset.dataset_generator("Clinton/Text-to-sql-v1")  
dl = torch.utils.data.DataLoader(ds_train, batch_size= constants.BATCH_SIZE, shuffle=True, collate_fn = ds_train.collate_fn)

#instantiate transformer and optimiser
transformer = Transformer(constants.VOCAB_SIZE,constants.EMBEDDING_DIM,constants.FF_DIM,constants.NUM_HEADS,constants.NUM_LAYERS,constants.DROPOUT, with_encoder =True).to(device)
optimizer = optim.Adam(transformer.parameters(), lr=constants.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

transformer.train()


##### Training Loop ####
for epoch in range(0, constants.EPOCHS):
    #iterate over data batches
    for idx, batch in enumerate(tqdm.tqdm(dl, desc="Processing")):
        d_input = batch['d_input'].to(device)
        d_labels = batch['d_target'].to(device)
        e_input = batch['e_text'].to(device)

        output = transformer(e_input,d_input) # generate outputs from transformer model

        #reshape output and labels for loss calc
        output = output.view(-1, output.size(-1))
        d_labels = d_labels.view(-1)

        loss = torch.nn.functional.cross_entropy(output,d_labels,label_smoothing=0.1) # calculate loss from model 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if wand == True:
            wandb.log({"loss":loss})
   
    print(f"Epoch {epoch+1}/{constants.EPOCHS}, Loss: {loss}")
    torch.save(transformer.state_dict(), f"./transformer_epoch_{epoch+1}.pt")  # save model at current epoch 

    if wand == True:
        wandb.log({"loss":loss})

if wand == True:   
    wandb.finish()
