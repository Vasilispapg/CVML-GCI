import torch
import torch.nn as nn
from EncoderDecoder import Encoder, Decoder
from Xception import xception
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.visual_model = xception(num_classes=1000)
        self.visual_model.fc = nn.Identity()  # Adapt final layer based on Xception architecture
        for param in self.visual_model.parameters():
            param.requires_grad = False  # Freeze all parameters of the Xception model
        
        self.encoder = Encoder()

        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        with torch.no_grad():
            xception_features = self.visual_model(images)  # Extract features using Xception
            
        encoded_features = self.encoder(xception_features)
        # output [batch_size, 512*7*7] 
        output = self.decoder(encoded_features, captions)
        # output [batch_size, seq_len, vocab_size]
        return output


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    num_batch=0
    loss_list=[]
    for img, captions in dataloader:
        timer=time.time()
        print("==="*20)
        num_batch+=1
        print(f"Batch: {num_batch}")
        img, captions = img.to(device), captions.to(device)
        
        # Assuming token dropping or other preprocessing here if needed
        outputs = model(img, captions[:, :-1])

        outputs = outputs[:, :-1, :]  # Trim the last timestep

        loss = loss_fn(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        total_loss += loss.item()
        loss_list.append(loss.item())
        print(f"Loss: {loss.item():.4f}")
        
        print(f"Time: {time.time()-timer}")
    
    average_loss = total_loss / len(dataloader)
    
    print(f"Average Loss: {average_loss:.4f}")
    
    # save lostlist in a file
    with open("loss.txt", "a") as f:
        f.write(str(loss_list))
        f.write("\n")