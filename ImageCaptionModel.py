import torch
import torch.nn as nn
from EncoderDecoder import Encoder, Decoder
from Xception import xception
import time
from evaluation import evaluate_model

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.visual_model = xception(num_classes=1000)
        self.visual_model.fc = nn.Identity()  # Adapt final layer based on Xception architecture
        for param in self.visual_model.parameters():
            param.requires_grad = True  # Freeze all parameters of the Xception model
        
        self.encoder = Encoder(2048, 512)
        self.decoder = Decoder(16, 4, vocab_size, 512, max_seq_len=max_seq_len)

    def forward(self, images, captions):
        # with torch.no_grad():
        xception_features = self.visual_model(images)  # Extract features using Xception
            
        encoded_features = self.encoder(xception_features)
        # output [batch_size, 512*7*7] 
        output = self.decoder(encoded_features, captions)
        # output [batch_size, seq_len, vocab_size]
        return output


def train_loop(dataloader, model, loss_fn, optimizer, device,scheduler=None,val_loader=None):
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
        
        optimizer.zero_grad()
        
        outputs , padding_mask = model(img, captions)
        # outputs [batch_size, seq_len-1, vocab_size]
        output = outputs.permute(1, 2, 0)

        loss = loss_fn(output,captions)

        loss_masked = torch.mul(loss, padding_mask)

        final_batch_loss = torch.sum(loss_masked)/torch.sum(padding_mask)

        final_batch_loss.backward()
        optimizer.step()
                        
        total_loss += final_batch_loss.item()
        loss_list.append(final_batch_loss.item())
        print(f"Loss: {final_batch_loss.item():.4f}")
        
        print(f"Time: {time.time()-timer}")
    
    average_loss = final_batch_loss / len(dataloader)

    
    
    print(f"Average Loss: {average_loss:.4f}")
    
    print('Evaluating model')
    evaluate_model(device, model, val_loader)
    
    # save lostlist in a file
    with open("loss.txt", "a") as f:
        f.write(str(loss_list))
        f.write("\n")