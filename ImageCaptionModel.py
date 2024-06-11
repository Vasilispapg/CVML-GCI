import torch
import torch.nn as nn
from EncoderDecoder import Encoder, Decoder
from Xception import xception
import time
import os
from evaluation import evaluate_model


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.visual_model = xception(num_classes=1000)
        self.visual_model.fc = nn.Identity()  # aAdapt final layer based on Xception architecture
        for param in self.visual_model.parameters():
            param.requires_grad = False  
        self.embed=1024
        self.encoder = Encoder(2048, self.embed)
        self.decoder = Decoder(self.embed, hidden_size=1024, vocab_size=vocab_size, num_layers=64)

    def forward(self, images, captions):
        # image : [batch_size, channels, height, width]
        # captions : [batch_size, max_seq_len]
        
        xception_features = self.visual_model(images)  # Extract features using ResNet
        # xception_features : [batch_size, 2048, 10, 10]
        
        encoded_features = self.encoder(xception_features)
        # x : [batch_size, height*width, embed_size]
        
        # Initialize the hidden state for the GRU
        hidden = torch.zeros(self.decoder.num_layers, encoded_features.shape[0], self.decoder.hidden_size).to(images.device)
        # [num_layers, batch_size, hidden_size]
        # Use the encoded features as input to the decoder
        output, hidden = self.decoder(encoded_features, hidden, captions.size(1))
        return output, captions


def train_loop(dataloader, model, criterion, optimizer, device, scheduler=None, val_loader=None, vocab=None, epochs=20):
    loss_list = []    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_idx=0 
        num_batches = len(dataloader)
        for (img, captions, _) in dataloader:
            timer = time.time()
            print(f"Batch: {(batch_idx + 1)} of {num_batches} batches")
            print(f'Epoch: {epoch + 1}/{epochs}')
            img, captions = img.to(device), captions.to(device)
            
            optimizer.zero_grad()
            outputs, padding_mask = model(img, captions)

            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size*height*width, vocab_size]
            captions = captions.contiguous().view(-1)  # [batch_size*max_seq_len]
            
            loss = criterion(outputs, captions)
            loss_masked = torch.mul(loss, padding_mask)
            final_batch_loss = torch.sum(loss_masked) / torch.sum(padding_mask)
            final_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            batch_idx += 1

            total_loss += final_batch_loss.item()
            loss_list.append(final_batch_loss.item())
            print(f"Loss: {final_batch_loss.item():.4f}")
            print(f"Time per batch: {time.time() - timer:.2f} seconds")
        with open("loss.txt", "a") as f:
            f.write(str(loss_list) + '\n')
        loss_list = []
        
        average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} Average Loss: {average_loss:.4f}")
        if val_loader:
            print('Evaluating model...')
            evaluate_model(device, model, val_loader, vocab, criterion, type=1)

        if epoch % 100 == 0 or epoch == epochs - 1:
            checkpoint_path = 'model_checkpoint.pth.tar'
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path)
            print("Model checkpoint saved.")

    print('Training complete')