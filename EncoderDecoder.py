import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size):
        super(Encoder, self).__init__()
        # Define the projection layer to map from input_dim to embed_size
        self.feature_projection = nn.Linear(input_dim, embed_size)

    def forward(self, x):
        # x expected shape: [batch_size, channels, height, width]
        batch_size, channels, height, width = x.size()
        # Flatten the spatial dimensions and maintain the feature/channel dimension
        x = x.view(batch_size, channels, height * width)  # [batch_size, channels, height*width]
        # x : [batch_size, channels, height*width]
        
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, height*width, channels] for linear projection
        # x : [batch_size, height*width, channels]
        x = self.feature_projection(x)  # Project from input_dim to embed_size
        # x : [batch_size, height*width, embed_size]
        return x

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden, seq_len):
        # x: [batch_size, height*width, embed_size]
        # hidden: [num_layers, batch_size, hidden_size]
        # Ensure x is of shape [batch_size, seq_len, embed_size] to match captions
        if x.size(1) != seq_len:
            x = x[:, :seq_len, :]  # [batch_size, max_seq_len, embed_size]
        output, hidden = self.gru(x, hidden)
        # output : [batch_size, max_seq_len, hidden_size]
        # hidden : [num_layers, batch_size, hidden_size]
        
        output = self.fc(output)  # [batch_size, max_seq_len, vocab_size]
        return output, hidden