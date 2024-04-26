import torch
import torch.nn as nn
import torch.optim as optim




class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2048, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)  # Adding batch normalization after each conv layer
        self.conv2 = nn.Conv2d(1024, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Apply batch normalization
        x = torch.relu(x)  # Activation function
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.flatten(x)
        return x
    
"""class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(512*10*10, embed_size)
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
        # Stack multiple LSTM layers
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=embed_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True)
            for i in range(num_layers)
        ])
        
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        features = self.linear(features)
        features = features.unsqueeze(1)  # Add time dimension for LSTM
        
        embeddings = self.embedding(captions)
        x = torch.cat((features, embeddings), dim=1)
        x = self.dropout(x)
        
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                x += lstm_output  # Residual connection
            lstm_output, _ = lstm(x)
            x = self.dropout(lstm_output)
        
        outputs = self.fc(x)
        
        return outputs
"""

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout=0.5):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(512*10*10, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)  # Batch normalization for the linear output

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
        # Use GRUs instead of LSTMs
        self.grus = nn.ModuleList([
            nn.GRU(input_size=embed_size if i == 0 else hidden_size,
                   hidden_size=hidden_size,
                   num_layers=1,
                   batch_first=True)
            for i in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        features = self.linear(features)
        features = self.bn(features)  # Apply batch normalization
        features = torch.relu(features)  # Activation function
        features = features.unsqueeze(1)  # Add time dimension for GRU
        
        embeddings = self.embedding(captions)
        x = torch.cat((features, embeddings), dim=1)
        x = self.dropout(x)
        
        for i, gru in enumerate(self.grus):
            if i > 0:
                x += gru_output  # Residual connection
            gru_output, _ = gru(x)
            x = self.dropout(gru_output)
        
        outputs = self.fc(x)
        
        return outputs
