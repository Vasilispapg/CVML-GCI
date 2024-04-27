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
        x = x.view(batch_size, channels, height * width)  # [batch_size, 2048, 100]
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, 100, 2048] for linear projection
        x = self.feature_projection(x)  # Project from input_dim to embed_size
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_size] for Transformer compatibility
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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1,max_seq_len=42):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        

    def forward(self, x, device='cuda'):
        if self.pe.size(0) < x.size(0):
            self.pe = self.pe.repeat(x.size(0), 1, 1).to(device)
        self.pe = self.pe[:x.size(0), : , : ]
        x = x + self.pe
        return self.dropout(x)
    
class Decoder(nn.Module):
    def __init__(self, n_head, n_decoder_layer, vocab_size, embedding_size,max_seq_len = 42):
        super(Decoder, self).__init__()
        self.pos_encoder = PositionalEncoding(embedding_size, 0.1,max_seq_len)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model =  embedding_size, nhead = n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer = self.TransformerDecoderLayer, num_layers = n_decoder_layer)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size , embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)

    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp, device = 'cuda'):
        # encoded_image = encoded_image.permute(2, 0, 1)
        # breakpoint()
        decoder_inp_embed = self.embedding(decoder_inp)* math.sqrt(self.embedding_size)
        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1,0,2)
        

        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask.to(device)
        decoder_input_pad_mask = decoder_input_pad_mask.to(device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(device)
        
        decoder_output = self.TransformerDecoder(tgt = decoder_inp_embed, memory = encoded_image, tgt_mask = decoder_input_mask, tgt_key_padding_mask = decoder_input_pad_mask_bool)
        
        final_output = self.last_linear_layer(decoder_output)

        return final_output,  decoder_input_pad_mask