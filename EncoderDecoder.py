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
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.4, max_seq_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings for max_seq_len positions
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure positional encoding is on the correct device and can handle different batch sizes
        # x: [batch_size, seq_len, d_model]
        # Extend pe to match batch size in x
        pe = self.pe[:, :x.size(1), :]  # [1, seq_len, d_model]
        pe = pe.expand(x.size(0), -1, -1)  # [batch_size, seq_len, d_model]
        x = x + pe
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
        initrange = 0.2
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