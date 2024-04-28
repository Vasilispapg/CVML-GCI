from ImageCaptionModel import ImageCaptioningModel,train_loop
import torch
import torch.nn as nn
from DataLoaders import FlickrDataset, collate_fnSimplyStack, Vocabulary
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from plotloss import plot_loss
from predict import generate_caption
import os
from evaluation import evaluate_model
from saveload import save_checkpoint, load_checkpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def SplitDataset(df):
    df['image'] = df['image'].map(lambda x: 'Flickr8k/images/' + x)

    # Group data by image
    grouped = df.groupby('image')
    
    # Make a vocabulary
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(df['caption'].tolist())
   
    # get the max sequence length for all dataset
    max_seq_len =  max([len(vocab.tokenizer_eng(sentence)) for sentence in df['caption']]) + 2

    # Split groups into training and validation sets
    train_groups, val_groups = train_test_split(list(grouped), test_size=0.4, random_state=42)
    print(f"Number of training groups: {len(train_groups)}")
    print(f"Number of validation groups: {len(val_groups)}")

    # Convert list of groups back into DataFrame
    train_df = pd.concat([group for _, group in train_groups]).reset_index(drop=True)
    val_df = pd.concat([group for _, group in val_groups]).reset_index(drop=True)

    train_dataset = FlickrDataset(train_df,max_seq_len=max_seq_len, vocab=vocab)
    val_dataset = FlickrDataset(val_df,max_seq_len=max_seq_len, vocab=vocab)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        collate_fn=collate_fnSimplyStack,
        shuffle=True, 
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=collate_fnSimplyStack,
        shuffle=True,
        pin_memory=True
    )   
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    return train_loader,val_loader,max_seq_len,vocab


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--pred', action='store_true', help='Predict using the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--vl', action='store_true', help='Visualize the loss of the model')
    parser.add_argument('--gm', action='store_true', help='Get the model')
    return parser.parse_args()


def main():
    
    df = pd.read_csv('Flickr8k/captions.txt')
    train_loader, val_loader,max_seq_len,vocab = SplitDataset(df)

    args = get_arguments()
    icm = ImageCaptioningModel(vocab_size=len(vocab), max_seq_len=max_seq_len)
    model_path = 'model_checkpoint.pth.tar'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args=get_arguments()
    
    if args.gm:
        icm = load_checkpoint(torch.load(model_path), icm)
    if args.train:
        train_model(icm, train_loader, val_loader, device,vocab)
    elif args.pred:
        # Load the model
        caption = generate_caption(icm, 'test.jpg', vocab,max_seq_len, device)
        print("Generated Caption:", caption)
    elif args.eval:
        evaluate_model(device, icm, val_loader, vocab)
    if args.vl:
        plot_loss()
    
    
def train_model(icm, train_loader, val_loader, device,vocab):
    print('Training model')
    print('Device:', torch.cuda.get_device_name(0))
    print('='*20)
    train_loop(model=icm.to(device), 
               dataloader=train_loader,
               val_loader=val_loader, 
               loss_fn=nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>']), 
               optimizer=torch.optim.Adam(icm.parameters(), lr=0.00001), 
               device=device)
    print('Training complete')
    
if __name__ == "__main__":
    main()
