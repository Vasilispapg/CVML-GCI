from ImageCaptionModel import ImageCaptioningModel,train_loop
import torch
import torch.nn as nn
from DataLoaders import FlickrDataset, collate_fnSimplyStack, Vocabulary
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from plotloss import plot_loss
from evaluation import evaluate_model
from saveload import  load_checkpoint

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
    max_seq_len =  max([len(vocab.tokenizer_eng(sentence)) for sentence in df['caption']]) + 1

    # Split groups into training and validation sets ensuring equal length
    all_groups = list(grouped)

    # Split into training and validation sets
    train_groups, val_groups = train_test_split(all_groups, test_size=0.5, random_state=42)

    # Ensure equal length by truncating the larger set
    min_len = min(len(train_groups), len(val_groups))
    train_groups = train_groups[:min_len]
    val_groups = val_groups[:min_len]


    print(f"Training groups: {len(train_groups)}")
    print(f"Validation groups: {len(val_groups)}")

    # Convert list of groups back into DataFrame
    train_df = pd.concat([group for _, group in train_groups]).reset_index(drop=True)
    val_df = pd.concat([group for _, group in val_groups]).reset_index(drop=True)

    train_dataset = FlickrDataset(train_df,max_seq_len=max_seq_len, vocab=vocab)
    val_dataset = FlickrDataset(val_df,max_seq_len=max_seq_len, vocab=vocab)
    

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        collate_fn=collate_fnSimplyStack,
        shuffle=True, 
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=16,
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
    model_path = 'model_checkpoint.pth.tar'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    icm = ImageCaptioningModel(vocab_size=len(vocab), max_seq_len=max_seq_len)
    optimizer = torch.optim.Adam(icm.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
    
    if args.gm:
        icm = load_checkpoint(torch.load(model_path), icm)
        
    if args.train:
        train_model(icm, 
                    train_loader, 
                    val_loader, 
                    device,
                    vocab,
                    optimizer,
                    criterion,
                    epochs=20)
    elif args.pred:
        # Load the model
        df= pd.read_csv('Flickr8k/captions.txt',nrows=1)
        df['image'][0]='test.jpg'
        dataloader= DataLoader(
            dataset=FlickrDataset(df,max_seq_len=max_seq_len, vocab=vocab),
            batch_size=1,
            collate_fn=collate_fnSimplyStack,
            shuffle=False,
            pin_memory=True
        )
        evaluate_model(device, icm, dataloader, vocab,criterion)

    elif args.eval:
        evaluate_model(device, icm, val_loader, vocab,criterion)
    if args.vl:
        plot_loss(type=2)
    
    
def train_model(icm, train_loader, val_loader, device, vocab, optimizer, criterion, epochs=3):
    print('Training model')
    print('Device:', device)
    print('=' * 20)

    # Ensure the model is on the correct device
    icm = icm.to(device)

    train_loop(model=icm, 
               dataloader=train_loader,
               val_loader=val_loader, 
               criterion=criterion,
               optimizer=optimizer,
               device=device,
               vocab=vocab,
               epochs=epochs)
    print('Training complete')

    
if __name__ == "__main__":
    main()
