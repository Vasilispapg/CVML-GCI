from ImageCaptionModel import ImageCaptioningModel,train_loop
import torch
import torch.nn as nn
from DataLoaders import FlickrDataset, collate_fnSimplyStack
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from plotloss import plot_loss
from predict import generate_caption
import os
from evaluation import evaluate_model
from saveload import save_checkpoint, load_checkpoint
import numpy as np

def main():

    dataset = FlickrDataset('Flickr8k/images/', 'Flickr8k/captions.txt')

    train_dataset, test_dataset,val_dataset = torch.utils.data.random_split(dataset, [30000, 5000, 5455])


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        collate_fn=collate_fnSimplyStack,
        shuffle=True, 
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fnSimplyStack,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=collate_fnSimplyStack,
        shuffle=True,
        pin_memory=True
    )   
    
    args = ArgumentParser()
    args.add_argument('--train', action='store_true', help='Train the model')
    args.add_argument('--pred', action='store_true', help='Predict using the model')
    args.add_argument('--eval', action='store_true', help='Evaluate the model')
    args.add_argument('--vl', action='store_true', help='Visualize the loss of the model')
    args.add_argument('--gm', action='store_true', help='Get the model')
    args = args.parse_args()
    
    icm = ImageCaptioningModel(vocab_size=len(dataset.vocab),max_seq_len=dataset.max_seq_len)
    model_path = 'model_checkpoint.pth.tar'
    
    
    if args.gm:
        icm = load_checkpoint(torch.load(model_path), icm)

    if args.train:
        assert os.path.exists('Flickr8k/images/'), 'Image folder not found'
        assert os.path.exists('Flickr8k/captions.txt'), 'Captions file not found'
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset, _ = torch.utils.data.random_split(train_dataset, [10000, 20000])
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            collate_fn=collate_fnSimplyStack,
            shuffle=False,
            pin_memory=True
        )
        icm=icm.to(device)
        
        
        optimizer = torch.optim.Adam(icm.parameters(), lr = 0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience=2, verbose = True)
        criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=dataset.vocab.stoi['<pad>'])
        num_epochs = 3
        
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch + 1}')
            train_loop(train_loader, icm, criterion, optimizer, device,scheduler,val_loader=val_loader)
            
            if(os.path.exists(model_path)):
                os.remove(model_path)
            save_checkpoint({
                "state_dict": icm.state_dict(),
                "optimizer": optimizer.state_dict(),
            })
            
        
    elif args.pred:
        # Load the model
        assert os.path.exists(model_path), 'Model checkpoint not found'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab = dataset.vocab
        icm = load_checkpoint(torch.load(model_path), icm)
        image_path = 'test.jpg'
        caption = generate_caption(icm, image_path, vocab,dataset.max_seq_len, device)
        print("Generated Caption:", caption)
    elif args.eval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluate_model(device, icm, test_loader, dataset.vocab)
    elif args.vl:
        plot_loss()
    
    
if __name__ == "__main__":
    main()
