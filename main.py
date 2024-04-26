from ImageCaptionModel import ImageCaptioningModel,train_loop
import torch
import torch.nn as nn
from DataLoaders import FlickrDataset, collate_fn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from plotloss import plot_loss
from predict import generate_caption
import os
from evaluation import evaluate_model
from saveload import save_checkpoint, load_checkpoint

def main():

    dataset = FlickrDataset('Flickr8k/images/', 'Flickr8k/captions.txt')

    train_dataset, test_dataset,val_dataset = torch.utils.data.random_split(dataset, [30000, 5000, 5455])


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        collate_fn=collate_fn,
        shuffle=True, 
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        
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
    
    icm = ImageCaptioningModel(embed_size=256, hidden_size=1024, vocab_size=len(dataset.vocab), num_layers=12)
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
            batch_size=64,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True
        )
        icm=icm.to(device)
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"], reduction='mean', weight=None, size_average=None)
        optimizer = torch.optim.Adam(icm.parameters(), lr=0.0001)
        # rmsDrop
        # optimizer= torch.optim.RMSprop(icm.parameters(), lr=0.0001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        # SGD 
        # optimizer = torch.optim.SGD(icm.parameters(), lr=0.002, momentum=0.9)
        
        num_epochs = 8
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch + 1}')
            train_loop(train_loader, icm, loss_fn, optimizer, device,epoch+1)
            
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
        caption = generate_caption(icm, image_path, vocab)
        print("Generated Caption:", caption)
    elif args.eval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluate_model(device, icm, test_loader)
    elif args.vl:
        plot_loss()
    
    
if __name__ == "__main__":
    main()
