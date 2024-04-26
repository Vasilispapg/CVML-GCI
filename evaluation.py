import torch
import torch.nn as nn

def evaluate_model(device, model, data_loader):
    """ Evaluate the model on a given dataset """
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():  # Disable gradient calculation
        for i, (imgs, captions) in enumerate(data_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(imgs, captions)
            breakpoint()
            
            # TODO Xtypaei edw
            # Trim the last 2 timestep Not good practice
            outputs = outputs[:, :-2, :]
            
            # Calculate the loss
            loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
            
            # Output the loss for this batch
            print(f'Batch {i + 1}, Loss: {loss.item()}')
            
            # Stop after processing 6 batches
            if i == 5:
                break
