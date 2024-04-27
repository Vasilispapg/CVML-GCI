import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def generate_caption(output, vocab):
    """ Generate a caption from the model output """
    caption = []
    output = output.permute(1,0)
    # output [vocab_size, 1]
    probs = torch.softmax(output, dim=0)
    word_idx = torch.argmax(probs, dim=1)
    for idx in word_idx:
        word = idx.item()
        if word == vocab.stoi['<end>']:
            break
        if(word == vocab.stoi['<start>'] or word == vocab.stoi['<pad>']):
            continue
        caption.append(vocab.itos[word])
    return ' '.join(caption)
   

def evaluate_model(device, model, data_loader,vocab):
    """ Evaluate the model on a given dataset """
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():  # Disable gradient calculation
        for i, (imgs, captions) in enumerate(data_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            # Forward pass: compute predicted outputs by passing inputs to the model
            breakpoint()
            outputs , padding_mask = model(imgs, captions)
            # outputs [batch_size, seq_len-1, vocab_size]
            output = outputs.permute(1, 2, 0)
            # [2941,42]

            loss = criterion(output,captions)

            loss_masked = torch.mul(loss, padding_mask)

            final_batch_loss = torch.sum(loss_masked)/torch.sum(padding_mask)
            
            # write the caption
            outputs=outputs.permute(1,2,0)
            print('Predicted:', generate_caption(outputs[0], vocab))
            
            # display image
            plt.imshow(imgs[0].permute(1, 2, 0).cpu().numpy())
            plt.show()
            
            # Output the loss for this batch
            print(f'Batch {i + 1}, Loss: {final_batch_loss.item()}')
            
            # Stop after processing 5 batches
            if i == 4:
                break
