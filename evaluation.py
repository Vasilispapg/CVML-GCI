import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display


def generate_caption(output, vocab):
    """ Generate a caption from the model output """
    caption = []
    output = output.permute(1,0)
    # output [vocab_size, 1]
    probs = torch.softmax(output, dim=0)
    word_idx = torch.argmax(probs, dim=1)
    for idx in word_idx:
        word = idx.item()
        if word == vocab.stoi['<end>']  :
            break
        if(word == vocab.stoi['<start>'] or word == vocab.stoi['<pad>']):
            continue
        caption.append(vocab.itos[word])
    return ' '.join(caption)
   

def evaluate_model(device, model, data_loader, vocab, criterion=None):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, (imgs, captions, img2display) in enumerate(data_loader):
            imgs, captions = imgs.to(device), captions.to(device)

            # Generate model outputs
            outputs, padding_mask = model(imgs, captions)  # Use all except last token as input
            outputs = outputs.permute(1,0,2) 

            # Calculate loss
            # Assuming outputs is [batch_size, seq_len, vocab_size]
            # Assuming captions is [batch_size, seq_len] with actual word indices
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions.contiguous().view(-1))  # Use all except first token as target

            # Generate and print the predicted caption
            print('Predicted:', generate_caption(outputs[0], vocab))
            
            # Display the image
            plt.imshow(Image.open(img2display[0]))
            plt.show()

            print(f'Batch {i + 1}, Loss: {loss.item()}')

            if i == 0:
                break