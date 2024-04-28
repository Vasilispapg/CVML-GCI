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
   

def evaluate_model(device, model, data_loader, vocab):
    model.eval()
    criterion = nn.CrossEntropyLoss()  # Ensure this uses any class weighting or reduction correctly
    model = model.to(device)
    with torch.no_grad():
        for i, (imgs, captions,img2display) in enumerate(data_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            outputs, padding_mask = model(imgs, captions)
            outputs = outputs.permute(1, 2, 0)  # Correct dimension: [seq_len, vocab_size, batch_size]

            # Apply padding mask if necessary before loss calculation
            if padding_mask is not None:
                outputs = outputs * padding_mask.unsqueeze(1)  # Mask non-relevant predictions

            loss = criterion(outputs.permute(0,2,1).contiguous().view(-1,2994), captions.contiguous().view(-1))
            final_batch_loss = torch.sum(loss) / torch.sum(padding_mask)

            print('Predicted:', generate_caption(outputs[0], vocab))

            # Display the image
            image = Image.open(img2display[0])

            display(image)
            image.show()

            print(f'Batch {i + 1}, Loss: {final_batch_loss.item()}')

            if i == 0:
                break