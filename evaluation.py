import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display


def generate_caption(output, vocab):
    """ Generate a caption from the model output """
    caption = []
    probs = torch.softmax(output, dim=0)
    word_idx = torch.argmax(probs, dim=1)

    print("Generated indices:", word_idx.tolist())  # Debugging print to check the generated indices

    for idx in word_idx:
        word = idx.item()
        if word == vocab.stoi['<end>']:
            break
        if word == vocab.stoi['<start>'] or word == vocab.stoi['<pad>']:
            continue
        caption.append(vocab.itos.get(word, '<UNK>'))  # Ensure to handle unknown tokens
    return ' '.join(caption)
   

def evaluate_model(device, model, data_loader, vocab, criterion=None,type=0):
    model.eval()
    model = model.to(device)
    total_loss = 0
    num_batches = len(data_loader)
    results = []
    loss_arr=[]

    with torch.no_grad():
        for i, (imgs, captions, img2display) in enumerate(data_loader):
            imgs, captions = imgs.to(device), captions.to(device)

            # Generate model outputs
            outputs, padding_mask = model(imgs, captions)  # Use all except last token as input

            # Assuming outputs is [batch_size, max_seq_len, vocab_size]
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size*max_seq_len, vocab_size]
            captions = captions.contiguous().view(-1)  # [batch_size*max_seq_len]

            # Calculate loss
            if criterion is not None:
                loss = criterion(outputs, captions)
                loss_arr.append(loss.item())
                total_loss += loss.item()

            # Reshape outputs back to [batch_size, max_seq_len, vocab_size] for caption generation
            outputs = outputs.view(imgs.size(0), -1, outputs.size(-1))  # [batch_size, max_seq_len, vocab_size]

            if type==0:
                for j in range(imgs.size(0)):
                    predicted_caption = generate_caption(outputs[j], vocab)
                    results.append((img2display[j], predicted_caption))

                    if i == 0 and j == 0:  # Display the first image and caption
                        print('Predicted:', predicted_caption)
                        plt.imshow(Image.open(img2display[j]))
                        plt.show()

            if criterion is not None:
                print(f'Batch {i + 1}, Loss: {loss.item()}')

    if criterion is not None:
        average_loss = total_loss / num_batches
        print(f"Evaluation Average Loss: {average_loss:.4f}")

        # Save evaluation statistics
        with open("evaluation_results.txt", "a") as f:
            for img_path, caption in results:
                f.write(f"{img_path}: {caption}\n")
                
        with open("evaluation_loss.txt", "a") as f:
            f.write(f"{loss_arr}\n")
            

    return average_loss if criterion is not None else results