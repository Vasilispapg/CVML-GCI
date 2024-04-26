import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt

# Preprocess the image to fit Xception's input requirements
def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def generate_caption(model, image_path, vocab, max_length=20, device='cuda'):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299,299)),
            # bw
            # transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # normalize to [0,1]
        ])
    image = load_image(image_path, transform)
    
    plt.imshow(image.squeeze(0).permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        features = model.visual_model(image)
        encoded_features = model.encoder(features)
        inputs = torch.tensor([vocab.stoi['<SOS>']]).unsqueeze(0).to(device)
        caption = []

        for _ in range(max_length):
            outputs = model.decoder(encoded_features, inputs)
            outputs = outputs.squeeze(0)  # Shape should be [seq_length, vocab_size]

            # We use the output of the last timestep for prediction
            last_output = outputs[-1].unsqueeze(0)  # Ensure it's [1, vocab_size]

            max_index = last_output.argmax(dim=1)
            if max_index.numel() != 1:
                raise RuntimeError(f"Expected max_index to have 1 element, but got {max_index.numel()} elements.")

            word = vocab.itos[max_index.item()]
            caption.append(word)
            if word == '<EOS>':
                break
            inputs = max_index.unsqueeze(0)  # Prepare input for the next timestep

    return ' '.join(caption)

