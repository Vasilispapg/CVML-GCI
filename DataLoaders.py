
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import spacy
from torch.nn.utils.rnn import pad_sequence

spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
        
    def denumericalize(self, tensor):
        text = [self.itos[token] for token in tensor]

        return text


class FlickrDataset(Dataset):
    def __init__(self, images_dir, captions_file, freq_threshold=5, transform=None):
        self.imgs = pd.read_csv(captions_file)['image'].map(lambda x: images_dir + x).tolist()
        self.captions = pd.read_csv(captions_file)['caption'].tolist()
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299,299)),
            # bw
            # transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # normalize to [0,1]
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        caption = self.captions[index]
        numericalized_caption = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(numericalized_caption, dtype=torch.long)
    


def collate_fn(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    # Pad the captions in the batch to have the same length
    captions = pad_sequence([torch.tensor(caption) for caption in captions], batch_first=True, padding_value=0)
    return images, captions