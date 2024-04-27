
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
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<UNK>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<UNK>": 3}
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

    def numericalize(self, tokens):
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokens
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
        self.max_seq_len = max([len(self.vocab.tokenizer_eng(sentence)) for sentence in self.captions]) + 2  # <start> and <end>

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return len(self.imgs)
    def remove_single_char_word(self, word_list):
        return [word for word in word_list if len(word) > 1]

    def preProccesCaption(self, caption):
        caption_tokens = self.remove_single_char_word(self.vocab.tokenizer_eng(caption))
        caption_tokens = ['<start>'] + [word for word in caption_tokens if word.isalpha()] + ['<end>']
        caption_tokens += ['<pad>'] * (self.max_seq_len - len(caption_tokens))
        return self.vocab.numericalize(caption_tokens)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        caption = self.captions[index]
        numericalized_caption = self.preProccesCaption(caption)

        return img, torch.tensor(numericalized_caption, dtype=torch.long)

    

def collate_fn(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    # Pad the captions in the batch to have the same length
    captions = pad_sequence([torch.tensor(caption) for caption in captions], batch_first=True, padding_value=0)
    return images, captions


def collate_fnSimplyStack(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)  # Stack already-padded captions
    return images, captions