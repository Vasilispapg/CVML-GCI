
# CVML-GCI Image Captioning System

## Overview
This repository provides the implementation for an image captioning system using a hybrid approach that combines a pre-trained Xception model with a custom Transformer architecture. This project is designed to generate captions from images, leveraging advancements in computer vision and natural language processing.

## Features
- Image feature extraction using the Xception model.
- Caption generation with a Transformer-based model.
- Evaluation scripts to test the model's performance.
- Utilities for handling dataset loading, model saving/loading, and loss visualization.

## Repository Contents
- `EncoderDecoder.py`: Implementation of the an Encoder(Extention of the Xception) and Custom Transformer's decoder.
- `Xception.py`: Xception model setup and feature extraction.
- `ImageCaptioningModel.py`: Integration of CNN and Transformer models for captioning.
- `DataLoaders.py`: Dataset preparation and loading utilities.
- `evaluation.py`: Model evaluation on a validation/custom set.
- `plotloss.py`: Visualization of training losses.
- `saveload.py`: Utilize the save/load functions.

## Prerequisites
To run the code in this repository, you will need:
- Python 3.6 or newer
- PyTorch 1.7 or newer
- Matplotlib for plotting
- Pandas for data manipulation
- SpaCy for tokenization
- PIL for image file operations

## Usage
### Training the Model
To train the image captioning model, run:
```bash
python train.py --train
```

### Evaluating the Model
Evaluate the model's performance on the validation dataset:
```bash
python evaluate.py --eval
```

### Generating Captions
Generate captions for new images:
```bash
python predict.py --img_path <path_to_image>
```

### Visualizing Loss
Visualize the training and validation loss:
```bash
python plot_loss.py
```

## Contributing
Contributions are welcome. Please fork the repository and submit pull requests with your suggested changes.

## License
This project is open-sourced under the MIT license.

## Authors
- [Your Name] - Initial work
