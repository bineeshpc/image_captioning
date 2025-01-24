# Eye-For-Blind

## Problem Statement

The goal of this project is to develop a deep learning model capable of generating spoken descriptions of images, using caption generation with an attention mechanism on the Flickr8K dataset. This model aims to assist visually impaired individuals by generating detailed, spoken descriptions of images. A CNN-RNN architecture will be employed, where the CNN-based encoder extracts image features and the RNN-based decoder generates captions. These captions will then be converted into speech using a text-to-speech library.

This problem statement is an application of both deep learning and natural language processing. The features of an image will be extracted by a CNN-based encoder and this will be decoded by an RNN model. The project is an extended application of the paper _Show, Attend and Tell: Neural Image Caption Generation with Visual Attention_ (https://arxiv.org/abs/1502.03044).

The dataset is taken from the Kaggle website and it consists of sentence-based image descriptions having a list of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events of the image. The dataset can be found here: https://www.kaggle.com/adityajn105/flickr8k.

## Project Pipeline

The major steps that you have to perform can be briefly summarized in the following steps:

1. **Data Understanding**: Load the data and understand the representation.
2. **Data Preprocessing**: Process both images and captions to the desired format.
3. **Train-Test Split**: Combine both images and captions to create the train and test dataset.
4. **Model Building**: Create your image captioning model by building Encoder, Attention, and Decoder models.
5. **Model Evaluation**: Evaluate the models using greedy search and BLEU score.

## Implementation

The implementation of the project is done in the `eye_for_blind.py` file. Below are some key functions and classes used in the implementation:

- **Data Loading and Visualization**:

  - `load_doc(filename)`: Load the document into memory.
  - `visualize_image(path)`: Display an image from the specified file path.
  - `visualize_image_and_captions(index)`: Visualize the image and all its captions for a given index in the dataframe.

- **Data Preprocessing**:

  - Tokenization and padding of captions.
  - Image resizing and normalization.

- **Model Building**:

  - `Encoder`: CNN-based encoder to extract image features.
  - `Attention_model`: Attention mechanism to focus on different parts of the image.
  - `Decoder`: RNN-based decoder to generate captions.

- **Model Training and Evaluation**:

  - Training and testing steps with loss calculation.
  - Evaluation using greedy search and BLEU score.

- **Text-to-Speech**:
  - Convert the generated captions to speech using the `gTTS` library.

## Usage

To run the project, execute the `eye_for_blind.py` file. Ensure you have all the required libraries installed, which can be found in the `requirements.txt` file.

## References

- _Show, Attend and Tell: Neural Image Caption Generation with Visual Attention_ (https://arxiv.org/abs/1502.03044)
- Flickr8K dataset: https://www.kaggle.com/adityajn105/flickr8k
