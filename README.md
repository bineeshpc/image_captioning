# Eye-For-Blind
## Problem statement:

The goal of this project is to develop a deep learning model capable of generating spoken descriptions of images, using caption generation with an attention mechanism on the Flickr8K dataset.

This model aims to assist visually impaired individuals by generating detailed, spoken descriptions of images. A CNN-RNN architecture will be employed, where the CNN-based encoder extracts image features and the RNN-based decoder generates captions. These captions will then be converted into speech using a text-to-speech library.

This project combines deep learning with natural language processing and builds on the concepts from the paper *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* (https://arxiv.org/abs/1502.03044).

The Flickr8K dataset, sourced from Kaggle, consists of 8,000 images, each with five unique captions that describe key elements and actions within the images. The dataset can be found here: https://www.kaggle.com/adityajn105/flickr8k.