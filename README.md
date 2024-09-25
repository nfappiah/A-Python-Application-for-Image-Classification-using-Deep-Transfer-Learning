# A Python Application for Image Classification using Deep Transfer Learning

In this project, I built a deep learning image classifier app capable of recognizing different species of flowers using PyTorch. The application consists of four python scripts that run from the command line and can be used for classifying any image dataset provided the dataset is in the right format.

## Dataset

The dataset, consisting of 102 flower categories, I used in building the app can be found at: 
[https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

## Methods

The methods employed in this project involve:
* Loading and preprocessing the images
* Training the deep learning image classifier on the images
* Evaluate the image classifier
* Using the trained classifier to predict new images

## Deep Learning Architectures and Transfer Learning

Transfer learning is the process of taking a a pre-trained network and adapting it to a new dataset. The app gives the option to choose between two deep learning architectures - vgg or alexnet. In both cases, I defined a new, untrained feed-forward neural network as a classifier head, using ReLU activations and dropout while all other parameters in the backbone are kept frozen. I then trained the classifier layers on the new dataset and tracked the loss and accuracy on the validation set to determine the best hyperparameters. I then evaluated the models using the test set. After model evaluation, I save the model and then use it to perform prediction for a single input image - simulating the model in production.

## Python Scripts

The python scripts comprising the image classifier app are train.py, my_model.py, predict.py, and my_model_predictions.py. train.py is the main script for training, evaluating, and saving the evaluated model as a checkpoint. my_model.py is a script that contains all the functions and modules necessary for the execution of train.py. Similarly, predict.py is the main script for performing prediction or inference on a single input image using the trained model.It first loads the saved model checkpoint before doing inference. my_model_predictions.py is a script that contains all the functions and modules necessary for the execution of predict.py.

## Python Scripts Availability

I do not make the python scripts available in this repository. If you are interested in the scripts and would like to use or run them on a custom dataset, please contact me personally on fogne38@gmail.com

## Materials and Packages

* Command line
* Python 3.8.8
* matplotlib
* torch 2.0.1
* torchvision
* collections
* PIL
* numpy
* random
* json
* os

## GPU

The notebook is set up to use GPU if it is available.

## Instructions 

##### To train a new network on the dataset with train.py
* _Basic usage:_ python train.py data_directory
* Prints out training loss, validation loss, and validation accuracy as the network trains.
##### Options: 
* _Set directory to save checkpoints:_ python train.py data_dir --save_dir save_directory
* _Choose architecture:_ python train.py data_dir --arch "vgg13"
* _Set hyperparameters:_ python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
* _Use GPU for training:_python train.py data_dir --gpu
##### To Predict flower name from an image with predict.py along with the probability of that name. 
* Pass in a single image "/path/to/image" and return the flower name and class probability.
* _Basic usage:_ python predict.py /path/to/image checkpoint
##### Options:
* _Return top K most likely classes:_ python predict.py input checkpoint --top_k 3
* _Use a mapping of categories to real names:_ python predict.py input checkpoint --category_names cat_to_name.json
* _Use GPU for inference:_ python predict.py input checkpoint --gpu
##### Others
* The flower directory must contain 3 sub directories: test, train, and valid. Each of the test, train and valid sub directories must contain further sub directories corresponding to the image categories. Each image category sub directory must contain images only belonging to that category. It is vital that the dataset is set up in this way for notebook to run.
* Users must also have the 'cat_to_name.json' file in the working directory. The 'cat_to_name.json' file is a dictionary that maps category label (number or digit) to category (flower) name. If you would like to run this app, you must generate this file. The categories can be found in the link to dataset provided above.


## Acknowledgements

I am grateful to AWS for awarding me the AWS AI & Machine Learning Scholarhip to pursue the nanodegree in "AI Programming with Python" through Udacity. This classifier app represents the second of two final projects I undertook as part of the requirements to complete the nanodegree.