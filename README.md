# 02466-Fagpakkeprojekt-Point-clouds
This GitHub repository for course: 02466 | authors: s214596 , s214638 and s214609


This repository is for the 02466 Project work - Bachelor of Artificial Intelligence and Data course.
___
This project aims to answer the questions:
*  To what extent will the use of a cycleGAN-model be able to generate visually convinc-
ing results for geometric style transfer on point clouds?
*  How does the model, using four different input shapes (plane, sphere, gaussian distri-
bution, and a self learned feature shape) in the generator, compare in terms of their
effectiveness in achieving geometric style transfer for point clouds?


This repository is built of using the PyTorch framework. The main file for this project to carry out experiments is the train.py file. Running this file will start the training of the cycleGAN, save the models and start a weights and biases pipeline to follow the training. A main config file 'config.py' is used to configure the arguments such as start_shape, epochs, data dirs and hyperparameters for the cycleGAN.

All dependencies required to run the code can be installed with the requirement.txt file with the following code:

`python3 -m pip install -r requirements.txt`

For the final results produced in the paper, we used the following config settings:
~~~
SAMPLE_POINTS = 2048
DECODE_M = 2025
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LAMBDA_CYCLE = 1100
NUM_WORKERS = 7
NUM_EPOCHS = 1201
save_pointclouds = 20  
~~~

## Data
___
The data used in this project is from a paper about Semantic Parametric Reshaping of Human Body Models and
the data can be found from the following link:
https://users.soe.ucsc.edu/~davis/papers/2014_3DV_SemanticBodyModels.pdf

The data folder contains all the data split into 3 different folder,
'dummy', 'train' and 'val'. Each folder is split into male and female meshes. The dummy dataset contains 10 of each point cloud domain to run experiments that are less computational heavy.
The train folder contains around 1417 male- and 1431 female meshes. The val folder contains 100 of each domain.

## Experiments
___
This project carried out 4 different experiments. 
4 different starting shape for FoldingNet to fold upon.
[plane, sphere, gaussian, feature_shape]

To carry out the experiments dependent on which shape is run, the 'START_SHAPE' argument in the config.py file can be changed to one of the following four:
~~~
START_SHAPE = "plane"
START_SHAPE = "sphere"
START_SHAPE = "gaussian"
START_SHAPE = "feature_shape"
~~~
and then run the code to start the cycleGAN training. This will intitialize a wandb run as well with a project name corresponding to what starting shape is being used.
~~~
python3 train.py
~~~

## Credits
___
We want to thank the following people for their great contributions to this project:

Johan Ziruo Ye (ziruoye@dtu.dk)