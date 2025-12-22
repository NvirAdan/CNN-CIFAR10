# CNN from Scratch for CIFAR-10 Classification (NumPy Only)

## Description

This project is a Convolutional Neural Network from scratch using only NumPy,trained and tested on the CIFAR-10 dataset.
The goal is to understand convolutional layers,their backpropagation and training without relying on deep learning frameworks.
The model is trained,tested and analyzed using standard classification metrics and visualizations.

## Why?

In this project I wanted to learn what is a convolution and how is the backpropagation of it from a deep perspective.
The focus is more in the learning aspect rather than the efficiency and accuracy of the model.
For that reason I choosed to do it with NumPy only and see what is the math and reasoning behind this type of models.
>[!NOTE]
>I've used PyTorch only for the charge of the dataset into the project and Matplotlib for the visualization of the metrics.

## FEATURES

-Convolutional Layers (Foward and Backward)\
-Maxpooling layers (Foward and Backward)\
-Fully connected layers\
-Relu Activation\
-Softmax and Cross entropy Loss\
-Weight initialization\
-Training loop with batching\
-Training checkpointing\
-Metrics tracking (Loss and Accuracy)\
-Visualization of Metrics

## MODEL ARCHITECTURE

Input (3x32x32)

->Conv2D(16 Kernels of 3x3)\
->ReLU\
->Conv2D(32 Kernels)\
->ReLU\
->MaxPool

->Conv2D(64 Kernels)\
->ReLU\
->Conv2D(128 Kernels)\
->ReLU\
->MaxPool

-> Flatten

->Fully Connected

->Softmax

## TRAINING DETAILS

Dataset = CIFAR-10\
Optimizer = SGD\
Learning rate = 0.01\
Batch_size = 64\
Epochs = 5 effective epochs of the 20 initially planned

The training was perfomed on CPU only reaching 5 effective epochs due to time reasons.
With an aproximate of 28 hours per epoch.

## PROBLEMS & SOLUTIONS

>[!NOTE]
>This are some problems that I've encountered while coding and learning and my reasoning and approach to them.
>Inside "CNN.ipynb" I did explain what the cell does but not the problem that I had to reach that conclusion.


### Theory Problems

The special way of manage the data in the convolutional part was the most complex thing on the entire project due to the new "movement" that now its presented in difference to the NN of MNIST.
This movement that continuously changes its location in the image (called "window/patch" inside the notebook) was very complex to compared to analogies, which I require to understand concepts.
At first my understanding of kernel was that they are "moving objects" that take information of tiny parts of the image like a scanner.
However I had to discard that approach because it wasn't 100% correct and brough me confusion later on.


The solution to this misinterpretation was divide the movement in two parts, first part was taking the window/patch of the image and seeing it as a "credit card" and inside of it an identification of a small region of the entire image and the second part,the kernel, seeing it as a payment terminal(the device where you swap your credit card) which instead of being in constant movement (the previous approach)now is static,so when the convolution happens, I interpret it as something similar to a payment but instead of accepting or declining your credit card it tells you if the pattern inside the kernel has activated or not.

Once understanding this concept after many hours of different approaches, the rest of the CNN was easier to connect including the backpropagation part.

-This made me learn how to separe in parts the difficulties in the Neural Networks in general and how to approch them more easily and efficient.

### Coding Problems

Here I had problems mostly with the training rather than the declaration of functions due to the simplicity of debugging each cell in jupyter notebook and the complexity to connecting all the cells in a process that makes sense.
Some issues were before start training, for example, I forgot that the ReLU function returns the tensor ReLU and Cache making a variable of type tuple later on by mistake making impossible to do backpropagation without fixing this issue(it also happened with the MaxPool function).And some problems were after the start of training,for example, I didn't know that the training will take so long (I was expecting less than 30 minutes like the NN of MNIST) for that reason at the beginning I didn't have the autosave per batch and only had the autosave per epoch, this is the reason why in the cell below the training I have a manual save of the weights.Later on in the project because this takes days to train in its current state, I leave the PC for some minutes and when I come back the monitor has turned off (The computer had entered the suspended mode)this makes the kernel of jupyter notebook change its state to unknown making me unable to continue normally,so I automatically restart it withour knowing that restarting the kernel I'll lose the metrics that were saved inside the variables,luckily the weights weren't drastically affected and once restarted the kernel I continued for the checkpoint of a couple of minutes ago, because of this I've changed the default configuration of the PC and now I'm more carefull before start training a model.

-Complications like this make you realize how importart is be careful and think some steps ahead because a simple error can cause the lose of hours of training, the computational cost needed when the model start to scale and the importance of optimization.


## RESULTS AND METRICS

Loss = 1.35

![Loss Visualization](/assets/lossgraph.png)

Accuracy = 0.52

![Accuracy Visualization](/assets/accuracygraph.png)

>[!NOTE]
>Inside the Notebook you can find more metrics and tests

Metrics every 50 batches

![Metrics per batch CIFAR10](/assets/lossperbatchcnncifar10.png)


While not having astonishing stadistics, the model proves to improve along the way of the epochs of the training,meaning that with more time the metrics will be better, this doesn't matter to the actual purpose of the project but it's important to know that it learned in the time it has trained.

## VISUALIZATION

Here are some results of the predictions of the test data after the training of the model

![Wrong Predictions](/assets/wrongpredicts.png)

![Correct Predictions](/assets/correctpredictions.png)

## HOW TO RUN 

### 1. Clone the Repository

```bash
git clone https://github.com/NvirAdan/CNN-CIFAR10
cd CNN-CIFAR10

```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv/Scripts/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the project

Open the CNN.ipynb notebook and run all the cells sequentially


## LICENSE
This project is released under the MIT License.
