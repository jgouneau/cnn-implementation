# cnn-implementation

This is an implementation of a Convolutional Neural Network done in Python, which allows to create, train and test a CNN whith any architecture over any dataset and scripts to preprocess its datas.
This was my first big coding project, done in year 2018/2019 for my studies, it is quite messy :

### /doc/
`/doc/Diapo_TIPE.pdf` : the final presentation which was shown to the jury, it contains all the results of this implementation (to sum up : a .7 accuracy and .6 loss on the CIFAR-10 in 10 epochs).
`MCOT.pdf` : the abstract.

### /src/

`/src/Network.py` :
The core file of the code, here the network creation, executtion and testing fonctions are implemented ;
  `Network` : implements the network as a pile of layers, the function of train over the whole dataset and of estimation of a pixel responsability in the output.
  *LAYERS IMPLEMENTATIONS :*
  `FullyConnected` : implements a fully-connected neural layer
  `Convolutional` : implements a convolutional neural layer
  `Pooling` : implements a pooling layer
  *ACTIVATION AND OUTPUT IMPLEMENTATIONS :*
  `Sigmoid` : implements the sigmoid activation function of a layer
  `Relu` : implements the ReLU activation function of a layer
  `Softmax` : implements the softmax activation function of a neural network
  *OTHER FUNCTIONS :*
  `Train_CIFAR()` : a function which creates/load a LeNet-5 based network and trains it on one loop of the whole dataset
  `Test_Response()`, `print_error()` : functions to evaluate the quality of the network over the test dataset
  
  `/src/Preprocessing.py` :
   Here are the scripts to preprocess the data (normalize, center, reshape).
   
   `src/hist` :
   Contains the whole evolution of the `Network.py` script.
