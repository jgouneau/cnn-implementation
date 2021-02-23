import numpy as np
import os
import pickle
from scipy.misc import imread, imsave, imresize
from math import exp, log, sqrt
from random import randint
import time


class Network:
    """
            NETWORK
    - layers : liste ou tableau de couches de neurones
    - depth : le nombre de couches
    - categories : classes prises en compte par le réseau
    > train : fonction train sur un tableau d'images et de labels
    > training : effectue train sur une série d'images
    > test : fonction test sur une image (ou un tableau d'images)
    """
    
    def __init__(self, name, B, layers_list, cat):
        self.id = name
        n = len(layers_list)
        self.depth = n
        self.batch_size = B
        self.layers = []

        # LAYERS
        Type, param = layers_list[0]
        self.layers.append(Input(B, param))
        for i in range(1, n):
            Type, param = layers_list[i]
            self.layers.append(Type(param, self.layers[i-1].size))
        FullyConnected.ident = 0
        Convolutionnal.ident = 0
        Pooling.ident = 0

        # CATEGORIES
        self.categories = [ "" for i in range(np.size(self.layers[n-1].output[0]))]
        for i in range(min(len(cat), len(self.categories))):
            self.categories[i] = cat[i]


    def define(self):
        print("Network : " + self.id)
        print("Depth : " + str(self.depth))
        print("Categories : " + str(self.categories), end = '\n\n')
        for i in range(self.depth):
            if type(self.layers[i]) == FullyConnected or type(self.layers[i]) == Convolutionnal:
                self.layers[i].define(self.layers[i+1])
            if type(self.layers[i]) == Pooling or type(self.layers[i]) == Normal:
                self.layers[i].define()
    

    def train(self, example):
        n = self.depth
        img, label = example
        self.layers[0].update(img)

        # propagation avant
        print("\n----------------------------------- FRONT : ------------------------------------------------")
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)

        # calcul de l'erreur
        self.layers[n-1].loss(label)

        # rétropropagation
        print("\n----------------------------------- BACK : -------------------------------------------------")
        for i in range(1, n):
            self.layers[n-i].back(self.layers[n-i-1])


        # retour console
        print("\n----------------------------------- RESULT : -----------------------------------------------")
        out = self.layers[n-1].output
        # for c in range(len(self.categories)):
        print("\nExpect : ")
        print(label)
        print("\nOutput : ")
        print(out)
        print(self.layers[n-1].error, end = "\n\n")
    

    def training(self, n, setup):
    	B = self.batch_size
    	img, labels = setup
    	for i in range(n) :
            print("\n============================================= BATCH : " + str(i) + " =============================================")
            ex = (img[i*B : (i+1)*B], labels[i*B : (i+1)*B])
            self.train(ex)


    def train_batch(self, example):
        n = self.depth
        img, label = example
        self.layers[0].update(img)

        # propagation avant
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)

        # calcul de l'erreur
        self.layers[n-1].loss(label)

        # rétropropagation
        print("\n----------------------------------- BACK : -------------------------------------------------")
        for i in range(1, n):
            self.layers[n-i].back(self.layers[n-i-1])


        # retour console
        print("\n----------------------------------- RESULT : -----------------------------------------------")
        for b in range(self.batch_size):
        	out = np.reshape(self.layers[n-1].output[b], len(self.categories))
        	print("\n")
	        for c in range(len(self.categories)):
	            print("[" + str(label[b, c]), end = "] [")
	            print(str(out[c]) + "]")
        print(np.sum(self.layers[n-1].error)/self.batch_size, end = "\n\n")


    def training_CIFAR(self, set_nb, dist):
    	B = self.batch_size
    	loaded = np.load("Data/batch_" + str(set_nb) + ".npz")
    	features, labels = loaded["f"], loaded["l"]
    	start, end = dist
    	if end > 10000 :
    		end = 10000
    	end = int((end-start)/B + start)
    	for i in range(start, end):
            print("\n============================================= SET : " + str(set_nb) + " - BATCH : " + str(i*B) + " =============================================")
            ex = (features[i*B : (i+1)*B], labels[i*B : (i+1)*B])
            self.train_batch(ex)


    def tempo(self):
    	for l in self.layers :
    		print(l.id + " : " + str(l.tempo))
    		l.tempo = 0


    def test(self, image):
        n = self.depth
        self.layers[0].update(image)
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)
        print(self.layers[n-1].output)



class Input:
    """
            INPUT
    + s'occupe de la transition entre le réseau et les donneés
    - size : tuple des dimensions (D, H, W)
    - output : tableau des valeurs d'entrée
    > update : fait prendre à output une nouvelle valeur
    """
    def  __init__(self, B, parameters):
         self.id = "Input"
         self.rank = 0
         D, H, W = parameters
         self.size = B, D, H, W
         self.output = np.zeros(self.size)
         self.term = np.zeros(self.size)
         self.tempo = [0]

    def define(self):
        print(self.id, end = ' : ')
        print(self.size)

    def update(self, out):
        start = time.clock()
        self.output = out
        self.tempo[0] += time.clock() - start


class FullyConnected:

    ident = 0

    def  __init__(self, parameters, prev_size): 
        FullyConnected.ident += 1
        D, H, W, rate = parameters
        B, D1, H1, W1 = prev_size
        self.id = "FullyConnected_n°" + str(FullyConnected.ident)
        self.size = (B, D, H, W)
        self.weigths = np.random.randn(D, H, W, D1, H1, W1) * 0.01 / sqrt(D1*H1*W1)
        self.delta = np.zeros((D, H, W, D1, H1, W1))
        self.bias = np.random.randn(D, H, W) * 0.01 / sqrt(D1*H1*W1)
        self.delta_bias = np.zeros((D, H, W))
        self.distrib = np.ones((B))
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.speed, self.moment, self.white = rate
        self.tempo = [0, 0]

    def define(self, next_layer):
        print(self.id + " - " + next_layer.id, end = " : ")
        print(self.size)

    def front(self, inp):
        start = time.clock()
        self.output = np.transpose(np.einsum('bijk,...ijk', inp, self.weigths), axes = [3, 0, 1, 2]) + np.einsum('dhw,...', self.bias, self.distrib)
        # sortie :
        print("\n" + self.id)
        print("    out max : " + str(np.max(self.output)))
        print("        min : " + str(np.min(self.output)))
        print("weigths max : " + str(np.max(self.weigths)))
        print("        min : " + str(np.min(self.weigths)))
        print("   bias max : " + str(np.max(self.bias)))
        print("        min : " + str(np.min(self.bias)))
        self.tempo[0] += time.clock() - start

    def back(self, prev_layer):
        start = time.clock()
        B, D, H, W = self.size

        # terme d'erreur :
        prev_layer.term = np.transpose(np.einsum('bdhw,dhw...', self.term, self.weigths), axes = [3, 0, 1, 2])

        # biais :
                           # part d'erreur									# partie inertielle
        self.delta_bias = (self.speed / B * np.sum(self.term, axis = 0)) + (self.moment*self.delta_bias)

        self.bias += self.delta_bias

        # poids :
                                            # part d'erreur								# white-decay			    # partie inertielle
        self.delta = self.speed / B * (np.einsum('bijk,b...', prev_layer.output, self.term) + self.white*self.weigths) + (self.moment*self.delta)
        self.weigths += self.delta

        # sortie :
        print("\n" + self.id)
        print("      term max : " + str(np.max(self.term)))
        print("           min : " + str(np.min(self.term)))
        print("     delta max : " + str(np.max(self.delta)))
        print("           min : " + str(np.min(self.delta)))
        print("delta_bias max : " + str(np.max(self.delta_bias)))
        print("           min : " + str(np.min(self.delta_bias)))
        self.tempo[1] += time.clock() - start


class Convolutionnal:
    """
            CONVOLUTIONNAL
    - size : tuple des dimensions (D, H, W)
    - weights : tableau des poids de taille (D, D1, F, F)
    - delta : tableau des ajustements précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    - speed, moment : vitesse d'apprentissage et inertie
    - hyper : (F, S, P) / F : taille du filtre, S : stride (décalage), P : zero-padding
    > init((D2, F, S, P, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """

    ident = 0

    def  __init__(self, parameters, prev_size):
        Convolutionnal.ident += 1
        self.id = "Convolutionnal_n°" + str(Convolutionnal.ident)
        B, D1, H1, W1 = prev_size
        D2, F, S, P, rate = parameters
        if P == -1 and S == 1 :
            P = int((F - 1)/2)
        self.hyper = F, S, P
        H2 = int(((H1 - F + 2*P) // S) + 1)
        W2 = int(((W1 - F + 2*P) // S) + 1)
        self.size = B, D2, H2, W2
        self.weigths = np.random.randn(D2, D1, F, F) / sqrt(D1*H1*W1)
        self.delta = np.zeros((D2, D1, F, F))
        self.bias = np.ones((D2)) / sqrt(D1*H1*W1)
        self.delta_bias = np.zeros((D2))
        self.distrib = np.ones((B))
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.speed, self.moment, self.white = rate
        self.tempo = [0, 0]

    def define(self, next_layer):
        print(self.id + " - " + next_layer.id, end = " : ")
        print(self.size)
        F, S, P = self.hyper
        print("     Weigths : " + str(np.shape(self.weigths)))
        print("     Term : " + str(np.shape(self.term)))
        print("     Output : " + str(np.shape(self.output)))
        print("     Filter : " + str(F))
        print("     Stride : " + str(S))
        print("     Padding : " + str(P))

    def front(self, inp):
        start = time.clock()
        B, D1, H1, W1 = inp.shape
        B, D, H, W = self.size
        F, S, P = self.hyper
        padded_inp = np.zeros((B, D1, H1 + 2*P, W1 + 2*P))
        padded_inp[:, :, P:H1+P, P:W1+P] = inp
        for h in range(H):
            for w in range(W):
                self.output[:, :, h, w] = np.transpose(np.einsum('...ijk,bijk', self.weigths, padded_inp[:, :, h*S:h*S+F, w*S:w*S+F])) + np.einsum('d,...', self.bias, self.distrib)
        # sortie :
        print("\n" + self.id)
        print("    out max : " + str(np.max(self.output)))
        print("        min : " + str(np.min(self.output)))
        print("weigths max : " + str(np.max(self.weigths)))
        print("        min : " + str(np.min(self.weigths)))
        print("   bias max : " + str(np.max(self.bias)))
        print("        min : " + str(np.min(self.bias)))
        self.tempo[0] += time.clock() - start

    def back(self, prev_layer):
        start = time.clock()
        B, D, H, W = self.size
        B, D1, H1, W1 = prev_layer.size
        F, S, P = self.hyper
        delta = np.zeros(np.shape(self.delta))
        padded_prev_out = np.zeros((B, D1, H1 + 2*P, W1 + 2*P))
        padded_prev_out[:, :, P:H1+P, P:W1+P] = prev_layer.output
        padded_prev_term = np.zeros((B, D1, H1 + 2*P, W1 + 2*P))

        # biais :
                           							# part d'erreur						  # partie inertielle
        self.delta_bias = (self.speed / (H*W*B) * np.sum(self.term, axis = (3, 2, 0))) + (self.moment*self.delta_bias)

        for h in range(H):
            for w in range(W):
                    # on envoie le terme d'erreur en le pondérant dans le neurone précedent
                padded_prev_term[:, :, h*S:h*S+F, w*S:w*S+F] += np.transpose(np.einsum('bd..., d...', self.term[:, :, h, w], self.weigths), axes = [3, 0, 1, 2])
                	# part d'erreur
                delta += np.einsum('bijk,b...', padded_prev_out[:, :, h*S:h*S+F, w*S:w*S+F], self.term[:, :, h, w])

        # poids :
                      				# part d'erreur			 # partie inertielle          # white-decay
        self.delta = self.speed * ((1 / (H*W*B) * delta)  - (self.white*self.weigths)) + (self.moment*self.delta)
        
        self.weigths += self.delta
        self.bias += self.delta_bias

        	# on envoie le terme d'erreur en le pondérant dans le neurone précedent
        prev_layer.term = padded_prev_term[:, :, P:H1+P, P:W1+P]

        # sortie :
        print("\n" + self.id)
        print("      term max : " + str(np.max(self.term)))
        print("           min : " + str(np.min(self.term)))
        print("     delta max : " + str(np.max(self.delta)))
        print("           min : " + str(np.min(self.delta)))
        print("delta_bias max : " + str(np.max(self.delta_bias)))
        print("           min : " + str(np.min(self.delta_bias)))
        self.tempo[1] += time.clock() - start


class Pooling:
    """
            POOLING
    - size : tuple des dimensions (D, H, W)
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de sortie
    - hyper : (F, S) / F : taille du filtre, S : stride (décalage)
    > init((F, S), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0

    def  __init__(self, parameters, prev_size):
        Pooling.ident += 1
        self.id = "Pooling_n°" + str(Pooling.ident)
        B, D1, H1, W1 = prev_size
        F, S = parameters
        self.hyper = parameters
        H2 = int(((H1 - F) // S) + 1)
        W2 = int(((W1 - F) // S) + 1)
        self.size = B, D1, H2, W2
        self.distrib = np.ones((F, F))
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.tempo = [0, 0]

    def define(self):
        print(self.id, end=' : ')
        print(self.size)

    def front(self, inp):
        start = time.clock()
        B, D, H, W = self.size
        F, S = self.hyper
        for h in range(H):
        	for w in range(W):
        		self.output[:, :, h, w] = np.max(np.max(inp[:, :, h*S:h*S+F, w*S:w*S+F], axis = 2), axis = 2)
        # sortie :
        print("\n" + self.id)
        self.tempo[0] += time.clock() - start

    def back(self, prev_layer):
        start = time.clock()
        B, D, H, W = self.size
        F, S = self.hyper
        prev_layer.term = np.zeros(np.shape(prev_layer.term))
        ones = np.ones((F, F))
        
        for h in range(H):
            for w in range(W):
                M = prev_layer.output[:, :, h*S:h*S+F, w*S:w*S+F] - np.transpose(np.einsum('b...,ij' , self.output[:, :, h, w], self.distrib), axes = [1, 0, 3, 2])
                M[M == 0] = 1
                M[M != 1] = 0
                prev_layer.term[:, :, h*S:h*S+F, w*S:w*S+F] += np.einsum('...,...ij', self.term[:, :, h, w], M)
        # sortie :
        print("\n" + self.id)
        self.tempo[1] += time.clock() - start


class Sigmoid:

    def  __init__(self, parameters, prev_size):
        self.id = "Sigmoid"
        self.size = prev_size
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.error = 0
        self.tempo = [0, 0, 0]

    def front(self, inp):
        start = time.clock()
        # sigmoid = lambda x : 1 / (1 + exp(-x))
        def sigmoid(array):
            return 1 / (1 + np.exp(-array))
        self.output = sigmoid(inp)
        self.tempo[0] += time.clock() - start

    def back(self, prev_layer):
        start = time.clock()
        # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
        prev_layer.term = self.term * self.output * (1 - self.output)
        self.tempo[1] += time.clock() - start

    def loss(self, expect):
        start = time.clock()
        out = np.copy(np.reshape(self.output, np.shape(expect)))
        err = expect - out
        self.term = np.reshape(err, np.shape(self.term))
        self.error = np.sum(err**2)/2
        self.tempo[2] += time.clock() - start


class Relu:

    def  __init__(self, parameters, prev_size):
        self.id = "ReLU"
        self.size = prev_size
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.error = 0
        self.tempo = [0, 0, 0]

    def front(self, inp):
        start = time.clock()
        def ReLU(array):
            zero = np.zeros(np.shape(array))
            return np.maximum(zero, array) + 0.01*np.minimum(zero, array)
        self.output = ReLU(inp)
        self.tempo[0] += time.clock() - start

    def back(self, prev_layer):
        start = time.clock()
        # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
        M = np.copy(self.output)
        M[M >= 0] = 1
        M[M < 0] = 0.01
        prev_layer.term = self.term * M
        self.tempo[1] += time.clock() - start

    def loss(self, expect):
        start = time.clock()
        out = np.copy(np.reshape(self.output, np.shape(expect)))
        err = expect - out
        self.term = np.reshape(err, np.shape(self.term))
        self.error = np.sum(err**2)/2
        self.tempo[2] += time.clock() - start


class Softmax:
    """
            SOFTMAX
    - size : tuple des dimensions (D, H, W)
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    > init((D2, H2, W2, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    def  __init__(self, param, prev_size):
        self.id = "Softmax"
        self.size = prev_size
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.error = 0
        self.tempo = [0, 0, 0]

    def front (self, inp):
        start = time.clock()
        B, D, H, W = self.size
        def soft(array):
        	array -= np.max(array)
        	e = np.exp(array)
        	return e / np.sum(e)
        for b in range(B):
        	self.output[b] = soft(inp[b])
        self.tempo[0] += time.clock() - start

    def back (self, prev_layer):
        start = time.clock()
        prev_layer.term = self.term
        self.tempo[1] += time.clock() - start

    def loss (self, expect):
        start = time.clock()
        B, D, H, W = self.size
        out = np.copy(np.reshape(self.output, np.shape(expect)))
        err = expect - out
        self.term = np.reshape(err, np.shape(self.term))
        self.error = - np.stack([np.sum(expect[b]*np.log(out[b])) for b in range(B)])
        self.tempo[2] += time.clock() - start



def saveArrays(net):
    if not (net.id + "_save") in os.listdir("."):
        os.mkdir(net.id + "_save")
    for l in net.layers :
        if type(l) == FullyConnected or type(l) == Convolutionnal:
            np.savez_compressed(net.id + "_save/" + l.id, w = l.weigths, d = l.delta, b = l.bias, db = l.delta_bias)


def loadArrays(net):
    for l in net.layers :
        if type(l) == FullyConnected or type(l) == Convolutionnal:
            loaded = np.load(net.id + "_save/" + l.id + ".npz")
            l.weigths = loaded["w"]
            l.delta = loaded["d"]
            l.bias = loaded["b"]
            l.delta_bias = loaded["db"]




class Normal:

    ident = 0

    def __init__(self, param, prev_size):
        self.id = "Normal"
        self.size = prev_size
        self.hyper = param
        self.sum = 0
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)

    def define(self):
        print(self.id, end=' : ')
        print(self.size)

    def front(self, inp):
    	min_val = np.min(inp)
    	max_val = np.max(inp)
    	self.norm = max_val - min_val
    	self.output = (inp - min_val) / (max_val - min_val)

    def back(self, prev_layer):
        prev_layer.term = self.term / self.norm






"""
_________________________________________________________________________________________
_________________________________________________________________________________________

                                        TESTS
_________________________________________________________________________________________
_________________________________________________________________________________________

"""


def Exemples():
    lay_list = [(Input, (3, 32, 32)),
                (ConvSigmoid, (4, 5, 1, 0, (1, 0.9))),
                (ConvSigmoid, (5, 5, 1, 0, (1, 0.9))),
                (FullySigmoid, (5, 1, 1, (1, 0.9, 0.0005)))]
    
    cat_list = ["dog", "cat", "car", "boat", "spoon"]
    
    net = Network(lay_list, cat_list)
    net.define()
    return "Done"


def Test_FC():
    d, h, w = 3, 16, 16
    volume = (d, h, w)
    
    lay_list = [(Input, volume),
                (FullyMax, (1, 1, 6, (0.01, 0.3, 0.))),
                (FullyMax, (1, 1, 2, (0.01, 0.3, 0.)))]
    cat_list = [0, 1]
    
    net = Network("Volume", lay_list, cat_list)
    net.define()
    

    img1 = np.ones(volume)
    img2 = np.ones(volume)
    img3 = np.ones(volume)
    img4 = np.ones(volume)
    
    for j in range(d):
    	for i in range(h*w):
    	    img1[j, randint(0, h-1), randint(0, w-1)] = 0.1
    	for i in range(int(h*w*0.75)):
    	    img2[j, randint(0, h-1), randint(0, w-1)] = 0.1
    	for i in range(int(h*w*0.50)):
    	    img3[j, randint(0, h-1), randint(0, w-1)] = 0.1
    	for i in range(int(h*w*0.25)):
    	    img4[j, randint(0, h-1), randint(0, w-1)] = 0.1

    print(img1)
    
    ex1 = (img1, np.array([[[0.1, 0.1]]]))
    ex2 = (img2, np.array([[[0.1, 1]]]))
    ex3 = (img3, np.array([[[1, 0.1]]]))
    ex4 = (img4, np.array([[[1, 1]]]))
    
    setup = []
    for i in range(100):
        setup.append(ex1)
        setup.append(ex2)
        setup.append(ex3)
        setup.append(ex4)

    net.training(setup)
    return "Done"


def Test_Deep_FC():
    lay_list = [(Input, (1, 1, 5)),
    			(FullyConnected, (1, 1, 5, (0.1, 0.9, 0.))), (Sigmoid, 0),
                (FullyConnected, (1, 1, 5, (0.1, 0.9, 0.))), (Sigmoid, 0)]
    
    cat_list = ["0", "1", "2", "3", "4"]
    
    ex1 = (np.array([[[0.1, 1, 1, 1, 0.1]]]), np.array([1, 0.1, 0.1, 1, 0.1]))
    ex2 = (np.array([[[1, 0.1, 0.1, 1, 1]]]), np.array([0.1, 1, 0.1, 0.1, 1]))
    ex3 = (np.array([[[1, 1, 1, 0.1, 1]]]), np.array([0, 1, 1, 0, 1]))
    ex4 = (np.array([[[0.1, 1, 0, 1, 0.1]]]), np.array([0.1, 0.1, 1, 1, 0.1]))
    
    setup = []
    for k in range(1000):
        setup.append(ex1)
        setup.append(ex2)
        setup.append(ex3)
        setup.append(ex4)
    
    net = Network("Test_Deep_FC", lay_list, cat_list)
    net.training(setup)

    net = Network("Test_Deep_FC", lay_list, cat_list)
    return "Done"


def Test_FC_And():
    lay_list = [(Input, (2, 1, 1)), 
                (FullyMax, (1, 1, 1, (0.1, 0.9, 0.0005)))]
    cat_list = ["True", "False"]
    
    ex1 = (np.array([[[0.]], [[0.]]]), np.array([[[0.]]]))
    ex2 = (np.array([[[0.]], [[1.]]]), np.array([[[0.]]]))
    ex3 = (np.array([[[1.]], [[0.]]]), np.array([[[0.]]]))
    ex4 = (np.array([[[1.]], [[1.]]]), np.array([[[1.]]]))
    
    setup = []
    for i in range(400):
        setup.append(ex1)
        setup.append(ex2)
        setup.append(ex3)
        setup.append(ex4)

    net = Network("And", lay_list, cat_list)
    net.training(setup)
    return "Done"


def Test_Conv():
    d, h, w = 3, 16, 16
    volume = (d, h, w)

    lay_list = [(Input, volume),
                (ConvMax, (5, 5, 1, -1, (0.1, 0.9, 0.0005))),
                (Pooling, (2, 2)),
                (FullyMax, (4, 1, 1, (0.1, 0.9, 0.0005)))]
    cat_list = ["True", "False"]
    
    net = Network("Conv_&_Pool", lay_list, cat_list)
    net.define()
    
    img1 = np.zeros(volume)
    img2 = np.zeros(volume)
    img3 = np.zeros(volume)
    img4 = np.zeros(volume)
    
    for j in range(d):
    	for i in range(h*w):
    	    img1[j, randint(0, h-1), randint(0, w-1)] = randint(0, 10)*0.1
    	for i in range(int(h*w*0.75)):
    	    img2[j, randint(0, h-1), randint(0, w-1)] = randint(0, 10)*0.1
    	for i in range(int(h*w*0.50)):
    	    img3[j, randint(0, h-1), randint(0, w-1)] = randint(0, 10)*0.1
    	for i in range(int(h*w*0.25)):
    	    img4[j, randint(0, h-1), randint(0, w-1)] = randint(0, 10)*0.1
    
    ex1 = (img1, np.array([[[0.]], [[1.]], [[0.]], [[0.]]]))
    ex2 = (img2, np.array([[[0.]], [[0.]], [[0.]], [[1.]]]))
    ex3 = (img3, np.array([[[0.]], [[1.]], [[1.]], [[0.]]]))
    ex4 = (img4, np.array([[[1.]], [[1.]], [[0.]], [[1.]]]))
    
    setup = []
    for i in range(100):
    	setup.append(ex1)
    	setup.append(ex2)
    	setup.append(ex3)
    	setup.append(ex4)
    net.training(setup)
    return "Done"


def Test_Deep_Conv(N):
    lay_list = [(Input, (3, 32, 32)),
                (Convolutionnal, (5, 5, 1, -1, (0.01, 0.9, 0.0005))), (Relu, 0),
                (Pooling, (2, 2)),
                (Convolutionnal, (5, 5, 1, -1, (0.01, 0.9, 0.0005))), (Relu, 0),
                (Pooling, (2, 2)),
                (Convolutionnal, (5, 5, 1, -1, (0.01, 0.9, 0.0005))), (Relu, 0),
                (FullyConnected, (4, 1, 1, (0.01, 0.9, 0.0005))), (Softmax, 0)]
    cat_list = ["chien", "chat", "pelle", "table"]

    d, h, w = 3, 32, 32
    volume = (4, d, h, w)
    
    img = np.random.randint(0, 255, volume)
    label = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    
    imgs = np.zeros((4*N, 3, 32, 32))
    labels = np.zeros((4*N, 4))
    for i in range(N):
    	for j in range(i*4, (i+1)*4):
    		r = randint(0, 3)
	    	imgs[j] = img[r]
	    	labels[j] = label[r]
    setup = (imgs, labels)

    net = Network("Deep_Conv", 4, lay_list, cat_list)
    net.define()
    net.training(N, setup)

    return "Done"



def The_Grand_Test():
    lay_list = [(Input, (3, 32, 32)),
                (Convolutionnal, (16, 5, 1, 2, (0.05, 0.9, 0.0005))), (Relu, 0),
                (Pooling, (2, 2)),
                (Convolutionnal, (20, 5, 1, 2, (0.05, 0.9, 0.0005))), (Relu, 0),
                (Pooling, (2, 2)),
                (Convolutionnal, (20, 5, 1, 2, (0.05, 0.9, 0.0005))), (Relu, 0),
                (Pooling, (2, 2)),
                (FullyConnected, (1, 1, 10, (0.05, 0.9, 0.0005))), (Softmax, 0)]

    cat_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    net = Network("The_Grand_Test", 4, lay_list, cat_list)
    net.define()
    start = time.clock()
    net.training_CIFAR(1, (0, 1000))
    print(time.clock() - start)
    net.tempo()


"""

with open("CIFAR-10/cifar-10-batches-py/data_batch_1", 'rb') as fo:
        	batch = pickle.load(fo, encoding='bytes')
for k in range(10000):
    img = np.zeros((32, 32, 3))
    for d in range(3):
    	for i in range(32):
    			img[i, :, d] = batch[b'data'][k, 1024*d + 32*i : 1024*d + 32*(i+1)]
    expect = np.zeros((10))
    expect[batch[b'labels'][k]] = 1
    ex = (img, expect)
    imsave("Images/img" + str(k) + ".jpg", img)

"""

The_Grand_Test()