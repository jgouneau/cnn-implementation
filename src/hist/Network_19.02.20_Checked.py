import numpy as np
import os
import pickle
from scipy.misc import imread, imsave, imresize
from math import exp, log, sqrt
from random import randint


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
    
    def __init__(self, title, layers_list, cat):
        self.name = title
        n = len(layers_list)
        self.depth = n
        self.layers = []

        # INPUT & LAYERS
        Type, param = layers_list[0]
        self.layers.append(Input(param))
        for i in range(1, n):
            Type, param = layers_list[i]
            self.layers.append(Type(i, param, self.layers[i-1].size))
        FullySigmoid.ident = 0
        FullyMax.ident = 0
        ConvSigmoid.ident = 0
        ConvMax.ident = 0
        Pooling.ident = 0

        # CATEGORIES
        self.categories = [ "" for i in range(np.size(self.layers[n-1].output))]
        for i in range(min(len(cat), len(self.categories))):
            self.categories[i] = cat[i]


    def define(self):
        print("Depth : " + str(self.depth))
        print("Categories : " + str(self.categories), end = '\n\n')
        for l in self.layers :
            l.define()
    

    def train(self, loss, example):
        n = self.depth
        image, expect = example
        self.layers[0].update(image)

        # propagation avant
        print("\n----------------------------------- FRONT : ------------------------------------------------")
        for i in range(1, n):
            print("\n" + str(self.layers[i].id))
            self.layers[i].front(self.layers[i-1].output)
            print("    out max : " + str(np.max(self.layers[i].output)))
            print("        min : " + str(np.min(self.layers[i].output)))
            if type(self.layers[i]) != Input and type(self.layers[i]) != Pooling and type(self.layers[i]) != Normal and type(self.layers[i]) != Softmax :
                print("weigths max : " + str(np.max(self.layers[i].weigths)))
                print("        min : " + str(np.min(self.layers[i].weigths)))
                print("   bias max : " + str(np.max(self.layers[i].bias)))
                print("        min : " + str(np.min(self.layers[i].bias)))
                print(" parameters : " + str(np.size(self.layers[i].weigths)))
            elif type(self.layers[i]) == Normal :
                print("      norme : " + str(self.layers[i].norm))
        out = np.copy(np.reshape(self.layers[n-1].output, len(self.categories)))

        # calcul de l'erreur
        if loss == "basic_loss" :
            err = expect - out
            self.layers[n-1].term = np.reshape(err, np.shape(self.layers[n-1].term))
            error = np.sum(err**2)/2

        elif loss == "cross_entropy_loss" :
            #*
            0


        # rétropropagation
        print("\n----------------------------------- BACK : -------------------------------------------------")
        for i in range(1, n):
            self.layers[n-i].back(self.layers[n-i-1])
            if type(self.layers[n-i]) != Input and type(self.layers[n-i]) != Pooling and type(self.layers[n-i]) != Normal and type(self.layers[n-i]) != Softmax :
                print("\n" + str(self.layers[n-i].id))
                print("      term max : " + str(np.max(self.layers[n-i].term)))
                print("           min : " + str(np.min(self.layers[n-i].term)))
                print("     delta max : " + str(np.max(self.layers[n-i].delta)))
                print("           min : " + str(np.min(self.layers[n-i].delta)))
                print("delta_bias max : " + str(np.max(self.layers[n-i].delta_bias)))
                print("           min : " + str(np.min(self.layers[n-i].delta_bias)))
            if type(self.layers[n-i]) == Normal :
                print("\n" + str(self.layers[n-i].id))
                print("      term max : " + str(np.max(self.layers[n-i].term)))
                print("           min : " + str(np.min(self.layers[n-i].term)))


        # retour console
        print("\n----------------------------------- RESULT : -----------------------------------------------")
        for c in range(len(self.categories)):
            print("[" + str(expect[c]), end = "] [")
            print(str(out[c]) + "]")
        print(error, end = "\n\n")
    

    def training(self, loss, setup):
        i = 1
        for ex in setup :
            print("\n============================================= EX : " + str(i) + " =============================================")
            i += 1
            self.train(loss, ex)


    def training_CIFAR(self, batch_nb, dist):
        loaded = np.load("Data/Data_Brut/batch_" + str(batch_nb) + ".npz")
        features, labels = loaded["f"], loaded["l"]
        start, end = dist
        for i in range(start, end):
            print("\n============================================= BATCH : " + str(batch_nb) + " - EX : " + str(i) + " =============================================")
            ex = (features[i], labels[i])
            self.train("cross_entropy_loss", ex)


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
    def  __init__(self, parameters):
         self.id = "Input"
         self.rank = 0
         self.size = parameters
         self.output = np.zeros(self.size)
         self.term = np.zeros(self.size)

    def define(self):
        print(self.id, end = ' : ')
        print(self.size)

    def update(self, out):
        self.output = out


class FullySigmoid:
    """
            FULLY CONNECTED - SIGMOID
    - size : tuple des dimensions (D, H, W)
    - weights : tableau des poids de taille (D2, H2, W2, D1, H1, W1)
    - delta : tableau des ajustements précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    - speed, moment, white : vitesse d'apprentissage, inertie et white-decay
    > init((D2, H2, W2, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0
    
    def  __init__(self, rg, parameters, prev_size): 
        FullySigmoid.ident += 1
        D, H, W, rate = parameters
        D1, H1, W1 = prev_size
        self.id = "FullyConnected_Sigmoid_n°" + str(FullySigmoid.ident)
        self.rank = rg
        self.size = (D, H, W)
        self.weigths = np.random.randn(D, H, W, D1, H1, W1) * 0.01 / sqrt(D1*H1*W1)
        self.delta = np.zeros((D, H, W, D1, H1, W1))
        self.bias = np.random.randn(D, H, W) * 0.01 / sqrt(D1*H1*W1)
        self.delta_bias = np.zeros(self.size)
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.speed, self.moment, self.white = rate

    def define(self):
        print(self.id, end=' : ')
        print(self.size)

    def front(self, inp):
        def sigmoid(array):
            return 1 / (1 + np.exp(-array))
        # sigmoid = lambda x : 1 / (1 + exp(-x))
        D, H, W = self.size
        self.output = sigmoid(np.sum(np.sum(np.sum(self.weigths[:, :, :] * inp, axis = 5), axis = 4), axis = 3) + self.bias)
        self.output[self.output > 0.99] = 0.99
        self.output[self.output < 0.001] = 0.001

    def back(self, prev_layer):
        D, H, W = self.size
        I, J, K = prev_layer.size
        prev_layer.term = np.zeros(np.shape(prev_layer.term))

        # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
        self.term = self.term * self.output * (1 - self.output)
        # on envoie le terme d'erreur en le pondérant dans le neurone précedent
        prev_layer.term = np.sum(np.sum(np.sum(np.transpose(np.transpose(self.term) * np.transpose(self.weigths)), axis = 0), axis = 0), axis = 0)

        # biais :
                           # partie inertielle             # part d'erreur
        self.delta_bias = (self.moment*self.delta_bias) + (self.speed*self.term)
        self.bias += self.delta_bias

        # poids :
                      # partie inertielle        # white-decay
        self.delta = (self.moment*self.delta) - (self.white*self.speed*self.weigths)
        for d in range(D):
            for h in range(H):
                for w in range(W):
                                            # part d'erreur
                    self.delta[d, h, w] += (self.speed * prev_layer.output * self.term[d, h, w])
        self.weigths += self.delta


class FullyMax:
    """
            FULLY CONNECTED - MAX(0, X)
    - size : tuple des dimensions (D, H, W)
    - weights : tableau des poids de taille (D2, H2, W2, D1, H1, W1)
    - delta : tableau des ajustements précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    - speed, moment, white : vitesse d'apprentissage, inertie et white-decay
    > init((D2, H2, W2, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0
    
    def  __init__(self, rg, parameters, prev_size): 
        FullyMax.ident += 1
        D2, H2, W2, rate = parameters
        D1, H1, W1 = prev_size
        self.id = "FullyConnected_Max_n°" + str(FullyMax.ident)
        self.rank = rg
        self.size = (D2, H2, W2)
        self.weigths = np.random.randn(D2, H2, W2, D1, H1, W1) * 0.01 / sqrt(D1*H1*W1)
        self.delta = np.zeros((D2, H2, W2, D1, H1, W1))
        self.bias = np.ones(self.size) * 0.01 / sqrt(D1*H1*W1)
        self.delta_bias = np.zeros(self.size)
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.speed, self.moment, self.white = rate

    def define(self):
        print(self.id, end=' : ')
        print(self.size)

    def front(self, inp):
        D, H, W = self.size
        def maxim(array):
            zero = np.zeros(np.shape(array))
            return np.maximum(zero, array) + 0.01*np.minimum(zero, array)
        self.output = maxim(np.sum(np.sum(np.sum(self.weigths[:, :, :] * inp, axis = 5), axis = 4), axis = 3) + self.bias)

    def back(self, prev_layer):
        D, H, W = self.size
        prev_layer.term = np.zeros(np.shape(prev_layer.term))

        # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
        M = self.output
        M[M >= 0] = 1
        M[M < 0] = 0.01
        self.term = self.term * M

        # biais :
                           # partie inertielle             # part d'erreur
        self.delta_bias = (self.moment*self.delta_bias) + (self.speed*self.term)
        self.bias += self.delta_bias

        # poids :
                      # partie inertielle        # white-decay
        self.delta = (self.moment*self.delta) - (self.white*self.speed*self.weigths) 
        for d in range(D):
            for h in range(H):
                for w in range(W):
                                            # part d'erreur
                    self.delta[d, h, w] += (self.speed * prev_layer.output * self.term[d, h, w])
                    # on envoie le terme d'erreur en le pondérant dans le neurone précedent
                    prev_layer.term += self.term[d, h, w] * self.weigths[d, h, w]
        self.weigths += self.delta


class ConvSigmoid:
    """
            CONVOLUTIONNAL - SIGMOID
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

    def  __init__(self, rg, parameters, prev_size):
        ConvSigmoid.ident += 1
        self.id = "Convolutionnal_Sigmoid_n°" + str(ConvSigmoid.ident)
        self.rank = rg
        D1, H1, W1 = prev_size
        D, F, S, P, rate = parameters
        if P == -1 and S == 1 :
            P = int((F - 1)/2)
        self.hyper = F, S, P
        H = int(((H1 - F + 2*P) // S) + 1)
        W = int(((W1 - F + 2*P) // S) + 1)
        self.size = D, H, W
        self.weigths = np.random.randn(D2, D1, F, F) * 0.01 / sqrt(D1*H1*W1)
        self.delta = np.zeros((D, D1, F, F))
        self.bias = np.random.randn(D) * 0.01 / sqrt(D1*H1*W1)
        self.delta_bias = np.zeros((D))
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.speed, self.moment, self.white = rate

    def define(self):
        print(self.id, end=' : ')
        print(self.size)
        F, S, P = self.hyper
        print("     Weigths : " + str(np.shape(self.weigths)))
        print("     Term : " + str(np.shape(self.term)))
        print("     Output : " + str(np.shape(self.output)))
        print("     Filter : " + str(F))
        print("     Stride : " + str(S))
        print("     Padding : " + str(P))

    def front(self, inp):
        D1, H1, W1 = inp.shape
        def sigmoid(array):
        	return 1 / (1 + np.exp(-array))
            # sigmoid = lambda x : 1 / (1 + exp(-x))
        D, H, W = self.size
        F, S, P = self.hyper
        padded_inp = np.zeros((D1, H1 + 2*P, W1 + 2*P))
        padded_inp[:, P:H1+P, P:W1+P] = inp
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    self.output[d, h, w] = np.sum(self.weigths[d] * padded_inp[:, h*S:h*S+F, w*S:w*S+F])  + self.bias[d]
                    self.output = sigmoid(self.output)
        self.output[self.output > 0.99] = 0.99
        self.output[self.output < 0.001] = 0.001

    def back(self, prev_layer):
        D, H, W = self.size
        D1, H1, W1 = prev_layer.size
        F, S, P = self.hyper
        padded_prev_out = np.zeros((D1, H1 + 2*P, W1 + 2*P))
        padded_prev_term = np.zeros((D1, H1 + 2*P, W1 + 2*P))
        padded_prev_out[:, P:H1+P, P:W1+P] = prev_layer.output

        # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
        self.term = self.term * self.output * (1 - self.output)

        # biais :
                           # partie inertielle
        self.delta_bias = (self.moment*self.delta_bias)

        # poids :
                      # partie inertielle        # white-decay
        self.delta = (self.moment*self.delta) - (self.white*self.speed*self.weigths)
        for h in range(H):
            for w in range(W):
                # on envoie le terme d'erreur en le pondérant dans le neurone précedent
                padded_prev_term[:, h*S:h*S+F, w*S:w*S+F] += np.sum(np.transpose(np.transpose(self.term[:, h, w]) * np.transpose(self.weigths)), axis = 0)
                for d in range(D):
                                            # part d'erreur
                    self.delta[d] += (self.speed * padded_prev_out[:, h*S:h*S+F, w*S:w*S+F] * self.term[d, h, w]) / (H*W)
                    self.delta_bias[d] += (self.speed*self.term[d, h, w]) / (H*W)
        prev_layer.term = padded_prev_term[:, P:H1+P, P:W1+P]
        self.weigths += self.delta
        self.bias += self.delta_bias


class ConvMax:
    """
            CONVOLUTIONNAL - MAX(0, X)
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

    def  __init__(self, rg, parameters, prev_size):
        ConvMax.ident += 1
        self.id = "Convolutionnal_Max_n°" + str(ConvMax.ident)
        self.rank = rg
        D1, H1, W1 = prev_size
        D2, F, S, P, rate = parameters
        if P == -1 and S == 1 :
            P = int((F - 1)/2)
        self.hyper = F, S, P
        H2 = int(((H1 - F + 2*P) // S) + 1)
        W2 = int(((W1 - F + 2*P) // S) + 1)
        self.size = D2, H2, W2
        self.weigths = np.random.randn(D2, D1, F, F) / sqrt(D1*H1*W1)
        self.delta = np.zeros((D2, D1, F, F))
        self.bias = np.ones((D2)) / sqrt(D1*H1*W1)
        self.delta_bias = np.zeros((D2))
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.speed, self.moment, self.white = rate

    def define(self):
        print(self.id, end=' : ')
        print(self.size)
        F, S, P = self.hyper
        print("     Weigths : " + str(np.shape(self.weigths)))
        print("     Term : " + str(np.shape(self.term)))
        print("     Output : " + str(np.shape(self.output)))
        print("     Filter : " + str(F))
        print("     Stride : " + str(S))
        print("     Padding : " + str(P))

    def front(self, inp):
        D1, H1, W1 = inp.shape
        D, H, W = self.size
        F, S, P = self.hyper
        def maxim(array):
            zero = np.zeros(np.shape(array))
            return np.maximum(zero, array) + 0.01*np.minimum(zero, array)
        padded_inp = np.zeros((D1, H1 + 2*P, W1 + 2*P))
        padded_inp[:, P:H1+P, P:W1+P] = inp
        for h in range(H):
            for w in range(W):
                self.output[:, h, w] = np.sum(np.sum(np.sum(self.weigths * padded_inp[:, h*S:h*S+F, w*S:w*S+F], axis = 3), axis = 2), axis = 1) + self.bias
        self.output = maxim(self.output)

    def back(self, prev_layer):
        D, H, W = self.size
        D1, H1, W1 = prev_layer.size
        F, S, P = self.hyper
        padded_prev_out = np.zeros((D1, H1 + 2*P, W1 + 2*P))
        padded_prev_out[:, P:H1+P, P:W1+P] = prev_layer.output
        padded_prev_term = np.zeros((D1, H1 + 2*P, W1 + 2*P))

        # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
        M = self.output
        M[M >= 0] = 1
        M[M < 0] = 0.01
        self.term = self.term * M

        # biais :
                           # partie inertielle
        self.delta_bias = (self.moment*self.delta_bias)

        # poids :
                      # partie inertielle        # white-decay
        self.delta = (self.moment*self.delta) - (self.white*self.speed*self.weigths)
        for h in range(H):
            for w in range(W):
                    # on envoie le terme d'erreur en le pondérant dans le neurone précedent
                padded_prev_term[:, h*S:h*S+F, w*S:w*S+F] += np.sum(np.transpose(np.transpose(self.term[:, h, w]) * np.transpose(self.weigths)), axis = 0)
                for d in range(D):
                                            # part d'erreur
                    self.delta[d] += (self.speed * padded_prev_out[:, h*S:h*S+F, w*S:w*S+F] * self.term[d, h, w]) / (H*W)
                    self.delta_bias[d] += (self.speed*self.term[d, h, w]) / (H*W)
        prev_layer.term = padded_prev_term[:, P:H1+P, P:W1+P]
        self.weigths += self.delta
        self.bias += self.delta_bias


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

    def  __init__(self, rg, parameters, prev_size):
        Pooling.ident += 1
        self.id = "Pooling_n°" + str(Pooling.ident)
        self.rank = rg
        D1, H1, W1 = prev_size
        F, S = parameters
        self.hyper = parameters
        H2 = int(((H1 - F) // S) + 1)
        W2 = int(((W1 - F) // S) + 1)
        self.size = D1, H2, W2
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)

    def define(self):
        print(self.id, end=' : ')
        print(self.size)

    def front(self, inp):
        D, H, W = self.size
        F, S = self.hyper
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    self.output[d, h, w] = np.max(inp[d, h*S:h*S+F, w*S:w*S+F])

    def back(self, prev_layer):
        D, H, W = self.size
        F, S = self.hyper
        prev_layer.term = prev_layer.term - prev_layer.term

        for d in range(D):
            for h in range(H):
                for w in range(W):
                    for j in range(h*S, h*S + F):
                        for k in range(w*S, w*S + F):
                            if prev_layer.output[d, j, k] == self.output[d, h, w] :
                                prev_layer.term[d, j, k] += self.term[d, h, w]



def saveArrays(net):
    if not (net.name + "_save") in os.listdir("."):
        os.mkdir(net.name + "_save")
    for layer in net.layers :
        if type(layer) != Input and type(layer) != Pooling and type(layer) != Normal and type(layer) != Softmax :
            np.savez_compressed(net.name + "_save/" + layer.id, w = layer.weigths, d = layer.delta, b = layer.bias, db = layer.delta_bias)


def loadArrays(net):
    for layer in net.layers :
        if type(layer) != Input and type(layer) != Pooling and type(layer) != Softmax :
            loaded = np.load(net.name + "_save/" + layer.id + ".npz")
            layer.weigths = loaded["w"]
            layer.delta = loaded["d"]
            layer.bias = loaded["b"]
            layer.delta_bias = loaded["db"]





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
    def  __init__(self, rg, param, prev_size):
        self.id = "Softmax"
        self.rank = rg
        self.size = prev_size
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)

    def define(self):
        print(self.id, end=' : ')
        print(self.size)

    def front (self, inp):
        e = np.exp(inp)
        self.output = e / np.sum(e)

    def back (self, prev_layer):
        prev_layer.term = self.term + self.output - prev_layer.output


class Normal:

    ident = 0

    def __init__(self, rg, param, prev_size):
        self.id = "Normal"
        self.rank = rg
        self.size = prev_size
        self.hyper = param
        self.sum = 0
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)

    def define(self):
        print(self.id, end=' : ')
        print(self.size)

    def front(self, inp):
        D, H, W = self.size
        if self.hyper == 0:
            self.norm = np.linalg.norm(inp)
        else:
            self.norm = np.linalg.norm(inp) * ((H*W) / (self.hyper**2))
        self.output = inp / self.norm

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
    			(FullyMax, (1, 1, 5, (0.1, 0.9, 0.))),
                (FullyMax, (1, 1, 5, (0.1, 0.9, 0.)))]
    
    cat_list = ["0", "1", "2", "3", "4"]
    
    ex1 = (np.array([[[0.1, 1, 1, 1, 0.1]]]), np.array([[[1, 0.1, 0.1, 1, 0.1]]]))
    ex2 = (np.array([[[1, 0.1, 0.1, 1, 1]]]), np.array([[[0.1, 1, 0.1, 0.1, 1]]]))
    ex3 = (np.array([[[1, 1, 1, 0.1, 1]]]), np.array([[[0, 1, 1, 0, 1]]]))
    ex4 = (np.array([[[0.1, 1, 0, 1, 0.1]]]), np.array([[[0.1, 0.1, 1, 1, 0.1]]]))
    
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


def Test_Deep_Conv():
    lay_list = [(Input, (3, 32, 32)),
                (ConvMax, (5, 5, 1, -1, (0.08, 0.9, 0.0005))),
                (Pooling, (2, 2)),
                (ConvMax, (5, 5, 1, -1, (0.08, 0.9, 0.0005))),
                (Pooling, (2, 2)),
                (ConvMax, (5, 5, 1, -1, (0.08, 0.9, 0.0005))),
                (FullyMax, (4, 1, 1, (0.08, 0.9, 0.0005)))]
    cat_list = [0, 1, 2]
    
    net = Network("Deep_Conv", lay_list, cat_list)
    net.define()
    

    d, h, w = 3, 32, 32
    volume = (d, h, w)
    
    img1 = np.zeros(volume)
    img2 = np.zeros(volume)
    img3 = np.zeros(volume)
    img4 = np.zeros(volume)
    
    for j in range(d):
    	for i in range(h*w):
    	    img1[j, randint(0, h-1), randint(0, w-1)] = randint(0, 10)
    	for i in range(int(h*w*0.75)):
    	    img2[j, randint(0, h-1), randint(0, w-1)] = randint(0, 10)
    	for i in range(int(h*w*0.50)):
    	    img3[j, randint(0, h-1), randint(0, w-1)] = randint(0, 10)
    	for i in range(int(h*w*0.25)):
    	    img4[j, randint(0, h-1), randint(0, w-1)] = randint(0, 10)

    ex1 = ((img1 - np.min(img1)) / (np.max(img1) - np.min(img1)), np.array([0., 1., 0., 0.]))
    ex2 = ((img2 - np.min(img2)) / (np.max(img2) - np.min(img2)), np.array([0., 0., 0., 1.]))
    ex3 = ((img3 - np.min(img3)) / (np.max(img3) - np.min(img3)), np.array([0., 1., 1., 0.]))
    ex4 = ((img4 - np.min(img4)) / (np.max(img4) - np.min(img4)), np.array([1., 1., 0., 1.]))
    
    setup = []
    for i in range(100):
    	setup.append(ex1)
    	setup.append(ex2)
    	setup.append(ex3)
    	setup.append(ex4)

    net.training("basic_loss", setup)

    return "Done"



def The_Grand_Test():
	lay_list = [(Input, (3, 32, 32)),
                (ConvMax, (16, 5, 1, -1, (0.01, 0.9, 0.0005))),
                (Pooling, (2, 2)),
                (ConvMax, (20, 5, 1, -1, (0.01, 0.9, 0.0005))),
                (Pooling, (2, 2)),
                (ConvMax, (20, 5, 1, -1, (0.01, 0.9, 0.0005))),
                (Pooling, (2, 2)),
                (FullyMax, (1, 1, 10, (0.01, 0.9, 0.0005)))]

	cat_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	net = Network("The_Grand_Test", lay_list, cat_list)
	net.define()
	net.training_CIFAR(1, (0, 1000))


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