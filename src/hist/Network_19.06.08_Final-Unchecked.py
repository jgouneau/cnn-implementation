import numpy as np
import os
import pickle
from scipy.misc import imread, imsave, imresize
from math import exp, log, sqrt
from random import randint
import time
import matplotlib.pyplot as plt


class Network:
    """
            NETWORK
    - layers : liste ou tableau de couches de neurones
    - depth : le nombre de couches
    - batch_size : nombre d'exemples parallèles par adaptation
    - loss : évolution de l'erreur au cours des entraînements
    - accuracy : évolution de la précision au cours des entraînements
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
        self.loss = np.array([])
        self.accuracy = np.array([])

        # LAYERS
        Type, param = layers_list[0]
        self.layers.append(Input(B, param))
        for i in range(1, n):
            Type, param = layers_list[i]
            self.layers.append(Type(param, self.layers[i-1].size))
        FullyConnected.ident = 0
        Convolutional.ident = 0
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
            if type(self.layers[i]) == FullyConnected or type(self.layers[i]) == Convolutional :
                self.layers[i].define(self.layers[i+1])
            if type(self.layers[i]) == Pooling :
                self.layers[i].define()



    # ENTRAINEMENT DU RÉSEAU

    def train(self, example):
        n = self.depth
        img, label = example
        self.layers[0].update(img)

        # propagation avant
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)

        # calcul de l'erreur
        self.layers[n-1].loss(label)

        # rétropropagation
        for i in range(1, n):
            self.layers[n-i].back(self.layers[n-i-1])


        # retour console
        """
        print("\n----------------------------------- RESULT : -----------------------------------------------")
        for b in range(self.batch_size):
        	out = np.reshape(self.layers[n-1].output[b], len(self.categories))
        	print("\n")
	        for c in range(len(self.categories)):
	            print("[" + str(label[b, c]), end = "] [")
	            print(str(out[c]) + "]")
	    """
        batch_loss = np.sum(self.layers[n-1].error)/self.batch_size
        print(batch_loss, end = "\n\n")
        self.loss = np.append(self.loss, batch_loss)


    def training(self, dataset, dist):
    	B = self.batch_size
    	batch, (features, labels) = dataset
    	start, end = dist
    	end = int((end-start)/B + start)
    	for i in range(start, end):
            print("\n BATCH : " + str(batch) + " - " + str(int(i/end*100)) + "%")
            ex = (features[i*B : (i+1)*B], labels[i*B : (i+1)*B])
            self.train(ex)


    def tempo(self):
    	for l in self.layers :
    		print(l.id + " : " + str(l.tempo))
    	print(self.loss)



    # TEST RAPIDE SANS MODIFICATION DES POIDS

    def test(self, example):
        n = self.depth
        B = self.batch_size
        C = len(self.categories)
        img, label = example

        self.layers[0].update(img)
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)

        out = np.reshape(self.layers[n-1].output, newshape = (B, C))
        for b in range(B):
        	for c in range(C):
        		print("[" + str(label[b, c]) + "] [" + str(out[b, c]) + "] -> " + self.categories[c])
        	print("")


    def testing(self, dataset, dist):
    	B = self.batch_size
    	features, labels = dataset
    	start, end = dist
    	end = int((end-start)/B + start)
    	for i in range(0, end-start):
            print("\nEX : " + str(start + i*B + 1))
            ex = (features[start + i*B : start + (i+1)*B], labels[start + i*B : start + (i+1)*B])
            self.test(ex)



    # VISUALISATION DE LA RESPONSABILITÉ DE CHAQUE PIXEL

    def responsability(self, example, name):
    	# fonction développée personnellement

    	n = self.depth
    	img, label = example
    	self.layers[0].update(img)

    	# propagation avant
    	for i in range(1, n):
    		self.layers[i].front(self.layers[i-1].output)

    	# calcul de la justesse
    	self.layers[n-1].gain(label)

    	# rétropropagation sans modification des poids
    	for i in range(1, n):
    		self.layers[n-i].retro(self.layers[n-i-1])
    	self.layers[0].response(name)


    def visual(self, example, F, name):
    	# fonction existante

    	n = self.depth
    	img, label = example
    	self.layers[0].update(img)
    	B, D, H, W = np.shape(img)
    	H1 = H - F + 1
    	W1 = W - F + 1
    	heat_map = np.zeros((B, H1, W1))

    	for h in range(H1):
    		for w in range(W1):
    			cache = np.copy(img)
    			cache[:, :, h : h+F, w : w+F] = np.zeros((B, D, F, F))
    			self.layers[0].update(cache)
    			for i in range(1, n):
    				self.layers[i].front(self.layers[i-1].output)
    			heat_map[:, h, w] = self.layers[n-1].gain(label)
    			print(heat_map[:, h, w])
    	for b in range(B):
    		imsave("Images/"+ name + "_" + str(b) + "_expect" + ".png", np.transpose(img[b], (1, 2, 0)))
    		imsave("Images/"+ name + "_" + str(b) + "_visual" + ".png", -heat_map[b])



    # ACCURACY / PRÉCISION DU RÉSEAU

    def veri(self, example):
    	n = self.depth
    	img, label = example
    	self.layers[0].update(img)

    	# propagation avant
    	for i in range(1, n):
    		self.layers[i].front(self.layers[i-1].output)

    	# calcul de la précision
    	points = self.layers[n-1].score(label)
    	print(points)
    	return points


    def verify(self, dataset):
    	B = self.batch_size
    	points = 0
    	features, labels = dataset
    	end = int(len(features)/B)
    	for i in range(end):
    		print("\n" + str(int(i/end*100)) + "%")
    		ex = (features[i*B : (i+1)*B], labels[i*B : (i+1)*B])
    		points += self.veri(ex)
    	self.accuracy = np.append(self.accuracy, points/(end*B))



    # ENREGISTREMENT / CHARGEMENT DU RÉSEAU

    def load(self):
	    loaded = np.load(self.id + "_save/Loss.npz")
	    self.loss = loaded["l"]
	    loaded = np.load(self.id + "_save/Accuracy.npz")
	    self.accuracy = loaded["a"]
	    for l in self.layers :
	        if type(l) == FullyConnected or type(l) == Convolutional:
	            loaded = np.load(self.id + "_save/" + l.id + ".npz")
	            l.weigths = loaded["w"]
	            l.delta = loaded["d"]
	            l.bias = loaded["b"]
	            l.delta_bias = loaded["db"]


    def save(self):
	    if not (self.id + "_save") in os.listdir("."):
	        os.mkdir(self.id + "_save")
	    np.savez_compressed(self.id + "_save/Loss", l = self.loss)
	    np.savez_compressed(self.id + "_save/Accuracy", a = self.accuracy)
	    for l in self.layers :
	        if type(l) == FullyConnected or type(l) == Convolutional:
	            np.savez_compressed(self.id + "_save/" + l.id, w = l.weigths, d = l.delta, b = l.bias, db = l.delta_bias)



class Input:
    """
            INPUT
    + s'occupe de la transition entre le réseau et les donneés
    - size : tuple des dimensions (D, H, W)
    - output : tableau des valeurs d'entrée
    > update : fait prendre à output une nouvelle valeur
    > response : donne la carte des responsabilités de chaque pixel dans le résultat et l'enregistre
    """
    def  __init__(self, B, parameters):
         self.id = "Input"
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
        self.output = np.reshape(out, self.size)
        self.tempo[0] += time.clock() - start

    def response(self, name):
    	B, D, H, W = self.size
    	self.term = np.absolute(self.term)
    	mapp = np.sum(self.term, axis = 1)
    	def normal(feat):
    		min_val = np.min(feat)
    		max_val = np.max(feat)
    		return (feat - min_val) / (max_val - min_val)
    	for b in range(B):
    		imsave("Images/"+ name + "_" + str(b) + "_expect" + ".png", np.transpose(self.output[b], (1, 2, 0)))
    		imsave("Images/"+ name + "_" + str(b) + "_response" + ".png", mapp[b])


class FullyConnected:
    """
            FULLY-CONNECTED
    - size : tuple des dimensions (D, H, W)
    - weights : tableau des poids de taille (D, H, W, D1, H1, W1)
    - delta : tableau des ajustements précedents
    - bias : tableau des biais de taille (D, H, W)
    - delta_bias : tableau des ajustements de biais précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    - speed, moment : vitesse d'apprentissage et inertie
    > init((D2, F, S, P, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    > retro : fonction donnant les responsabilités de chaque entrée dans la sortie
    """

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
        B, D, H, W = self.size
        self.output = np.transpose(np.einsum('bijk,...ijk', inp, self.weigths), axes = [3, 0, 1, 2]) + np.einsum('dhw,...', self.bias, self.distrib)

        # sortie :
        """
        print("\n" + self.id)
        print("    out max : " + str(np.max(self.output)))
        print("        min : " + str(np.min(self.output)))
        print("weigths max : " + str(np.max(self.weigths)))
        print("        min : " + str(np.min(self.weigths)))
        print("   bias max : " + str(np.max(self.bias)))
        print("        min : " + str(np.min(self.bias)))
        #
        for b in range(B):
        	print("Output : " + str(b+1))
        	print(self.output[b])
        """
        self.tempo[0] += time.clock() - start

    def back(self, prev_layer):
    	start = time.clock()
    	B, D, H, W = self.size

    	# terme d'erreur :
    	prev_layer.term = np.transpose(np.einsum('bdhw,dhw...', self.term, self.weigths), axes = [3, 0, 1, 2])

    	# biais :
    	self.delta_bias = (self.speed / B * np.sum(self.term, axis = 0)) + (self.moment*self.delta_bias)
    	self.bias += self.delta_bias

    	# poids :
    	self.delta = self.speed / B * (np.einsum('bijk,b...', prev_layer.output, self.term) - self.white*self.weigths) + (self.moment*self.delta)
    	self.weigths += self.delta

    	#sortie :
    	"""
        print("\n" + self.id)
        print("      term max : " + str(np.max(self.term)))
        print("           min : " + str(np.min(self.term)))
        print("     delta max : " + str(np.max(self.delta)))
        print("           min : " + str(np.min(self.delta)))
        print("delta_bias max : " + str(np.max(self.delta_bias)))
        print("           min : " + str(np.min(self.delta_bias)))
        """

    	self.tempo[1] += time.clock() - start

    def retro(self, prev_layer):
        start = time.clock()
        B, D, H, W = self.size

        # terme d'erreur :
        prev_layer.term = np.transpose(np.einsum('bdhw,dhw...', self.term, self.weigths), axes = [3, 0, 1, 2])
        self.tempo[1] += time.clock() - start


class Convolutional:
    """
            CONVOLUTIONAL
    - size : tuple des dimensions (D, H, W)
    - weights : tableau des poids de taille (D, D1, F, F)
    - delta : tableau des ajustements précedents
    - bias : tableau des biais de taille (D)
    - delta_bias : tableau des ajustements de biais précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    - speed, moment : vitesse d'apprentissage et inertie
    - hyper : (F, S, P) / F : taille du filtre, S : stride (décalage), P : zero-padding
    > init((D2, F, S, P, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    > retro : fonction donnant les responsabilités de chaque entrée dans la sortie
    """

    ident = 0

    def  __init__(self, parameters, prev_size):
        Convolutional.ident += 1
        self.id = "Convolutional_n°" + str(Convolutional.ident)
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
    	for b in range(B):
    		for h in range(H):
    			for w in range(W):
    				self.output[b, :, h, w] = np.transpose(np.einsum('...ijk,ijk', self.weigths, padded_inp[b, :, h*S:h*S+F, w*S:w*S+F])) + self.bias

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
    	self.delta_bias = (self.speed / (H*W*B) * np.sum(self.term, axis = (3, 2, 0))) + (self.moment*self.delta_bias)

    	for b in range(B):
    		for h in range(H):
    			for w in range(W):
    				# on envoie le terme d'erreur dans les neurones précedents
    				padded_prev_term[b, :, h*S:h*S+F, w*S:w*S+F] += np.einsum('d..., d...', self.term[b, :, h, w], self.weigths)
    				delta += np.einsum('ijk,...', padded_prev_out[b, :, h*S:h*S+F, w*S:w*S+F], self.term[b, :, h, w])

    	# poids :
    	self.delta = self.speed * ((delta / (H*W*B)) + (self.moment*self.delta) - (self.white*self.weigths))

    	self.weigths += self.delta
    	self.bias += self.delta_bias
    	prev_layer.term = padded_prev_term[:, :, P:H1+P, P:W1+P]

    	# sortie :
    	"""
    	print("\n" + self.id)
    	print("      term max : " + str(np.max(self.term)))
    	print("           min : " + str(np.min(self.term)))
    	print("     delta max : " + str(np.max(self.delta)))
    	print("           min : " + str(np.min(self.delta)))
    	print("delta_bias max : " + str(np.max(self.delta_bias)))
    	print("           min : " + str(np.min(self.delta_bias)))
    	"""
    	self.tempo[1] += time.clock() - start

    def retro(self, prev_layer):
    	start = time.clock()
    	B, D, H, W = self.size
    	B, D1, H1, W1 = prev_layer.size
    	F, S, P = self.hyper
    	padded_prev_out = np.zeros((B, D1, H1 + 2*P, W1 + 2*P))
    	padded_prev_out[:, :, P:H1+P, P:W1+P] = prev_layer.output
    	padded_prev_term = np.zeros((B, D1, H1 + 2*P, W1 + 2*P))

    	for b in range(B):
    		for h in range(H):
    			for w in range(W):
    				padded_prev_term[b, :, h*S:h*S+F, w*S:w*S+F] += np.einsum('d..., d...', self.term[b, :, h, w], self.weigths)

    		# on envoie le terme d'erreur en le pondérant dans le neurone précedent
    	prev_layer.term = padded_prev_term[:, :, P:H1+P, P:W1+P]
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
    > retro : fonction donnant les responsabilités de chaque entrée dans la sortie
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
        # print("\n" + self.id)
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
        # print("\n" + self.id)
        self.tempo[1] += time.clock() - start

    def retro(self, prev_layer):
    	self.back(prev_layer)


class Sigmoid:
    """
            SIGMOID
    - size : tuple des dimensions (D, H, W)
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    > init((D2, H2, W2, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    > retro : fonction donnant les responsabilités de chaque entrée dans la sortie
    > loss : fonction donnant l'erreur sur le résultat et l'enregistre
    > gain : fonction de départ utilisée dans Network.responsability()
    > score : fonction donnant le nombre d'exemples réussis dans le batch
    """

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

    def retro(self, prev_layer):
    	self.back(prev_layer)

    def loss(self, expect):
        start = time.clock()
        out = np.copy(np.reshape(self.output, np.shape(expect)))
        err = expect - out
        self.term = np.reshape(err, np.shape(self.term))
        self.error = np.sum(err**2)/2
        self.tempo[2] += time.clock() - start

    def gain (self, expect):
    	B, D, H, W = self.size
    	out = np.copy(np.reshape(self.output, np.shape(expect)))
    	ga = expect
    	self.term = np.reshape(ga, np.shape(self.term))
    	return np.sum(out*ga, axis = 1)

    def score(self, expect):
    	B, D, H, W = self.size
    	out = np.copy(np.reshape(self.output, np.shape(expect)))
    	points = 0
    	for b in range(B):
    		if np.argmax(out[b]) == np.argmax(expect[b]):
    			points += 1
    	return points


class Relu:
    """
            RELU
    - size : tuple des dimensions (D, H, W)
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    > init((D2, H2, W2, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    > retro : fonction donnant les responsabilités de chaque entrée dans la sortie
    > loss : fonction donnant l'erreur sur le résultat et l'enregistre
    > gain : fonction de départ utilisée dans Network.responsability()
    > score : fonction donnant le nombre d'exemples réussis dans le batch
    """
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
            return np.maximum(array, 0.01*array)
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

    def retro(self, prev_layer):
    	self.back(prev_layer)

    def loss(self, expect):
        start = time.clock()
        out = np.copy(np.reshape(self.output, np.shape(expect)))
        err = expect - out
        self.term = np.reshape(err, np.shape(self.term))
        self.error = np.sum(err**2)/2
        self.tempo[2] += time.clock() - start

    def gain (self, expect):
    	B, D, H, W = self.size
    	out = np.copy(np.reshape(self.output, np.shape(expect)))
    	ga = expect
    	self.term = np.reshape(ga, np.shape(self.term))
    	return np.sum(out*ga, axis = 1)

    def score(self, expect):
    	B, D, H, W = self.size
    	out = np.copy(np.reshape(self.output, np.shape(expect)))
    	points = 0
    	for b in range(B):
    		if np.argmax(out[b]) == np.argmax(expect[b]):
    			points += 1
    	return points


class Softmax:
    """
            SOFTMAX
    - size : tuple des dimensions (D, H, W)
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque sortie
    > init((D2, H2, W2, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    > retro : fonction donnant les responsabilités de chaque entrée dans la sortie
    > loss : fonction donnant l'erreur sur le résultat et l'enregistre
    > gain : fonction de départ utilisée dans Network.responsability()
    > score : fonction donnant le nombre d'exemples réussis dans le batch
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

    def retro(self, prev_layer):
    	self.back(prev_layer)

    def loss (self, expect):
        start = time.clock()
        B, D, H, W = self.size
        out = np.copy(np.reshape(self.output, np.shape(expect)))
        err = expect - out
        self.term = np.reshape(err, np.shape(self.term))
        self.error = - np.stack([np.sum(expect[b]*np.log(out[b])) for b in range(B)])
        self.tempo[2] += time.clock() - start

    def gain (self, expect):
    	B, D, H, W = self.size
    	out = np.copy(np.reshape(self.output, np.shape(expect)))
    	ga = expect
    	self.term = np.reshape(ga, np.shape(self.term))
    	return np.sum(out*ga, axis = 1)

    def score(self, expect):
    	B, D, H, W = self.size
    	out = np.copy(np.reshape(self.output, np.shape(expect)))
    	points = 0
    	for b in range(B):
    		if np.argmax(out[b]) == np.argmax(expect[b]):
    			points += 1
    	return points











def Train_CIFAR():
	# Test_Cifar :     0.010 - 0.008 - 0.006 - 0.005 - 0.004 - 0.003 - 0.002 - 0.001 - 0.0008
	# Test_Cifar-2.0 : 0.010 - 0.008 - 0.006 - 0.005 - 0.004 - 0.002 - 0.001 - 0.00075 - 0.0005 - 0.00025
	lay_list = [(Input, (3, 32, 32)),
				(Convolutional, (16, 5, 1, 2, (0.00025, 0.9, 0.0001))), (Relu, 0),
				(Pooling, (2, 2)),
				(Convolutional, (20, 5, 1, 2, (0.00025, 0.9, 0.0001))), (Relu, 0),
				(Pooling, (2, 2)),
				(Convolutional, (20, 5, 1, 2, (0.00025, 0.9, 0.0001))), (Relu, 0),
				(Pooling, (2, 2)),
				(FullyConnected, (1, 1, 10, (0.00025, 0.9, 0.0001))), (Softmax, 0)]

	cat_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	net = Network("Test_CIFARbite", 5, lay_list, cat_list)
	net.define()

	"""
	net.load()
	for b in range(1, 6):
		loaded = np.load("Data/Data_Brut/batch_" + str(b) + ".npz")
		dataset = loaded["f"], loaded["l"]
		print("############################################## BATCH : " + str(b) + " ##############################################")
		net.training((b, dataset), (0, 10000))
		net.save()

		loaded = np.load("Data/Data_Brut/batch_test.npz")
		dataset = loaded["f"], loaded["l"]
		net.verify(dataset)
		net.save()
		print(net.accuracy)
	"""
	loaded = np.load("Data/Data_Brut/batch_1.npz")
	dataset = loaded["f"], loaded["l"]
	net.training((1, dataset), (0, 1000))



def Test_Response():
	lay_list = [(Input, (3, 32, 32)),
				(Convolutional, (16, 5, 1, 2, (0.002, 0.9, 0.0001))), (Relu, 0),
				(Pooling, (2, 2)),
				(Convolutional, (20, 5, 1, 2, (0.002, 0.9, 0.0001))), (Relu, 0),
				(Pooling, (2, 2)),
				(Convolutional, (20, 5, 1, 2, (0.002, 0.9, 0.0001))), (Relu, 0),
				(Pooling, (2, 2)),
				(FullyConnected, (1, 1, 10, (0.002, 0.9, 0.0001))), (Softmax, 0)]

	cat_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	net = Network("Test_CIFAR", 5, lay_list, cat_list)
	net.define()

	net.load()
	loaded = np.load("Data/Data_Brut/batch_1.npz")
	img, lbl = loaded["f"], loaded["l"]

	a, b = 4, 9

	net.testing((img, lbl), (a, b))
	net.responsability((img[a : b], lbl[a : b]), "new")
	net.visual((img[a : b], lbl[a : b]), 10, "new")
		


def print_error():
	loaded = np.load("Test_CIFAR_save/Loss.npz")
	loss = loaded["l"]
	loaded = np.load("Test_CIFAR_save/Accuracy.npz")
	accuracy = loaded["a"]
	l = len(loss)
	p = len(accuracy)
	n = 1000

	err = np.array([])
	x = np.array([])
	for i in range(n):
		err = np.append(err, np.sum(loss[int(i*l/n) : int((i+1)*l/n)])/l*n)
		x = np.append(x, i)

	y = np.array([])
	for i in range(p):
		y = np.append(y, i*n/p)

	plt.plot(x, err)
	# plt.plot(y, accuracy)
	plt.show()


#print_error()
Train_CIFAR()