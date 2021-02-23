import numpy as np
from math import exp
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
    
    def __init__(self, layers_list, cat):
        n = len(layers_list)
        self.depth = n
        self.layers = []

        # INPUT & LAYERS
        Type, param = layers_list[0]
        self.layers.append(Input(param))
        for i in range(1, n):
            Type, param = layers_list[i]
            self.layers.append(Type(i, param, self.layers[i-1].size))

        # CATEGORIES
        self.categories = [ "" for i in range(self.layers[n-1].size[0])]
        for i in range(min(len(cat), len(self.categories))):
            self.categories[i] = cat[i]

    def define(self):
        print("Depth : " + str(self.depth))
        print("Categories : " + str(self.categories), end = '\n\n')
        for l in self.layers :
            l.define()
    
    def train(self, example):
        n = self.depth
        image, expect = example
        self.layers[0].update(image)

        # propagation avant
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)

        # calcul de l'erreur et appliquation à layers[n-1]
        D, H, W = self.layers[n-1].size
        out = self.layers[n-1].output
        error = 0
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    err = expect[d, h, w] - out[d, h, w]
                    error += err**2
                    self.layers[n-1].term[d, h, w] = err
                    print(expect[d, h, w], end='')
                    print("     ", end='')
                    print(out[d, h, w])
                    print("\n")

        
        # rétropropagation
        for i in range(1, n):
            self.layers[n-i].back(self.layers[n-i-1])

    def training(self, setup):
        for ex in setup :
            self.train(ex)
    
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
 

class FullyConnected:
    """
            FULLY CONNECTED
    - size : tuple des dimensions (D, H, W)
    - weights : tableau des poids de taille (D2, H2, W2, D1, H1, W1)
    - delta : tableau des ajustements précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque poids
    - speed, moment, white : vitesse d'apprentissage, inertie et white-decay
    > init((D2, H2, W2, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0
    
    def  __init__(self, rg, parameters, prev_size): 
        FullyConnected.ident += 1
        D2, H2, W2, rate = parameters
        D1, H1, W1 = prev_size
        self.id = "FullyConnected n°" + str(FullyConnected.ident)
        self.rank = rg
        self.size = (D2, H2, W2)
        self.weigths = np.random.randn(D2, H2, W2, D1, H1, W1) #* peut-être ajouter un '* 0.01 / sqrt(D1*H1*W1)'
        self.delta = np.zeros((D2, H2, W2, D1, H1, W1))
        self.bias = np.random.randn(D2, H2, W2) #* peut-être ajouter un '* 0.01 / sqrt(D1*H1*W1)'
        self.delta_bias = np.zeros(self.size)
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.speed, self.moment, self.white = rate

    def define(self):
        print(self.id, end=' : ')
        print(self.size)

    def front(self, inp):
        sigmoid = lambda x : 1 / (1 + exp(-x))
        D, H, W = self.size
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    self.output[d, h, w] = sigmoid(np.sum(self.weigths[d, h, w] * inp) + self.bias[d, h, w])

    def back(self, prev_layer):
        D, H, W = self.size
        I, J, K = prev_layer.size
        prev_layer.term = np.zeros((I, J, K))

        # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
        self.term = self.term * self.output * (1 - self.output)

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


class Convolutional:
    """
            CONVOLUTIONNAL
    - size : tuple des dimensions (D, H, W)
    - weights : tableau des poids de taille (D, D1, F, F)
    - delta : tableau des ajustements précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque poids
    - speed, moment : vitesse d'apprentissage et inertie
    - hyper : (F, S, P) / F : taille du filtre, S : stride (décalage), P : zero-padding
    > init((D2, F, S, P, (speed, moment, white)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0

    def  __init__(self, rg, parameters, prev_size):
        Convolutional.ident += 1
        self.id = "Convolutional n°" + str(Convolutional.ident)
        self.rank = rg
        D1, H1, W1 = prev_size
        D2, F, S, P, rate = parameters
        if P == -1 and S == 1 :
            P = int((F - 1)/2)
        self.hyper = F, S, P
        H2 = int(((H1 - F + 2*P) // S) + 1)
        W2 = int(((W1 - F + 2*P) // S) + 1)
        self.size = D2, H2, W2
        self.weigths = np.random.randn(D2, D1, F, F) #* peut-être ajouter un '* 0.01 / sqrt(D1*H1*W1)'
        self.delta = np.zeros((D2, D1, F, F))
        self.bias = np.random.randn(D2) #* peut-être ajouter un '* 0.01 / sqrt(D1*H1*W1)'
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
        sigmoid = lambda x : 1 / (1 + exp(-x))
        D1, H1, W1 = inp.shape
        D, H, W = self.size
        F, S, P = self.hyper
        padded_inp = np.zeros((D1, H1 + 2*P, W1 + 2*P))
        padded_inp[:, P:H1+P, P:W1+P] = inp
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    self.output[d, h, w] = sigmoid(np.sum(self.weigths[d] * padded_inp[:, h*S:h*S+F, w*S:w*S+F]) + self.bias[d])

    def back(self, prev_layer):
        D, H, W = self.size
        D1, H1, W1 = prev_layer.size
        F, S, P = self.hyper
        padded_prev_out = np.zeros((D1, H1 + 2*P, W1 + 2*P))
        padded_prev_out[:, P:H1+P, P:W1+P] = prev_layer.output
        padded_prev_term = np.zeros((D1, H1 + 2*P, W1 + 2*P))

        # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
        self.term = self.term * self.output * (1 - self.output)

        # biais :
                           # partie inertielle
        self.delta_bias = (self.moment*self.delta_bias)

        # poids :
                      # partie inertielle        # white-decay
        self.delta = (self.moment*self.delta) - (self.white*self.speed*self.weigths)
        for d in range(D):
            for h in range(H):
                for w in range(W):
                                            # part d'erreur
                    self.delta[d] += (self.speed * padded_prev_out[:, h*S:h*S+F, w*S:w*S+F] * self.term[d, h, w]) / (H*W)
                    self.delta_bias[d] += (self.speed*self.term[d, h, w]) / (H*W)
                    # on envoie le terme d'erreur en le pondérant dans le neurone précedent
                    padded_prev_term[:, h*S:h*S+F, w*S:w*S+F] += self.term[d, h, w] * self.weigths[d]
        prev_layer.term = padded_prev_term[:, P:H1+P, P:W1+P]
        self.weigths += self.delta
        self.bias += self.delta_bias


class Pooling:
    """
            CONVOLUTIONNAL
    - size : tuple des dimensions (D, H, W)
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque poids
    - hyper : (F, S) / F : taille du filtre, S : stride (décalage)
    > init((F, S), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0

    def  __init__(self, rg, parameters, prev_size):
        Pooling.ident += 1
        self.id = "Pooling n°" + str(Poolin.ident)
        self.rank = rg
        D1, H1, W1 = prev_size
        F, S = parameters
        self.hyper = parameters
        H2 = ((H1 - F) // S) + 1
        W2 = ((W1 - F) // S) + 1
        self.size = D1, H2, W2
        self.term = 0
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


class Output(FullyConnected):
    """
            OUTPUT
    + s'occupe de la transition entre la sortie du réseau et la lecture
    - output : tableau des valeurs de sortie
    (- expect : tableau des valeurs attendues)
    """
    def  __init__(self, rg, prev_size):
        self.id = "Output"
        self.rank = rg

    def define(self):
        print(self.id, end=' : ')
        print(self.size)






"""
_________________________________________________________________________________________
_________________________________________________________________________________________

                                        TESTS
_________________________________________________________________________________________
_________________________________________________________________________________________

"""


# :::::::::::::::::::::::::::::::::::: Exemple ::::::::::::::::::::::::::::::::::::::::::
"""
lay_list = [(Input, (3, 32, 32)),
            (Convolutional, (4, 5, 1, 0, (1, 0.9))),
            (Convolutional, (5, 5, 1, 0, (1, 0.9))),
            (FullyConnected, (5, 1, 1, (1, 0.9, 0.0005)))]

cat_list = ["dog", "cat", "car", "boat", "spoon"]

net = Network(lay_list, cat_list)
net.define()
"""

# :::::::::::::::::::::::::::::::::: Test_Deep_FC :::::::::::::::::::::::::::::::::::::::
"""
back_test_lay_list = [(Input, (1, 1, 5)), 
            (FullyConnected, (1, 1, 5, (0.5, 0.9, 0.))),
            (FullyConnected, (1, 1, 5, (0.5, 0.9, 0.))),
            (FullyConnected, (1, 1, 5, (0.5, 0.9, 0.))),
            (FullyConnected, (1, 1, 5, (0.5, 0.9, 0.)))]

back_test_cat_list = ["0", "1", "2", "3", "4"]

back_test_ex1 = (np.array([[[-1, 5, 2, 3, 0]]]), np.array([[[1, 0, 0, 0, 0]]]))
back_test_ex2 = (np.array([[[7, 0, -2, 6, 1]]]), np.array([[[0, 1, 0, 0, 0]]]))
back_test_ex3 = (np.array([[[2, 1, 2, -3, 6]]]), np.array([[[0, 0, 1, 0, 0]]]))
back_test_ex4 = (np.array([[[0, 4, 0, 1, -5]]]), np.array([[[0, 0, 0, 1, 0]]]))

back_test_setup = []
for k in range(100):
    for i in range(3):
        back_test_setup.append(back_test_ex1)
    for i in range(3):
        back_test_setup.append(back_test_ex2)
    for i in range(3):
        back_test_setup.append(back_test_ex3)
    for i in range(3):
        back_test_setup.append(back_test_ex4)
for j in range(10):
    back_test_setup.append(back_test_ex1)
    back_test_setup.append(back_test_ex2)
    back_test_setup.append(back_test_ex3)
    back_test_setup.append(back_test_ex4)

back_test_net = Network(back_test_lay_list, back_test_cat_list)
back_test_net.training(back_test_setup)
"""


# ::::::::::::::::::::::::::::::::::: Test_FC_And :::::::::::::::::::::::::::::::::::::::
"""
back_test_lay_list = [(Input, (2, 1, 1)), 
            (FullyConnected, (1, 1, 1, (0.1, 0.9, 0.0005)))]
back_test_cat_list = ["True", "False"]

back_test_ex1 = (np.array([[[0.]], [[0.]]]), np.array([[[0.]]]))
back_test_ex2 = (np.array([[[0.]], [[1.]]]), np.array([[[0.]]]))
back_test_ex3 = (np.array([[[1.]], [[0.]]]), np.array([[[0.]]]))
back_test_ex4 = (np.array([[[1.]], [[1.]]]), np.array([[[1.]]]))

back_test_setup = []
for i in range(1000):
    back_test_setup.append(back_test_ex1)
    back_test_setup.append(back_test_ex2)
    back_test_setup.append(back_test_ex3)
    back_test_setup.append(back_test_ex4)

back_test_net = Network(back_test_lay_list, back_test_cat_list)
back_test_net.training(back_test_setup)
"""


# :::::::::::::::::::::::::::::::::: Test_Deep_Conv :::::::::::::::::::::::::::::::::::::
"""
lay_list = [(Input, (3, 32, 32)),
            (Convolutional, (5, 5, 1, -1, (1, 0.9, 0.0005))),
            (Convolutional, (5, 2, 2, 0, (1, 0.9, 0.0005))),
            (Convolutional, (5, 5, 1, -1, (1, 0.9, 0.0005))),
            (Convolutional, (5, 2, 2, 0, (1, 0.9, 0.0005))),
            (Convolutional, (5, 5, 1, -1, (1, 0.9, 0.0005))),
            (FullyConnected, (4, 1, 1, (1, 0.9, 0.0005)))]
cat_list = [0, 1, 2]

net = Network(lay_list, cat_list)
net.define()


img1 = np.ones((3, 32, 32))
img2 = np.ones((3, 32, 32))
img3 = np.ones((3, 32, 32))
img4 = np.ones((3, 32, 32))

for i in range(32*32):
    img1[randint(0, 2), randint(0, 31), randint(0, 31)] = 0
for i in range(600):
    img2[randint(0, 2), randint(0, 31), randint(0, 31)] = 0
for i in range(300):
    img3[randint(0, 2), randint(0, 31), randint(0, 31)] = 0
for i in range(100):
    img4[randint(0, 2), randint(0, 31), randint(0, 31)] = 0

ex1 = (img1, np.array([[[0.]], [[0.]], [[0.]], [[0.]]]))
ex2 = (img2, np.array([[[0.]], [[0.]], [[0.]], [[1.]]]))
ex3 = (img3, np.array([[[0.]], [[1.]], [[1.]], [[0.]]]))
ex4 = (img4, np.array([[[1.]], [[1.]], [[0.]], [[1.]]]))

setup = []
for j in range(100):
    setup.append(ex1)
    setup.append(ex2)
    setup.append(ex3)
    setup.append(ex4)

net.training(setup)
"""


# :::::::::::::::::::::::::::::::::: Test_FC_Volume :::::::::::::::::::::::::::::::::::::

d, h, w = 2, 8, 8
volume = (d, h, w)

lay_list = [(Input, volume),
            (FullyConnected, (1, 1, 6, (1, 0.9, 0.0005))),
            (FullyConnected, (1, 1, 2, (1, 0.9, 0.0005)))]
cat_list = [0, 1]

net = Network(lay_list, cat_list)
net.define()


img1 = np.ones(volume)
img2 = np.ones(volume)
img3 = np.ones(volume)
img4 = np.ones(volume)

for i in range(h*w):
    img1[0, randint(0, 7), randint(0, 7)] = 0
for i in range(int(h*w*0.75)):
    img2[0, randint(0, 7), randint(0, 7)] = 0
for i in range(int(h*w*0.50)):
    img3[0, randint(0, 7), randint(0, 7)] = 0
for i in range(int(h*w*0.25)):
    img4[0, randint(0, 7), randint(0, 7)] = 0

ex1 = (img1, np.array([[[0, 0]]]))
ex2 = (img2, np.array([[[0, 1]]]))
ex3 = (img3, np.array([[[1, 0]]]))
ex4 = (img4, np.array([[[1, 1]]]))

setup = []
for i in range(100):
    setup.append(ex1)
    setup.append(ex2)
    setup.append(ex3)
    setup.append(ex4)
net.training(setup)
print(net.test(img1))
