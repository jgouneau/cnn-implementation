import numpy as np
from math import exp

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
            self.layers.append(Type(param, self.layers[i-1].size))

        # CATEGORIES
        self.categories = [ "" for i in range(self.layers[n-1].size[0])]
        for i in range(min(len(cat), len(self.categories))):
            self.categories[i] = cat[i]

    
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
                    err = out[d, h, w] - expect[d, h, w]
                    error += err**2
                    self.layers[n-1].term[d, h, w] = err
        
        # rétropropagation
        for i in range(1, n-1):
            self.layers[n-i].back(self.layers[n-i-1])

        print(expect)
        print(out)
        print(error/2)


    def training(self, setup):
        for ex in setup :
            self.train(ex)
    
    
    def test(self):
        n = self.depth
        self.layers[0].update
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)
        return self.layers[n-1].output


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
         self.size = parameters
         self.output = np.zeros(self.size)
         self.term = np.zeros(self.size) 

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
    > init((D2, H2, W2, (speed, moment)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0
    
    def  __init__(self, parameters, prev_size): 
        FullyConnected.ident += 1
        D2, H2, W2, rate = parameters
        D1, H1, W1 = prev_size
        self.id = "FullyConnected n°" + str(FullyConnected.ident)
        self.size = (D2, H2, W2)
        self.weigths = np.random.randn(D2, H2, W2, D1, H1, W1) #* peut-être ajouter un '* 0.01 / sqrt(D1*H1*W1)'
        self.delta = np.zeros((D2, H2, W2, D1, H1, W1))
        self.term = np.zeros(self.size)
        self.output = np.zeros(self.size)
        self.speed, self.moment, self.white = rate

    def front(self, inp):
        sigmoid = lambda x : 1 / (1 + exp(-x))
        D, H, W = self.size
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    self.output[d, h, w] = sigmoid(np.sum(np.dot(self.weigths[d, h, w], inp)))

    def back(self, prev_layer):
        D, H, W = self.size
        I, J, K = prev_layer.size
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    # on adapte le terme d'erreur qui n'est alors que la somme des termes précédents pondérés
                    self.term[d, h, w] = self.term[d, h, w] * self.output[d, h, w] * (1 - self.output[d, h, w])
                    for i in range(I):
                        for j in range(J):
                            for k in range(K):
                                # on envoie le terme d'erreur en le pondérant dans le neurone précedent
                                prev_layer.term[i, j, k] += self.term[d, h, w]*self.weigths[d, h, w, i, j, k]
                                # nouvelle valeur du delta          # partie inertielle                         # white decay                                           # part d'erreur
                                self.delta[d, h, w, i, j, k] = (self.moment*self.delta[d, h, w, i, j, k]) - (self.white*self.speed*self.weigths[d, h, w, i, j, k])+ (self.speed*prev_layer.output[i, j, k]*self.term[d, h, w])
                                self.weigths[d, h, w, i, j, k] += self.delta[d, h, w, i, j, k] # changement du poids


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
    > init((D2, F, S, P, (speed, moment)), (D1, H1, W1))
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0

    def  __init__(self, parameters, prev_size):
        Convolutional.ident += 1
        self.id = "Convolutional n°" + str(Convolutional.ident)
        D1, H1, W1 = prev_size
        D2, F, S, P, rate = parameters
        self.hyper = F, S, P
        H2 = ((H1 - F + 2*P) // S) + 1
        W2 = ((W1 - F + 2*P) // S) + 1
        self.size = D2, H2, W2
        self.weigths = np.random.randn(D2, D1, F, F)
        self.delta = np.zeros((D2, D1, F, F))
        self.term = np.zeros((D2, D1, F, F))
        self.output = np.zeros(self.size)
        self.speed, self.moment = rate

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
                    self.output[d, h, w] = sigmoid(np.sum(np.dot(self.weigths[d], padded_inp[:, h*S:h*S+F, w*S:w*S+F])))


    def back(self, prev_layer):
        self.nul = 0


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

    def  __init__(self, parameters, prev_size):
        Pooling.ident += 1
        self.id = "Pooling n°" + str(Poolin.ident)
        D1, H1, W1 = prev_size
        F, S = parameters
        self.hyper = parameters
        H2 = ((H1 - F) // S) + 1
        W2 = ((W1 - F) // S) + 1
        self.size = D1, H2, W2
        self.term = 0
        self.output = np.zeros(self.size)

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
    def  __init__(self, prev_size):
        self.id = "Output"






# ----- EXEMPLES ------------------------------------------------------------------------

# liste pour la création d'un réseau :
lay_list = [(Input, (3, 32, 32)),
            (Convolutional, (4, 5, 1, 0, (1, 0.9))),
            (Convolutional, (5, 5, 1, 0, (1, 0.9))),
            (FullyConnected, (5, 1, 1, (1, 0.9, 0.0005)))]

cat_list = ["dog", "cat", "car", "boat", "spoon"]



# ----- TEST ---------------------------------------------------------------------------

"""
net = Network(lay_list, cat_list)
print(net.layers[net.depth-1].delta)
print(net.categories)
for l in net.layers :
    print(l.size)
print(net.test())
"""

# Test de la rétropropagation de FullyConnected
# Test1 :
"""
back_test_lay_list = [(Input, (5, 1, 1)), 
            (FullyConnected, (5, 1, 1, (0.1, 0.9, 0.))),
            (FullyConnected, (5, 1, 1, (0.1, 0.9, 0.)))]

back_test_cat_list = ["0", "1", "2", "3", "4"]

back_test_ex1 = (np.array([[[-1]], [[5]], [[2]], [[-4]], [[0]]]), np.array([[[1]], [[0]], [[0]], [[0]], [[0]]]))
back_test_ex2 = (np.array([[[7]], [[-1]], [[2]], [[-2]], [[1]]]), np.array([[[0]], [[1]], [[0]], [[0]], [[0]]]))
back_test_ex3 = (np.array([[[2]], [[3]], [[-4]], [[6]], [[6]]]), np.array([[[0]], [[0]], [[1]], [[0]], [[0]]]))
back_test_ex4 = (np.array([[[0]], [[-4]], [[1]], [[8]], [[-5]]]), np.array([[[0]], [[0]], [[0]], [[1]], [[0]]]))

back_test_setup = []
for i in range(100):
    back_test_setup.append(back_test_ex1)
    back_test_setup.append(back_test_ex2)
    back_test_setup.append(back_test_ex3)
    back_test_setup.append(back_test_ex4)

back_test_net = Network(back_test_lay_list, back_test_cat_list)
back_test_net.training(back_test_setup)
"""

# Test2 :

back_test_lay_list = [(Input, (2, 1, 1)), 
            (FullyConnected, (1, 1, 1, (0.1, 0.9, 0.0005)))]
back_test_cat_list = ["True", "False"]

back_test_ex1 = (np.array([[[0.1]], [[0.1]]]), np.array([[[0.1]]]))
back_test_ex2 = (np.array([[[0.1]], [[0.9]]]), np.array([[[0.1]]]))
back_test_ex3 = (np.array([[[0.9]], [[0.1]]]), np.array([[[0.1]]]))
back_test_ex4 = (np.array([[[0.9]], [[0.9]]]), np.array([[[0.9]]]))

back_test_setup = []
for i in range(100):
    back_test_setup.append(back_test_ex1)
    back_test_setup.append(back_test_ex2)
    back_test_setup.append(back_test_ex3)
    back_test_setup.append(back_test_ex4)

back_test_net = Network(back_test_lay_list, back_test_cat_list)
back_test_net.training(back_test_setup)