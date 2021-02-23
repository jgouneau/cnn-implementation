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

    
    def train(self):
        n = self.depth
        self.layers[0].update
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)

        #* reste à déterminer comment calculer l'erreur et l'appliquer à layers[n-1]
        #*   -> il est possible d'avoir à créer une classe output
        #* + on peut afficher cette erreur et les résultats
        for i in range(2, n):
            self.layers[n-i].back(self.layers[n-i+1].term, self.layers[n-i+1].weigths)


    def training(self):
        return 0
    
    
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
         self.output = np.ones(self.size)

    def update(self):
        return 0
 

class FullyConnected:
    """
            FULLY CONNECTED
    - size : tuple des dimensions (D, H, W)
    - weights : tableau des poids de taille (D2, H2, W2, D1, H1, W1)
    - delta : tableau des ajustements précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque poids
    - speed, moment : vitesse d'apprentissage et inertie
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
        self.weigths = 0.01 * np.random.randn(D2, H2, W2, D1, H1, W1) #* peut-être ajouter un ' / sqrt(D1*H1*W1)'
        self.delta = np.zeros((D2, H2, W2, D1, H1, W1))
        self.term = np.zeros((D2, H2, W2, D1, H1, W1))
        self.output = np.zeros((D2, W2, H2))
        self.speed, self.moment = rate

    def front(self, inp):
        sigmoid = lambda x : 1 / (1 + exp(-x))
        D, H, W = self.size
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    self.output[d, h, w] = sigmoid(np.sum(np.dot(self.weigths[d, h, w], inp)))

    def back(self, next_layer):
        return 0
    

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
        self.output = np.zeros((D2, H2, W2))
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


    def back(self, next_layer):
        self.nul = 0


class Pooling:

    ident = 0
    def  __init__(self, parameters, prev_size):
        Pooling.id += 1
        self.id = "Pooling"


class Output:
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
            (Convolutional, (4, 5, 1, 0, (2, 5))),
            (Convolutional, (5, 5, 1, 0, (2, 5))),
            (FullyConnected, (5, 1, 1, (2, 5)))]

cat_list = ["dog", "cat", "car", "boat", "spoon"]



# ----- TEST ---------------------------------------------------------------------------

net = Network(lay_list, cat_list)
print(net.layers[net.depth-1].delta)
print(net.categories)
for l in net.layers :
    print(l.size)
print(net.test())