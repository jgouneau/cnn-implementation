from numpy import *
from random import *

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
        self.categories = cat
        n = len(layers_list)
        self.depth = n
        self.layers = []

        Type, param = layers_list[0]                                    # INPUT
        self.layers.append(Input(param))
        for i in range(1, n):
            Type, param = layers_list[i]                                # LAYERS
            self.layers.append(Type(param, self.layers[0].size))

    
    def train():
        n = self.depth
        self.layers[0].update
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)

        #* reste à déterminer comment calculer l'erreur et l'appliquer à layers[n-1]
        #*   -> il est possible d'avoir à créer une classe output
        #* + on peut afficher cette erreur et les résultats
        for i in range(2, n):
            self.layers[n-i].back(self.layers[n-i+1].term, self.layers[n-i+1].weigths)


    def training():
        return 0
    
    
    def test():
        n = self.depth
        self.layers[0].update
        for i in range(1, n):
            self.layers[i].front(self.layers[i-1].output)
        return self.layers[n-1].output #* à paufiner


class Input:
    """
            INPUT
    + s'occupe de la transition entre le réseau et les donneés
    - size : tuple des dimensions (D, W, L)
    - output : tableau des valeurs d'entrée
    > update : fait prendre à output une nouvelle valeur
    """
    def  __init__(self, parameters):
         self.id = "Input"
         self.size = parameters
 

class FullyConnected:
    """
            FULLY CONNECTED
    - size : tuple des dimensions (D, W, L)
    - weights : tableau des poids de taille (D2 W2, H2, D1, W1, H1)
    - delta : tableau des ajustements précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque poids
    - speed, inert : vitesse d'apprentissage et inertie
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0
    
    def  __init__(self, parameters, prev_size):
        FullyConnected.ident += 1
        W2, H2, D2, rate = parameters
        W1, H1, D1 = prev_size
        self.id = "FullyConnected n°" + str(FullyConnected.ident)
        self.size = (W2, H2, D2)
        self.weigths = array([[[[[[uniform(-10, 10) for i in range(D2)] for i in range(W2)]for i in range(H2)] for i in range(D1)] for i in range(W1)]for i in range(H1)])
        self.delta = array([[[[[[ 0. for i in range(D2)] for i in range(W2)]for i in range(H2)] for i in range(D1)] for i in range(W1)]for i in range(H1)])
        self.output = array([[[ 0. for i in range(D1)] for i in range(W1)]for i in range(H1)])
        self.term = array([[[[[[0. for i in range(D2)] for i in range(W2)]for i in range(H2)] for i in range(D1)] for i in range(W1)]for i in range(H1)])
        self.speed, self.inert = rate

    def front(self, input):
        return 0

    def back(self, next_layer):
        return 0
    

class Convolutional:
    """
            CONVOLUTIONNAL
    - size : tuple des dimensions (W, L, D)
    - weights : tableau des poids
    - delta : tableau des ajustements précedents
    - output : tableau des valeurs en sortie
    - term : tableau des termes d'erreur de chaque poids
    - speed, inert : vitesse d'apprentissage et inertie
    > front : fonction calculant les sorties
    > back : fonction modifiant les paramètres de la couches durant l'entrainement
    """
    ident = 0

    def  __init__(self, parameters, prev_size):
        Convolutional.ident += 1
        self.id = "Convolutional n°" + str(Convolutional.ident)
        D1, W1, H1 = prev_size
        K, F, S, P, rate = parameters
        W2 = (W1 - F + 2*P)/S + 1   #* à vérifier
        H2 = (H1 - F + 2*P)/S + 1   #* à vérifier
        self.size = K, W2, H2
        self.weigths = 0
        self.delta = 0
        self.output = array([[[ 0. for i in range(D1)] for i in range(W1)]for i in range(H1)])
        self.term = 0
        self.speed, self.inert = rate

    def front(self, prev_layer):
        self.nul = 0

    def back(self, next_layer):
        self.nul = 0


class Pooling:

    ident = 0
    def  __init__(self, parameters, prev_size):
        Pooling.id += 1
        self.id = "Pooling"


class Relu:

    ident = 0
    def  __init__(self, parameters, prev_size):
        Relu.id += 1
        self.id = "Relu"


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
            (Convolutional, (4, 3, 1, 1, (2, 5))),
            (Convolutional, (5, 3, 1, 1, (2, 5))),
            (FullyConnected, (5, 1, 1, (2, 5)))]

cat_list = ["dog", "cat", "car", "boat", "spoon"]



# ----- TEST ---------------------------------------------------------------------------

net = Network(lay_list, cat_list)
print(net.layers[net.depth-1].id)
print(net.layers[net.depth-1].weigths)