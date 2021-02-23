import numpy as np
import pickle
from scipy.misc import imread, imsave, imresize


def extract_batch(batch_path):
	with open(batch_path, 'rb') as fo:
	        	batch = pickle.load(fo, encoding='bytes')
	return batch[b'data'], batch[b'labels']

def reshape(features):
	return np.reshape(features, (len(features), 3, 32, 32))

def one_hot(labels):
	encoded = np.zeros((len(labels), 10))
	for idx, val in enumerate(labels):
		encoded[idx, val] = 1
	return encoded

def routine(batch_nb):
	(ftr, lbl) = extract_batch("CIFAR-10/data_batch_" + str(batch_nb))
	features = reshape(ftr)
	labels = one_hot(lbl)

	np.savez_compressed("Data/Data_Brut/batch_" + str(batch_nb), f = features, l = labels)


routine(5)