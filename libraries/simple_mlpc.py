import torch 
import numpy as np
#import mat.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#X = torch.tensor([1.0,2.0,3.0])

#w = torch.tensor([0.1,0.1,0.1], dtype= torch.float32, requires_grad=True)

#print(w*X)

iris = load_iris()
sc = StandardScaler()

Y = torch.from_numpy(iris.target, dtype=np.float32)
X = torch.from_numpy(iris.data, dtype=np.float32)

print(Y)

#input_size = 
num_classes = 3 # NOTE es gibt nur 3 Blumen :) 
batch_size = 20 
num_epochs = 100
learning_rate = 0.01

if torch.cuda.is_available():
    print("BARACUDAAA")