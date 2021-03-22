import random
import os
import numpy as np
import matplotlib.pyplot as plt
from Network import Network


############################################### Control Panel ###################################################
########################## Data ########################
fromFile = np.loadtxt(fname='training_data_1.csv', delimiter=' ')
trening_data_1 = fromFile[:,0:2]

fromFile = np.loadtxt(fname='training_data_2.csv', delimiter=' ')
trening_data_2 = fromFile[:,0:2]

fromFile = np.loadtxt(fname='test_data.csv', delimiter=' ')
test_data = fromFile[:,0:2]

if not os.path.exists('out'):
    os.makedirs('out')
Path = os.path.join(os.getcwd(), 'out')

Data = trening_data_1
# Data = trening_data_2

######################### Center #######################
Radial_amount = 8
Experiments_amount = 10
Epoches_of_mach = 10

######################### Radial #######################
Neighbour_amount = 3

####################### Output Layer ###################
Epoches_output_layer = 100
Mom = 0.01
Lr_r = 0.2

#################################################################################################################

Name = '1'
Title1 = 'Wykres funkcji błędu dla:\n' + 'danych nr ' + str(nr) + ', ' + str(Radial_amount) + ' neuronów radialnych' 
Title2 = 'Wykres rozkładu neuronów radialnych dla:\n' + 'danych nr ' + str(nr) + ', ' + str(Radial_amount) + ' neuronów radialnych' 
Title3 = 'Wykres danych treningowych i uzyskanej funkcji dla:\n' + 'danych nr ' + str(nr) + ', ' + str(Radial_amount) + ' neuronów radialnych' 
net = Network(Radial_amount, Data, Lr_r, Mom)
net.train(Experiments_amount, Epoches_of_mach, Neighbour_amount, Epoches_output_layer, Path, Name, Title1, Title2, Title3)
Title3 = 'Wykres danych testowych i uzyskanej funkcji dla:\n' + 'danych nr ' + str(nr) + ', ' + str(Radial_amount) + ' neuronów radialnych' 
net.test(test_data, Path, Name, Title3)
