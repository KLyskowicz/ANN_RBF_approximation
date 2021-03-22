import random
import numpy as np
import math
import os
from Radial import Radial
from Neuron import Neuron
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, radial_amount, data, learning_rate, momentum):
        if radial_amount > len(data):
            sys.exit("więcej neuronów radialnych niż punktów danych")
        self.data = data
        self.radial_amount = radial_amount
        self.min_x = data[np.argmin(data, axis=0)[0]][0]
        self.min_y = data[np.argmin(data, axis=0)[1]][1]
        self.max_x = data[np.argmax(data, axis=0)[0]][0]
        self.max_y = data[np.argmax(data, axis=0)[1]][1]
        self.last_radials_xy = []
        self.last_error = 0
        self.last_error_for_epo = 0
        self.radials = []
        self.radials2 = []
        self.output_layer = Neuron(radial_amount, 0, momentum, learning_rate)
        self.errorY = []
        self.errorX = []
        self.sum_error = 0

    def new_xy_na3(self):
        self.radials.clear()
        points = []
        i = self.radial_amount
        while i != 0:
            point = random.randint(0,len(self.data)-1)
            if point in points:
                point = random.randint(0,len(self.data)-1)
            points.append(point)
            i = i - 1
        self.radials = [ Radial( self.data[point][0], 0, num ) for num, point in enumerate(points) ]

    def new_xy_na4(self):
        self.radials.clear()
        self.radials = [ Radial( random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y), i ) for i in range(self.radial_amount) ]

    def allocate_data(self):
        for one_data in self.data:
            distance = []
            for radial in self.radials:
                distance.append( radial.get_distance(one_data) )
            self.radials[ np.argmin(distance, axis=0) ].data.append(one_data)

    def clear_center_data(self):
        for radial in self.radials:
            radial.data.clear()

    def update_centers(self):
        for radial in self.radials:
            radial.update_xy()

    def save_centers_xy(self):
        self.last_radials_xy.clear()
        for radial in self.radials:
            self.last_radials_xy.append(radial.get_xy())

    def set_data(self, data):
        self.data = data

    def get_centers_error(self):
        error = 0
        for center in self.radials:
            error += center.get_error()
        return error/len(self.data)
        
    def set_best_centers_xy(self):
        self.radials.clear()
        for num, xy in enumerate(self.last_radials_xy):
            self.radials.append( Radial(xy[0], xy[1], num) )
        self.allocate_data()

    def print_center_and_data(self, path, name, title):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        for radial in self.radials:
            plt.scatter(radial.get_dat_x(), radial.get_dat_y())
            plt.scatter(radial.get_x(), radial.get_y(), color='r', marker='^')
        name1 = os.path.join(path, str(name) + "_radial.png")
        plt.savefig(name1)
        plt.close()

    def train(self, experiments_amount, epoches, neighbour_amount, epoches_output_layer, path, name, title1, title2, title3):
        for i in range(experiments_amount):
            ################################ na 3 lub na 4 ######################################################
            self.new_xy_na3()
            #####################################################################################################
            self.clear_center_data()
            self.allocate_data()
            new_error = self.get_centers_error()
            if (self.last_error > new_error) or (self.last_error == 0):
                last_error2 = self.last_error
                self.last_error = new_error
                self.save_centers_xy()
        self.set_best_centers_xy()
        self.print_center_and_data(path, name, title2)
        ######################## k-średnich #####################################################################
        # for i in range(experiments_amount):
        #     self.new_xy_na3()
        #     for epo in range(epoches):
        #         self.clear_center_data()
        #         self.allocate_data()
        #         self.update_centers() 
        #         new_error = self.get_centers_error()
        #         if (self.last_error_for_epo - 0.00001 < new_error) and (self.last_error_for_epo != 0):
        #             # print(epo)
        #             break
        #         self.last_error_for_epo = new_error
        #     self.last_error_for_epo = 0
        #     new_error = self.get_centers_error()
        #     if (self.last_error > new_error) or (self.last_error == 0):
        #         last_error2 = self.last_error
        #         self.last_error = new_error
        #         self.save_centers_xy()
        # self.set_best_centers_xy()
        # self.print_center_and_data(path, name)
        ###################### dobór parametru #################################################################
        self.parameter_set(neighbour_amount)
        ###################### wyjściowa warstwa ###############################################################
        for e in range(epoches_output_layer):
            #random.shuffle(self.data)
            self.sum_error = 0
            for point in self.data:
                for radial in self.radials:
                    radial.gauss(point[0])
                self.output_layer.predict(self.radials)
                self.output_layer.output_layer_factor(point[1])
                self.output_layer.update_weights(self.radials)
                self.safe_error(point[1])
            self.errorX.append(e)
            self.errorY.append(self.sum_error/len(self.data))
        self.print_error(path, name, title1)
        self.print_fun_points(path, name, title3)

    def print_fun_points(self, path, name, title):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        X = []
        Y = []
        for i in self.data:
            X.append(i[0])
            Y.append(i[1])
        plt.scatter(X, Y, color='b')
        xx = []
        yy = []
        a = 0.01
        b = -4
        for i in range(800):
            xx.append([b])
            b += a
        for x in xx:
            for r in self.radials:
                r.gauss(x)
            self.output_layer.predict(self.radials)
            yy.append(self.output_layer.output_value)
        plt.plot(xx, yy, color='r', linewidth=3.0)
        name1 = os.path.join(path, str(name) + "_nauka.png")
        plt.savefig(name1)
        plt.close()
        
    def print_error(self, path, name, title):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(0,2)
        plt.title(title)
        plt.plot(self.errorX, self.errorY)
        name1 = os.path.join(path, str(name) + "_error.png")
        plt.savefig(name1)
        plt.close()

    def parameter_set(self, neighbour_amount):
        for radial in self.radials:
            close_nei = []
            x = radial.get_x()
            for nei in self.radials:
                close_nei.append([nei.get_len(x), nei.number])
            close_nei.sort()
            new_param = 0
            for nei in close_nei[1:neighbour_amount+1]:
                new_param += math.pow( nei[0] , 2 )
            new_param = math.sqrt(new_param/neighbour_amount)
            radial.set_param(new_param)

    def test(self, test_data, path, name, title):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        for point in test_data:
            plt.scatter(point[0], point[1], color='b')
        self.sum_error = 0
        for point in test_data:
            for radial in self.radials:
                radial.gauss(point[0])
            self.output_layer.predict(self.radials)
            self.safe_error(point[1])
        xx = []
        yy = []
        a = 0.01
        b = -4
        for i in range(800):
            xx.append([b])
            b += a
        for x in xx:
            for r in self.radials:
                r.gauss(x)
            self.output_layer.predict(self.radials)
            yy.append(self.output_layer.output_value)
        plt.plot(xx, yy, color='r', linewidth=3.0)
        name1 = os.path.join(path, str(name) + "_test.png")
        plt.savefig(name1)
        plt.close()
        name2 = os.path.join(path, str(name) + "_test_error.txt")
        error_save = open(name2,"w+")
        error_save.write(str(self.sum_error/len(test_data)))
        error_save.close()
    

    def safe_error(self, out):
        error = self.output_layer.output_value - out
        self.sum_error += error*error/2
