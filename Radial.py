
import math 

class Radial(object):

    def __init__(self, x, y, number):
        self.x = x
        self.y = y
        self.param = 1.0
        self.data = []
        self.number = number
        self.output_value = 0.0

    def get_distance(self, xy):
        return math.sqrt((self.x - xy[0]) * (self.x - xy[0]) + (self.y - xy[1]) * (self.y - xy[1]))

    def update_xy(self):
        if len(self.data) != 0:
            sum_x = 0
            sum_y = 0
            for one_data in self.data:
                sum_x += one_data[0]
                sum_y += one_data[1]
            self.x = sum_x/len(self.data)
            self.y = sum_y/len(self.data)

    def get_xy(self):
        return [self.x, self.y]

    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y

    def get_dat_x(self):
        tab = []
        for one_data in self.data:
            tab.append(one_data[0])
        return tab

    def get_dat_y(self):
        tab = []
        for one_data in self.data:
            tab.append(one_data[1])
        return tab

    def get_error(self):
        error = 0
        for one_data in self.data:
            error += self.get_distance(one_data)
        return error

    def get_param(self):
        return self.param

    def set_param(self, new_param):
        self.param = new_param

    def get_len(self, x):
        w = abs(self.x - x)
        return w

    def gauss(self, x):
        self.output_value = math.exp(- math.pow( self.get_len(x), 2 ) / ( 2 * math.pow( self.param, 2 ) ) )