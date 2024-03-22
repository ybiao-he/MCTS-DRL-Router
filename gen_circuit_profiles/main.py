import numpy as np
import random
from collections import defaultdict

class PCBLayout(object):
    """ Now, we consider a simple version of ciruits, the generated circuits do not ahve obstacles;
        each net only contains two pins, the optimization goal is to find the shortest path and to have least vias.

    """
    def __init__(self, size=(60,60), num_nets=20, num_layers=2):
        # define circuit informations
        self.size = size
        self.num_nets = num_nets
        self.nets = defaultdict(list)
        self.num_layers = num_layers
        self.num_obstacles = None  # do not consider this now
        self.num_pins = list(range(2,6))

    def write_profile(self, filename):

        f = open(filename, "w")
        f.write('GridBoundaryIdx 1 1 {} {}\n'.format(self.size[0], self.size[1]))
        f.write('NumLayer {}\n'.format(self.num_layers))
        f.write('NumNets {}\n'.format(self.num_nets))

        for i in range(1, self.num_nets+1):
            f.write('Net N{} {}\n'.format(i, len(self.nets[i])))
            for pin in self.nets[i]:
                f.write('Pin {} {} {}\n'.format(pin[0], pin[1], pin[2]))

        f.close()

    def generate_nets(self):

        pins = set()  # use this to avoid two pins are too close
        for net_idx in range(1, self.num_nets+1):
            num_pins = random.choice(self.num_pins)
            for i in range(num_pins):
                pin = self.generate_one_pin(pins)
                pins.add(pin)
                self.nets[net_idx].append(pin)

    def generate_one_pin(self, pins):

        directions = [(0,1,0), (0,-1,0), (1,0,0), (-1,0,0), (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0)]
        while True:
            legal = True
            pin = np.array([np.random.randint(1, self.size[i]-1) for i in range(2)])
            pin = np.append(pin, [0])
            for d in directions:
                tmp = np.array(d)+pin
                if tuple(tmp) in pins or tuple(pin) in pins:
                    legal = False
                    break
            if not legal:
                print("generated pins are not allowed!!")
            if legal:
                # all the pins are placed in the first layer, thus +(0,)
                return tuple(pin)

    def generate_profile(self, filename):

        self.generate_nets()

        self.write_profile(filename)


if __name__== "__main__":

    filename = "case4.txt"
    PCB = PCBLayout((88,72), 80, 2)
    PCB.generate_profile(filename)
    print(PCB.nets)
