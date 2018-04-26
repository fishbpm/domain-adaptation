__author__ = 'Spyridon Samothrakis ssamot@essex.ac.uk'

from snes import SNES
from numpy import dot, ones
import numpy as np
from random import randint

class SSNES():
    def __init__(self, x0, learning_rate_mult, popsize):
        self.snes = SNES(x0,learning_rate_mult,popsize)
        self.gen()

    def gen(self):
        self.asked = self.snes.ask()
        self.scores = {n:[] for n in range(len(self.asked))}


    def predict(self):
        r = randint(0,len(self.asked)-1)
        asked = self.asked[r]
        return asked, r


    def fit(self, scores, r):

        # sort them out
        for i, score in enumerate(scores):
            self.scores[r[i]].append(score)

        told = []
        for i in range(len(self.asked)):
            told.append(np.array(self.scores[i]).mean())

        self.snes.tell(self.asked,told)
        self.gen()

if __name__ == "__main__":

    #output_file = open('snes_output.txt', 'w')
    # 100-dimensional ellipsoid function
    dim = 20
    A = np.array([np.power(1000, 2 * i / (dim - 1.)) for i in range(dim)])
    def elli(x):
        return -dot(A * x, x)

    snes = SSNES(ones(dim), 2, 400)
    for i in range(0, 10):#00):
        asked = snes.asked
        #if i > 25 and i < 35:
        #print('asked',asked[0][:2], file=output_file)
        told = [elli(a) for a in asked ]
        #print(told)
        snes.fit(told, range(400))


    # # example run
    # print SNES(elli, ones(dim), verbose=True)