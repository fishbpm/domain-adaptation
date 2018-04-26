__author__ = 'Tom Schaul, tom@idsia.ch, Spyridon Samothrakis ssamot@essex.ac.uk'

## ssamot hacked ask/tell interface, algorithmic implementation is from Tom Schaul

from numpy import dot, exp, log, sqrt, ones, zeros_like, Inf, argmax
import numpy as np

output_file = open('snes_output.txt', 'w')


def computeUtilities(fitnesses):
    L = len(fitnesses)
    ranks = zeros_like(fitnesses)
    l = list(zip(fitnesses, range(L)))
    l.sort()
    for i, (_, j) in enumerate(l):
        ranks[j] = i
    # smooth reshaping
    utilities = np.array([max(0., x) for x in log(L / 2. + 1.0) - log(L - np.array(ranks))])
    utilities /= sum(utilities)       # make the utilities sum to 1
    utilities -= 1. / L  # baseline
    return utilities


class SNES():
    def __init__(self, x0, learning_rate_mult, popsize):
        self.x0 = x0
        self.batchSize = popsize
        self.dim = len(x0)
        self.learningRate =  0.2 * (3 + log(self.dim)) / sqrt(self.dim)
        #print self.learningRate
        self.learningRate = self.learningRate*learning_rate_mult
        #self.learningRate = 0.000001
        self.numEvals = 0
        self.bestFound = None
        self.sigmas = ones(self.dim)
        self.bestFitness = -Inf
        self.center = x0.copy()
        self.verbose = True #False

    def ask(self):
        #self.samples = [np.random.randn(self.dim) for _ in range(self.batchSize)]
        self.samples = [np.random.uniform(-1, 1, self.dim) for _ in range(self.batchSize)]
        #print(self.samples)
        asked = [(self.sigmas * s + self.center) for s in self.samples]
        print(' centre:',self.center[:2])#, file=output_file)#
        self.asked = asked
        return asked

    def tell(self, asked, fitnesses):
        samples = self.samples
        #output_file = open('snes_output.txt', 'w')

        assert(np.array_equal(asked, self.asked))
        if max(fitnesses) > self.bestFitness:
            self.bestFitness = max(fitnesses)
            #self.bestFound = samples[argmax(fitnesses)]
            self.bestFound = argmax(fitnesses)
        self.numEvals += self.batchSize
        if self.verbose: print ("Step", self.numEvals/self.batchSize, ":", max(fitnesses), "best:",
                                self.bestFitness, self.bestFound, end='')#, file=output_file)#

        # update center and variances
        utilities = computeUtilities(fitnesses)
        self.center += self.sigmas * dot(utilities, samples)
        covGradient = dot(utilities, [s ** 2 - 1 for s in samples])
        self.sigmas = self.sigmas * exp(0.5 * self.learningRate * covGradient)



if __name__ == "__main__":

    #output_file = open('snes_output.txt', 'w')
    # 100-dimensional ellipsoid function
    dim = 20
    A = np.array([np.power(1000, 2 * i / (dim - 1.)) for i in range(dim)])
    def elli(x):
        return -dot(A * x, x)

    snes = SNES(ones(dim), 2, 400)
    for i in range(0, 10):#00):
        asked = snes.ask()
        #if i > 25 and i < 35:
        #print('asked',asked[0][:2], file=output_file)
        told = [elli(a) for a in asked ]
        #print(told)
        snes.tell(asked,told)


    # # example run
    # print SNES(elli, ones(dim), verbose=True)