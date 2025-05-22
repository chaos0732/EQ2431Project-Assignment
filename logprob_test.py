from PattRecClasses import DiscreteD, GaussD, HMM, MarkovChain
from matplotlib import pyplot as plt

import numpy as np

# test forward algorithm

q = np.array([1, 0])
A = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])

g1 = GaussD(means=[0], stdevs=[1])  # Distribution for state = 1
g2 = GaussD(means=[3], stdevs=[2])  # Distribution for state = 2
mc = MarkovChain(q, A)
h = HMM(mc, [g1, g2])

x = np.array([-0.2, 2.6, 1.3])


logp = h.logprob(x)
print("logp:", logp)