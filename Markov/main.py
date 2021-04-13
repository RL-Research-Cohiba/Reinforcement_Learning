"""

Markov models and Markov chains explained in real life:
probabilistic workout routine

https://towardsdatascience.com/markov-models-and-markov-chains-explained-in-real-life-probabilistic-workout-routine-65e47b5c9a73

"""

import numpy as np
    """
    Takes the transition matrix and runs through each state of the Markov
    chain for n time steps. When the chain reaches a steady state, returns
    the transition probabilities and the time step of the convergence.
   
    @params:
    - transition matrix: transition probabilities
    - n: number of time steps to run. default is 10 steps
    - print_transitions: tells if we want to print the transition matrix at
      each time step
"""
    
transition_matrix = np.array([[0.1, 0.4, 0.3, 0.2],
                              [0.35, 0.1, 0.25, 0.3],
                              [0.4, 0.3, 0.05, 0.25],
                              [0.42, 0.42, 0.08, 0.08]])
