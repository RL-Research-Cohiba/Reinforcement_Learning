"""

Markov models and Markov chains explained in real life:
probabilistic workout routine

https://towardsdatascience.com/markov-models-and-markov-chains-explained-in-real-life-probabilistic-workout-routine-65e47b5c9a73

"""

import numpy as np

def run_markov_chain(transition_matrix, n=10, print_transitions=False):
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
    step = transition_matrix
    
    for time_step in range(1, n):
          if print_transitions:
                print("Transition Matrix at step:" + str(time_step))
                print(step)
                print('--------------------------')
                
          next_step = np.matmul(step, transition_matrix).round(2)
          
          if np.array_equal(step, next_step):
                print('Markov chain reached steady-state at time-step = '
                      + str(time_step))
                
                if not print_transitions:
                      print(step)
                      
                return step
              else:
                    step = next_step
                    
              return step
                
    
transition_matrix = np.array([[0.1, 0.4, 0.3, 0.2],
                              [0.35, 0.1, 0.25, 0.3],
                              [0.4, 0.3, 0.05, 0.25],
                              [0.42, 0.42, 0.08, 0.08]])

power_transition_matrix = run_markov_chain(transition_matrix)
