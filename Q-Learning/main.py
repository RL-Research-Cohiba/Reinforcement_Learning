transition_probabilities = [ # shape=[s, a, s']
        [[0.7, 0,3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
        [None, [0.8, 0.1, 0.1], None]]

rewards = [ # shape=[s, a, s']
        [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
        [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]

possible_actions = [[0, 1, 2], [0, 2], [1]]

Q_values = np.full((3, 3), np.inf) # -np.inf for imposible possible_actions
for state, actions, in enumerate(possible_actions):
    Q_values[state, actions] = 0.0 # for all possible actions 
