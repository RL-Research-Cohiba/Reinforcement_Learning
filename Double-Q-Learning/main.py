import numpy as np
import matplotlib.pyplot as plt


# these are the control variables, change them to customize the execution of this program
# number of experiments to run, large number means longer execution time
cntExperiments = 1001
# number of episodes per experiment, large number means longer execution time
MAX_ITER = 301
ACTIONS_FOR_B = 10  # number of actions at state B


# identify the states
STATE_A = 0
STATE_B = 1
STATE_C = 2
STATE_D = 3

# identify the actions
ACTION_LEFT = 0
ACTION_RIGHT = 1


# map actions to states
actionsPerState = {}
actionsPerState[STATE_A] = [ACTION_LEFT, ACTION_RIGHT]
actionsPerState[STATE_B] = [i for i in range(ACTIONS_FOR_B)]
actionsPerState[STATE_C] = [ACTION_RIGHT]
actionsPerState[STATE_D] = [ACTION_LEFT]

# init Q values
Q1 = {}
Q2 = {}


GAMMA = 1


# reset the variables, to be called on each experiment
def reset():
    Q1[STATE_A] = {}
    Q1[STATE_A][ACTION_LEFT] = 0
    Q1[STATE_A][ACTION_RIGHT] = 0

    Q1[STATE_B] = {}

    Q1[STATE_C] = {}
    Q1[STATE_C][ACTION_LEFT] = 0
    Q1[STATE_C][ACTION_RIGHT] = 0

    Q1[STATE_D] = {}
    Q1[STATE_D][ACTION_LEFT] = 0
    Q1[STATE_D][ACTION_RIGHT] = 0

    Q2[STATE_A] = {}
    Q2[STATE_A][ACTION_LEFT] = 0
    Q2[STATE_A][ACTION_RIGHT] = 0

    Q2[STATE_B] = {}

    Q2[STATE_C] = {}
    Q2[STATE_C][ACTION_LEFT] = 0
    Q2[STATE_C][ACTION_RIGHT] = 0

    Q2[STATE_D] = {}
    Q2[STATE_D][ACTION_LEFT] = 0
    Q2[STATE_D][ACTION_RIGHT] = 0
    for i in range(ACTIONS_FOR_B):
        Q1[STATE_B][i] = 0
        Q2[STATE_B][i] = 0


# epsilon greedy action
# it return action a 1-epsilon times
# and a random action epsilon times
def random_action(s, a, eps=.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(actionsPerState[s])


# move from state s using action a
# it returns the reward and the new state
def move(s, a):
    if(s == STATE_A):
        if(a == ACTION_LEFT):
            return 0, STATE_B
        else:
            return 0, STATE_C
    if s == STATE_B:
        return np.random.normal(-.5, 1), STATE_D
    return 0, s

# returns the action that makes the max Q value, as welle as the max Q value


def maxQA(q, s):
    max = -9999
    sa = 0
    for k in q[s]:
        if(q[s][k] > max):
            max = q[s][k]
            sa = k
    return sa, max

# return true if this is a terminal state


def isTerminal(s):
    return s == STATE_C or s == STATE_D

# select the initial action at state A, it uses greedy method
# it takes into the mode doubleQLearning or not


def selectInitialAction(doubleQLearning, startState):
    if doubleQLearning:
        QS = {}
        QS[STATE_A] = {}
        QS[STATE_A][ACTION_LEFT] = Q1[STATE_A][ACTION_LEFT] + \
            Q2[STATE_A][ACTION_LEFT]
        QS[STATE_A][ACTION_RIGHT] = Q1[STATE_A][ACTION_RIGHT] + \
            Q2[STATE_A][ACTION_RIGHT]
        a, _ = maxQA(QS, startState)

    else:
        a, _ = maxQA(Q1, startState)
    return a

# update Q values depending on whether the mode  is doubleQLearning or not


def updateQValues(doubleQLearning, s, a, r, nxt_s, alpha):
    if doubleQLearning:
        p = np.random.random()
        if (p < .5):
            nxt_a, maxq = maxQA(Q1, nxt_s)
            Q1[s][a] = Q1[s][a] + alpha * \
                (r + GAMMA * Q2[nxt_s][nxt_a] - Q1[s][a])
        else:
            nxt_a, maxq = maxQA(Q2, nxt_s)
            Q2[s][a] = Q2[s][a] + alpha * \
                (r + GAMMA * Q1[nxt_s][nxt_a] - Q2[s][a])
    else:
        nxt_a, maxq = maxQA(Q1, nxt_s)
        Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * maxq - Q1[s][a])
    return nxt_a


# do the experiment by running MAX_ITER episodes and fill the restults in the episods parameter
def experiment(episods, doubleQLearning=False):
    reset()
    # contains the number of times left action is chosen at A
    ALeft = 0

    # contains the number of visits for each state
    N = {}
    N[STATE_A] = 1
    N[STATE_B] = 1
    N[STATE_C] = 1
    N[STATE_D] = 1

    # contains the number of visits for each state and action
    NSA = {}

    # loop for MAX_ITER episods
    for i in range(1, MAX_ITER):

        s = STATE_A
        gameover = False

        # use greedy for the action at STATE A
        a = selectInitialAction(doubleQLearning, s)

        # loop until game is over, this will be ONE episode
        while not gameover:

            # apply epsilon greedy selection (including for action chosen at STATE A)
            a = random_action(s, a, 1/np.sqrt(N[s]))

            # update the number of visits for state s
            N[s] += 1

            # if left action is chosen at state A, increment the counter
            if (s == STATE_A and a == ACTION_LEFT):
                ALeft += 1

            # move to the next state and get the reward
            r, nxt_s = move(s, a)

            # update the number of visists per state and action
            if not s in NSA:
                NSA[s] = {}
            if not a in NSA[s]:
                NSA[s][a] = 0
            NSA[s][a] += 1

            # compute alpha
            alpha = 1 / np.power(NSA[s][a], .8)

            # update the Q values and get the best action for the next state
            nxt_a = updateQValues(doubleQLearning, s, a, r, nxt_s, alpha)

            # if next state is terminal then mark as gameover (end of episode)
            gameover = isTerminal(nxt_s)

            s = nxt_s
            a = nxt_a

        # update stats for each episode
        if not (i in episods):
            episods[i] = {}
            episods[i]["count"] = 0
            episods[i]["Q1(A)"] = 0
            episods[i]["Q2(A)"] = 0
        episods[i]["count"] = ALeft
        episods[i]["percent"] = ALeft / i
        episods[i]["Q1(A)"] = ((episods[i]["Q1(A)"] * (i-1)) +
                               Q1[STATE_A][ACTION_LEFT])/i
        episods[i]["Q2(A)"] = ((episods[i]["Q2(A)"] * (i-1)) +
                               Q2[STATE_A][ACTION_LEFT])/i


# init a report structure
def initReport(report):
    for i in range(1, MAX_ITER):
        report[i] = {}
        report[i]["steps"] = i
        report[i]["count"] = 0
        report[i]["percent"] = 0
        report[i]["Q1(A)"] = 0
        report[i]["Q2(A)"] = 0


# run the learning
def runLearning(dblQLearn, report, experimentsCount):
    # run batch of experiments
    for k in range(1, experimentsCount):
        tmp = {}
        experiment(tmp, dblQLearn)
        # aggregate every experiment result into the final report
        for i in report:
            report[i]["count"] = (
                (report[i]["count"] * (k-1)) + tmp[i]["count"])/k
            report[i]["percent"] = 100*report[i]["count"] / i
            report[i]["Q1(A)"] = (
                (report[i]["Q1(A)"] * (k-1)) + tmp[i]["Q1(A)"])/k
            report[i]["Q2(A)"] = (
                (report[i]["Q2(A)"] * (k-1)) + tmp[i]["Q2(A)"])/k


# print the report
def printReport(dblQLearn, report):
    # display the final report
    print("Double Q Learning" if dblQLearn else "Q learning")
    for i in report:
        if(i == 1 or i % 10 == 0):
            print(i, ", ", report[i]["percent"], ", ",
                  report[i]["Q1(A)"], ", ", report[i]["Q2(A)"])

# draw graphs of both curves QL and Double QL


def drawGraph(reportQl, reportDQl):
    steps = []
    yQL = []
    yDQL = []
    for i in reportQl:
        steps.append(i)
        yQL.append(reportQl[i]["percent"])
        yDQL.append(reportDQl[i]["percent"])

    df = {'steps': steps, 'yQL': yQL, 'yDQL': yDQL}
    # multiple line plot
    plt.plot('steps', 'yQL', data=df, marker='',
             color='red', linewidth=1, label="Q-Learning")
    plt.plot('steps', 'yDQL', data=df, marker='', color='blue',
             linewidth=1, label="Double Q-Learning")
    plt.legend()
    plt.title("Double Q-Learning vs Q-Learning")
    plt.show()


# main ----------------------------------
# init report variables that will hold all the results
reportQL = {}
reportDQL = {}

initReport(reportQL)
initReport(reportDQL)

# run and print QLearning
runLearning(False, reportQL, cntExperiments)
printReport(False, reportQL)

# run and print Double QLearning
runLearning(True, reportDQL, cntExperiments)
printReport(True, reportDQL)

# print graphs
drawGraph(reportQL, reportDQL)
