import numpy as np
from itertools import product
from nltk.corpus import treebank

A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])
pi = np.array([0.5, 0.2, 0.3])
O = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
states = UP, DOWN, UNCHANGED = 0, 1, 2
observations = [UP, UP, DOWN]
alpha = np.zeros((len(observations), len(states)))
alpha[:,:] = float('-inf')
backpointers = np.zeros((len(observations), len(states)), 'int')

# The base case for the recursion sets the starting state probs based on pi and generating the observation.
alpha[0, :] = pi * O[:,UP]

# Now for the recursive step, where we maximise over incoming transitions reusing the best incoming score, computed above.
for t1 in states:
    for t0 in states:
        score = alpha[0, t0] * A[t0, t1] * O[t1,UP]
        if score > alpha[1, t1]:
            alpha[1, t1] = score
            backpointers[1, t1] = t0
for t2 in states:
    for t1 in states:
        score = alpha[1, t1] * A[t1, t2] * O[t2,DOWN]
        if score > alpha[2, t2]:
            alpha[2, t2] = score
            backpointers[2, t2] = t1
alpha
np.argmax(alpha[2,:])
backpointers[2,1]
backpointers[1,0]
def viterbi(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    alpha = np.zeros((M, S))
    alpha[:,:] = float('-inf')
    backpointers = np.zeros((M, S), 'int')
    # base case
    alpha[0, :] = pi * O[:,observations[0]]
    # recursive case
    for t in range(1, M):
        for s2 in range(S):
            for s1 in range(S):
                score = alpha[t-1, s1] * A[s1, s2] * O[s2, observations[t]]
                if score > alpha[t, s2]:
                    alpha[t, s2] = score
                    backpointers[t, s2] = s1
    ss = []
    ss.append(np.argmax(alpha[M-1,:]))
    for i in range(M-1, 0, -1):
        ss.append(backpointers[i, ss[-1]])
    return list(reversed(ss)), np.max(alpha[M-1,:])
viterbi((pi, A, O), [UP, UP, DOWN])
viterbi((pi, A, O), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])
def exhaustive(params, observations):
    pi, A, O = params
    M = len(observations)
    S = pi.shape[0]
    # track the running best sequence and its score
    best = (None, float('-inf'))
    # loop over the cartesian product of |states|^M
    for ss in product(range(S), repeat=M):
        # score the state sequence
        score = pi[ss[0]] * O[ss[0],observations[0]]
        for i in range(1, M):
            score *= A[ss[i-1], ss[i]] * O[ss[i], observations[i]]
        # update the running best
        if score > best[1]:
            best = (ss, score)
    return best
exhaustive((pi, A, O), [UP, UP, DOWN])
exhaustive((pi, A, O), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])
### Supervised training, aka "visible" Markov model
# Let's train the HMM parameters on the Penn Treebank, using the sample from NLTK. Note that this is a small fraction of the treebank, so we shouldn't expect great performance of our method trained only on this data.
corpus = treebank.tagged_sents()
word_numbers = {}
tag_numbers = {}
num_corpus = []
for sent in corpus:
    num_sent = []
    for word, tag in sent:
        wi = word_numbers.setdefault(word.lower(), len(word_numbers))
        ti = tag_numbers.setdefault(tag, len(tag_numbers))
        num_sent.append((wi, ti))
    num_corpus.append(num_sent)
word_names = [None] * len(word_numbers)
for word, index in word_numbers.items():
    word_names[index] = word
tag_names = [None] * len(tag_numbers)
for tag, index in tag_numbers.items():
    tag_names[index] = tag
training = num_corpus[:-10]
testing = num_corpus[-10:]
S = len(tag_numbers)
V = len(word_numbers)
eps = 0.1
pi = eps * np.ones(S)
A = eps * np.ones((S, S))
O = eps * np.ones((S, V))
# count
for sent in training:
    last_tag = None
    for word, tag in sent:
        O[tag, word] += 1
        if last_tag != None:
            pi[tag] += 1
        else:
            A[last_tag, tag] += 1
        last_tag = tag
# normalise
pi /= np.sum(pi)
for s in range(S):
    O[s,:] /= np.sum(O[s,:])
    A[s,:] /= np.sum(A[s,:])
predicted, score = viterbi((pi, A, O), map(lambda (w,t): w, testing[0]))
import pdb; pdb.set_trace()
print '%20s\t%5s\t%5s' % ('TOKEN', 'TRUE', 'PRED')
for (wi, ti), pi in zip(testing[0], predicted):
    print '%20s\t%5s\t%5s' % (word_names[wi], tag_names[ti], tag_names[pi])
def forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]
    alpha = np.zeros((N, S))
    alpha[0, :] = pi * O[:,observations[0]]
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                alpha[i, s2] += alpha[i-1, s1] * A[s1, s2] * O[s2, observations[i]]
    return (alpha, np.sum(alpha[N-1,:]))
A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])
pi = np.array([0.5, 0.2, 0.3])
O = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])
forward((pi, A, O), [UP, UP, DOWN])
forward((pi, A, O), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])
def exhaustive_forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]
    total = 0.0
    for ss in product(range(S), repeat=N):
        score = pi[ss[0]] * O[ss[0],observations[0]]
        for i in range(1, N):
            score *= A[ss[i-1], ss[i]] * O[ss[i], observations[i]]
        total += score
    return total
exhaustive_forward((pi, A, O), [UP, UP, DOWN])
exhaustive_forward((pi, A, O), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])
def backward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]
    beta = np.zeros((N, S))
    beta[N-1, :] = 1
    # recursive case
    for i in range(N-2, -1, -1):
        for s1 in range(S):
            for s2 in range(S):
                beta[i, s1] += beta[i+1, s2] * A[s1, s2] * O[s2, observations[i+1]]
    return (beta, np.sum(pi * O[:, observations[0]] * beta[0,:]))
backward((pi, A, O), [UP, UP, DOWN])
def baum_welch(training, pi, A, O, iterations):
    pi, A, O = np.copy(pi), np.copy(A), np.copy(O)
    S = pi.shape[0]
    for it in range(iterations):
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)
        for observations in training:
            # compute forward-backward matrices
            alpha, za = forward((pi, A, O), observations)
            beta, zb = backward((pi, A, O), observations)
            assert abs(za - zb) < 1e-6, "it's badness 10000 if the marginals don't agree"
            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0,:] * beta[0,:] / za
            for i in range(0, len(observations)):
                O1[:, observations[i]] += alpha[i,:] * beta[i,:] / za
            for i in range(1, len(observations)):
                for s1 in range(S):
                    for s2 in range(S):
                        A1[s1, s2] += alpha[i-1,s1] * A[s1, s2] * O[s2, observations[i]] * beta[i,s2] / za
        # normalise pi1, A1, O1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
    return pi, A, O
pi2, A2, O2 = baum_welch([[UP, UP, DOWN]], pi, A, O, 10)
forward((pi2, A2, O2), [UP, UP, DOWN])
forward((pi2, A2, O2), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])
pi3, A3, O3 = baum_welch([[UP, UP, DOWN], [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP]], pi, A, O, 10)
forward((pi3, A3, O3), [UP, UP, DOWN])
forward((pi3, A3, O3), [UP, UP, DOWN, UNCHANGED, UNCHANGED, DOWN, UP, UP])
