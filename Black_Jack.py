from collections import namedtuple
from typing import *
import random
import numpy as np
import torch
import torch.nn as nn
from functools import partial
import matplotlib.pyplot as plt

Layer = List[nn.Module]
list_or_int = Union[int, List[int]]
Hand = List[int]
Net = nn.Sequential
Table = Dict[str, List]
TensorLike = torch.tensor
Qtable = NamedTuple(
    "Transition",
    [
        ("state", Hand),
        ("action", int),
        ("next_state", Hand),
        ("reward", int),
    ],
)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relu_layer(input_dim: int, output_dim: int) -> Layer:
    return [nn.Linear(input_dim, output_dim), nn.ReLU()]


def net(input_dim: int, h: list_or_int, output_dim: int) -> Net:
    h = [h] if not isinstance(h, list) else h
    dims = [input_dim] + h + [output_dim]
    dims = [[dims[i], dims[i + 1]] for i in range(len(dims) - 1)]
    layers = [
        relu_layer(*dims[i]) if i != len(dims) - 1 else [nn.Linear(*dims[i])]
        for i in range(len(dims))
    ]
    return nn.Sequential(*sum(layers, []))


def fresh_deck(n_decks: int = 6) -> List:
    _cards = list(range(1, 14)) * n_decks
    _cards = [x if x < 10 else 10 for x in _cards]
    random.shuffle(_cards)
    return _cards


def make_table() -> Table:
    return {"house": fresh_deck(), "dead": [], "dealer": [], "gambler": []}


def give_card(t: Table, player: str) -> Table:
    t[player].append(t["house"].pop(0))
    return t


def start_hand(t: Table) -> Table:
    for _ in range(2):
        t = give_card(t, "gambler")
        t = give_card(t, "dealer")

    if 21 in [score_hand(t[k]) for k in ["dealer", "gambler"]]:
        t = start_hand(t)
    return t


def clear_table(t: Table) -> Table:
    t["dealer"] = []
    t["gambler"] = []
    return t


def end_hand(t: Table) -> Table:
    t["dead"] += t["gambler"] + t["dealer"]
    t = clear_table(t)
    return t


def reset_table(t: Table) -> Table:
    t["house"] += t["dead"]
    t["dead"] = []
    random.shuffle(t["house"])
    return t


def score_hand(h: Hand) -> int:
    tot = sum(h)
    if tot > 21:
        if 11 in h:
            h[h.index(11)] = 1
            return score_hand(h)
        else:
            return -1
    else:
        return tot


def dealer_decide(t: Table) -> int:
    score = score_hand(t["dealer"])
    if score == -1:
        return 0
    elif score >= 17:
        return 0
    else:
        return 1


def pad_hand(h: Hand, padding: int) -> Hand:
    return h + [0] * (padding - len(h))


def get_state(t: Table) -> List[int]:
    # dealer can have at most 17 cards
    dealer_cards = pad_hand(t["dealer"], 17)
    # player can have at most 21 cards
    gambler_cards = pad_hand(t["gambler"], 21)
    state = [dealer_cards] + [gambler_cards]
    return sum(state, [])


def gambler_decide(t: Table, n: Net, epsilon: float) -> int:
    state = get_state(t)
    state = torch.tensor(state, device=device, dtype=torch.float)
    p = random.random()
    if p > epsilon:
        with torch.no_grad():
            return n(state).cpu().numpy().argmax()
    else:
        return random.choice([0, 1])


def create_memory(capacity: int) -> Dict:
    return {"memory": [], "capacity": capacity}


def add_memory(
    memory: Dict, state: Hand, action: int, next_state: Hand, reward: float
) -> Dict:
    args = [state, action, next_state, reward]
    args = [x if isinstance(x, list) else [x] for x in args]
    # args = [torch.tensor(x, device = device) for x in args]
    memory["memory"].append(Transition(*args))
    if len(memory["memory"]) > memory["capacity"]:
        _ = memory["memory"].pop(0)
    return memory


def sample_memory(memory: Dict, batch_size: int) -> List[Qtable]:
    return random.sample(memory["memory"], batch_size)


def generate_batch(memory: Dict, batch_size: int) -> Qtable:
    batch = Transition(*zip(*sample_memory(memory, batch_size)))
    return batch


def q_sa(batch: Qtable, n: Net) -> TensorLike:
    state = torch.tensor(batch.state, device=device, dtype=torch.float)
    action = torch.tensor(batch.action, device=device, dtype=torch.long)
    return n(state).gather(1, action)


def v_sf(batch: Qtable, n: Net, gamma: float) -> TensorLike:
    # do we need to do anything about the final state being the same if the game
    # is over?? I do not think so and we can just hope for nn magic
    next_state = torch.tensor(batch.next_state, device=device, dtype=torch.float)
    next_value = n(next_state).max(1)[0].detach()
    out = (next_value * gamma) + torch.tensor(
        batch.reward, device=device, dtype=torch.float
    ).t()
    return out.t()


# main loop

policy_net = net(38, [500, 500, 2000], 2).to(device)
target_net = net(38, [500, 500, 2000], 2).to(device)
target_net.load_state_dict(policy_net.state_dict())
memory = create_memory(250)
table = make_table()
epsilon_max = 0.9
epsilon_min = 0.1
decay = 1e-4
epochs = 100000
gamma = 0.8
batch_size = 64
optimizer = torch.optim.RMSprop(policy_net.parameters())
losses = []
rewards = []

for e in range(epochs):
    epsilon = epsilon_max * 1 / (1 + decay * e)
    if epsilon < epsilon_min:
        epsilon = epsilon_min
    table = start_hand(table)
    game_over = False
    actions = []
    while not game_over:
        current_state = get_state(table)
        gambler_action = gambler_decide(table, policy_net, epsilon)
        dealer_action = dealer_decide(table)
        if gambler_action == 1:
            table = give_card(table, "gambler")
        if dealer_action == 1:
            table = give_card(table, "dealer")
        next_state = get_state(table)

        scores = {k: score_hand(table[k]) for k in ["gambler", "dealer"]}

        if current_state == next_state:
            # do something to mask this or whatever the silly torch guys did
            game_over = True
            if scores["gambler"] == scores["dealer"]:
                # everyone gets their money
                reward = 0.5
            elif scores["dealer"] < scores["gambler"]:
                # we lost but we didnt bust
                reward = 0
            else:
                reward = 1
        elif scores["gambler"] == -1 and scores["dealer"] != -1:
            game_over = True
            # yikes
            reward = 0
        elif scores["gambler"] == -1 and scores["dealer"] == -1:
            game_over = True
            # we busted! but so did the dealer! tough luck!
            reward = 0
        elif scores["dealer"] == -1 and scores["gambler"] != -1:
            game_over = True
            reward = 1
        else:
            reward = 0

        if game_over:
            table = end_hand(table)
            if len(table["house"]) < 52:
                table = reset_table(table)

        memory = add_memory(memory, current_state, gambler_action, next_state, reward)
        if len(memory["memory"]) <= batch_size:
            continue

        batch = generate_batch(memory, batch_size)
        qsa = q_sa(batch, policy_net)
        vsf = v_sf(batch, target_net, gamma)

        loss = nn.functional.smooth_l1_loss(qsa, vsf)
        if loss.cpu().detach() > 10:
            import pdb; pdb.set_trace()  # XXX BREAKPOINT

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.data.clamp_(-0.5, 0.5)
        optimizer.step()

        if e % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if e % 20 == 0:
            print("\x1bc")
            print(
                """
                  player_hand: {}
                  dealer_hand: {}
                  player_choice: {}
                  result: {},
                  loss: {},
                  epoch: {}
                  """.format(
                    [x for x in current_state[17:] if x != 0],
                    [x for x in current_state[:17] if x != 0],
                    "hit" if gambler_action == 1 else "stay",
                    reward,
                    loss.cpu().detach(),
                    e,
                )
            )

plt.plot(losses)
plt.show()
