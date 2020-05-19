def possible_moves(node, game_map):
    """
    Finds the possible neighbors of a node given the map
    """
    y, x = node
    neighs = []

    for i in [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]:  # Creates the list of tuples of possible neighbors
        if 0 <= i[0] < len(game_map) and 0 <= i[1] < len(game_map[0]):  # If it's inside the map, keep it
            neighs.append(i)  # add it to neighbors
    neighs = list(filter(lambda x: game_map[x[0]][x[1]] > -99, neighs))  # filter out walls
    return list(neighs)  # force it to be a list


def bellman(node, game_map, gamma=0.9):
    """
    This gives you the expected reward for a given best action at a node.
    Naively explores the graph with a whatever-first search, returning the best expected value for each choice.
    Not a good example of how this really works, mostly just a math demo!
    """
    print(node)
    y, x = node  # Unpack the node tuple

    if game_map[y][x] == 1 or game_map[y][x] == -1:
        print('Reward found!', str(float(game_map[y][x]) * gamma))
        return float(game_map[y][x]) * gamma  # If it's a reward, return it times gamma

    expected = []

    for neigh in possible_moves(node, game_map):
        temp_map = [[point for point in submap] for submap in game_map]  # deep copy the stupid map
        temp_map[y][x] = -99  # Mark the node on the map

        expected.append(bellman(neigh, temp_map, gamma) * gamma)
        print(f"Back to {node}")
    return float(max(expected)) if len(expected) else -999  # return huge neg if none found


# Map visual - https://i.imgur.com/jd1uMVq.png
# Video credit - https://www.youtube.com/watch?v=14BfO5lMiuk
game_map = [[0, 0, 0, 1],
            [0, -99, 0, -1],
            [0, 0, 0, 0]]
# Note here that the coordinates are functionally backwards and upside down-
# 0,0 is the top left corner, 3,2 is the bottom right.

print(bellman((2, 0), game_map))  # change the point to see how it varies
# I left the print statements so you can see how the graph is explored
