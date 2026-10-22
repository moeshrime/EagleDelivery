from collections import deque
import heapq
from itertools import count


class SearchResult:
    def __init__(self, path, expanded):
        self.path = path
        self.expanded = expanded


def reconstruct_path(parents, goal):
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = parents[state]
    path.reverse()
    return path


def bfs(campus, start, goal):
    frontier = deque([start])
    parents = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)
        expanded += 1
        for action, successor in campus.neighbors(state):
            if successor not in parents:
                parents[successor] = state
                frontier.append(successor)
    return SearchResult(None, expanded)


def manhattan(state, goal):
    """TODO 1: Return Manhattan distance between two (row, column) tuples."""
    # h(s) = |r_s - r_g| + |c_s - c_g|
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def astar(campus, start, goal):
    """TODO 2: Return SearchResult(path, expanded) for A* graph search."""
    if start == goal:
        return SearchResult([start], 0)

    insertion = count()          # breaks equal-f ties by insertion order
    best_g = {start: 0}          # best known path cost g(s) to each state
    parents = {start: None}      # parent on the best known path
    frontier = []                # heap of (f, insertion number, g, state)
    heapq.heappush(frontier, (manhattan(start, goal), next(insertion), 0, start))
    expanded = 0

    while frontier:
        f, order, g, state = heapq.heappop(frontier)
        if g != best_g[state]:
            continue             # stale entry: a cheaper route was found later
        if state == goal:        # goal test when a node is removed
            return SearchResult(reconstruct_path(parents, goal), expanded)
        expanded += 1            # count the node as we generate its successors
        for action, successor in campus.neighbors(state):
            new_g = g + 1        # every traversable move costs 1
            if successor not in best_g or new_g < best_g[successor]:
                best_g[successor] = new_g
                parents[successor] = state
                new_f = new_g + manhattan(successor, goal)
                heapq.heappush(frontier, (new_f, next(insertion), new_g, successor))
    return SearchResult(None, expanded)


def epsilon_greedy(q_values, state, actions, epsilon, rng):
    """TODO 3: Choose an action with epsilon-greedy exploration."""
    # With probability epsilon, explore: pick any of the actions uniformly.
    if rng.random() < epsilon:
        return rng.choice(actions)
    # Otherwise exploit: pick an action with the largest Q(state, action).
    values = [q_values.get((state, a), 0.0) for a in actions]
    best_value = max(values)
    best_actions = [a for a, v in zip(actions, values) if v == best_value]
    return rng.choice(best_actions)  # break ties uniformly at random


def q_learning_update(q_values, state, action, reward, next_state,
                      actions, alpha, gamma, terminal):
    """TODO 4: Update exactly one Q-table entry in place; return nothing."""
    old_value = q_values.get((state, action), 0.0)
    if terminal:
        target = reward          # no continuation value after delivery
    else:
        best_next = max(q_values.get((next_state, a), 0.0) for a in actions)
        target = reward + gamma * best_next
    q_values[(state, action)] = old_value + alpha * (target - old_value)
