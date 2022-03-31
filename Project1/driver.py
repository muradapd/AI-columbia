"""
Skeleton code for Project 1 of Columbia University's AI EdX course (8-puzzle).
Python 3
"""
import queue as Q
import time
import resource
import sys
import math
import heapq


class PuzzleState(object):
    """docstring for PuzzleState"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        if n*n != len(config) or n < 2:
            raise Exception("the length of config is not correct!")

        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []

        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i // self.n
                self.blank_col = i % self.n
                break

    def __lt__(self, other):
        return calculate_total_cost(self) < calculate_total_cost(other)

    def display(self):
        for i in range(self.n):
            line = []

            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print(line)

    def move_left(self):
        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):
        if self.blank_col == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]
            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        """expand the node"""
        # add child nodes in order of UDLR
        if len(self.children) == 0:
            up_child = self.move_up()
            if up_child is not None:
                self.children.append(up_child)

            down_child = self.move_down()
            if down_child is not None:
                self.children.append(down_child)

            left_child = self.move_left()
            if left_child is not None:
                self.children.append(left_child)

            right_child = self.move_right()
            if right_child is not None:
                self.children.append(right_child)
        return self.children


def writeOutput(path, cost, nodes, depth, max_depth, run_time, memory):
    lines = []
    lines.append("path_to_goal: " + str(path))
    lines.append("\ncost_of_path: " + str(cost))
    lines.append("\nnodes_expanded: " + str(nodes))
    lines.append("\nsearch_depth: " + str(depth))
    lines.append("\nmax_search_depth: " + str(max_depth))
    lines.append("\nrunning_time: " + str(run_time))
    lines.append("\nmax_ram_usage: " + str(memory))
    output = open("output.txt", "w")
    output.writelines(lines)
    output.close()


def get_path(state):
    actions = []

    while state.action is not 'Initial':
        actions.insert(0, state.action)
        state = state.parent
    return actions


def bfs_search(initial_state):
    start_time = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    curr_mem = 0
    max_mem = 0
    frontier = [initial_state]
    frontier_set = {initial_state.config}
    explored = set()
    max_depth = 0

    while len(frontier) > 0:
        curr_mem = (resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss) - start_mem
        if curr_mem > max_mem:
            max_mem = curr_mem
        state = frontier.pop(0)
        frontier_set.remove(state.config)
        explored.add(state.config)

        if goal_test(state):
            run_time = time.time() - start_time
            path = get_path(state)
            depth = state.cost
            return writeOutput(path, state.cost, len(explored)-1, depth, max_depth, run_time, max_mem / 1000000)

        for neighbor in state.expand():
            if neighbor.config not in explored and neighbor.config not in frontier_set:
                if neighbor.cost > max_depth:
                    max_depth = neighbor.cost
                frontier.append(neighbor)
                frontier_set.add(neighbor.config)
    return print("Failure")


def dfs_search(initial_state):
    start_time = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    curr_mem = 0
    max_mem = 0
    frontier = [initial_state]
    frontier_set = {initial_state.config}
    explored = set()
    max_depth = 0

    while len(frontier) > 0:
        curr_mem = (resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss) - start_mem
        if curr_mem > max_mem:
            max_mem = curr_mem
        state = frontier.pop()
        frontier_set.remove(state.config)
        explored.add(state.config)

        if goal_test(state):
            run_time = time.time() - start_time
            path = get_path(state)
            depth = state.cost
            return writeOutput(path, state.cost, len(explored)-1, depth, max_depth, run_time, max_mem / 1000000)

        neighbors = state.expand()
        neighbors.reverse()
        for neighbor in neighbors:
            if neighbor.cost > max_depth:
                max_depth = neighbor.cost
            if neighbor.config not in explored and neighbor.config not in frontier_set:
                frontier.append(neighbor)
                frontier_set.add(neighbor.config)
    return print("Failure")


def A_star_search(initial_state):
    start_time = time.time()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    curr_mem = 0
    max_mem = 0
    frontier = []
    heapq.heappush(frontier, initial_state)
    frontier_set = {initial_state.config}
    explored = set()
    max_depth = 0

    while len(frontier) > 0:
        curr_mem = (resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss) - start_mem
        if curr_mem > max_mem:
            max_mem = curr_mem
        state = heapq.heappop(frontier)
        frontier_set.remove(state.config)
        explored.add(state.config)

        if goal_test(state):
            run_time = time.time() - start_time
            path = get_path(state)
            depth = state.cost
            return writeOutput(path, state.cost, len(explored)-1, depth, max_depth, run_time, max_mem / 1000000)

        for neighbor in state.expand():
            if neighbor.cost > max_depth:
                max_depth = neighbor.cost
            if neighbor.config not in explored and neighbor.config not in frontier_set:
                heapq.heappush(frontier, neighbor)
                frontier_set.add(neighbor.config)
            elif neighbor.config in frontier_set:
                # calc new total cost and replace in heap if less than existing cost
                existing_state = next(
                    (x for x in frontier if x.config == neighbor.config), None)
                if (calculate_total_cost(neighbor) < calculate_total_cost(existing_state)):
                    frontier.remove(existing_state)
                    heapq.heappush(frontier, neighbor)
    return print("Failure")


def calculate_total_cost(state):
    return state.cost + calculate_manhattan_dist(state)


def calculate_manhattan_dist(state):
    initial_config = state.config
    dist = 0
    i = 0
    while i < len(initial_config):
        curr_row = int(i / 3)
        curr_col = i % 3
        goal_row = int(initial_config[i] / 3)
        goal_col = initial_config[i] % 3
        dist += abs(curr_row - goal_row) + abs(curr_col - goal_col)
        i += 1
    return dist


def goal_test(puzzle_state):
    """test the state is the goal state or not"""
    if puzzle_state.config == (0, 1, 2, 3, 4, 5, 6, 7, 8):
        return True
    else:
        return False


# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    sm = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size)

    if sm == "bfs":
        bfs_search(hard_state)
    elif sm == "dfs":
        dfs_search(hard_state)
    elif sm == "ast":
        A_star_search(hard_state)
    else:
        print("Enter valid command arguments !")


if __name__ == '__main__':
    main()
