from BaseAI import BaseAI
import random
import math


class PlayerAI(BaseAI):
    def getMove(self, grid):
        return alphaBetaSearch(grid)


def alphaBetaSearch(state):
    max_val = -1
    move_choice = None
    working_grid = state.clone()

    for move in get_ordered_moves(state):
        working_grid.move(move)

        if move == 0:
            val = 0
        elif move == 1:
            val = minValue(working_grid, -math.inf, math.inf, 0)
        elif move == 2 and bottom_is_full and bottom_is_ordered:
            val = minValue(working_grid, -math.inf, math.inf, 0)
        elif move == 2 and bottom_is_full and not bottom_is_ordered:
            val = minValue(working_grid, -math.inf, math.inf, 0) + 10
        elif move == 2 and not bottom_is_full:
            val = 0
        elif move == 3:
            val = minValue(working_grid, -math.inf, math.inf, 0)

        if val > max_val:
            max_val = val
            move_choice = move

        working_grid = state.clone()
    return move_choice


def maxValue(state, a, b, depth):
    if terminalTest(state, depth, 'max'):
        return state_score(state)

    val = -math.inf
    working_grid = state.clone()

    for move in get_ordered_moves(state):
        working_grid.move(move)
        val = max(val, minValue(working_grid, a, b, depth + 1))

        if val >= b:
            return val

        working_grid = state.clone()

    a = max(a, val)

    return val


def minValue(state, a, b, depth):
    if terminalTest(state, depth, 'min'):
        return state_score(state)

    val = math.inf
    working_grid = state.clone()

    for cell in state.getAvailableCells():
        working_grid.insertTile(cell, 2)
        val = min(val, maxValue(working_grid, a, b, depth + 1))

        if val <= a:
            return val

        working_grid = state.clone()

    b = min(b, val)

    return val


def terminalTest(state, depth, min_max):
    if depth >= 3:
        return True
    elif min_max == 'max':
        return not state.canMove()
    elif min_max == 'min':
        return not len(state.getAvailableCells()) > 0


def state_score(state):
    if bottom_is_full and bottom_is_ordered:
        score = 100
    else:
        score = 0

    if state.getCellValue([3, 3]) == state.getMaxTile():
        score += 1000
    score += len(state.getAvailableCells()) * 3
    return score


def bottom_is_full(state):
    for i in range(state.size):
        cell_value = state.getCellValue([3, i])

        if cell_value == 0:
            return False
    return True


def bottom_is_ordered(state):
    for i in range(state.size):
        this_cell_value = state.getCellValue([3, i])
        last_cell_value = state.getCellValue([3, i - 1])

        if last_cell_value == None:
            last_cell_value = 0

        if this_cell_value < last_cell_value:
            return False
    return True


def get_ordered_moves(state):
    moves = state.getAvailableMoves()
    ordered_moves = []

    if 1 in moves:
        ordered_moves.append(1)
    if 3 in moves:
        ordered_moves.append(3)
    if 2 in moves:
        ordered_moves.append(2)
    if 0 in moves:
        ordered_moves.append(0)

    return ordered_moves
