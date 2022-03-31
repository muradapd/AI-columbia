import sys
from BTS import backtrack

letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
squares = []
variables = {}
domains = {}
unassigned = []


def dict_to_string(algorithm):
    string = ""

    for letter in letters:
        for i in range(9):
            key = letter + str(i + 1)
            value = variables.get(key)
            string = string + str(value)
    string = string + " " + algorithm

    return string


def dict_to_string2(algorithm, assignment):
    string = ""

    for letter in letters:
        for i in range(9):
            key = letter + str(i + 1)
            value = assignment.get(key)
            string = string + str(value)
    string = string + " " + algorithm

    return string


def test():
    for letter in letters:
        for i in range(9):
            key = letter + str(i + 1)
            value = variables.get(key)
            if (value == 0):
                return False
    return True


def write_out(output):
    f = open("output.txt", "w+")
    f.write(output)
    f.close()


def generate_squares():
    for x in range(3):
        for y in range(3):
            square = []
            for i in range(3):
                for j in range(3):
                    x_scale = x * 3
                    y_scale = y * 3
                    square.append(letters[y_scale + i] +
                                  str((x_scale + j) + 1))
            squares.append(square)


def generate_arcs(x, z):
    arcs = []
    x_letter = x[0]
    x_number = x[1]

    for letter in letters:
        if letter != x_letter:
            y = letter + x_number
            if y != z:
                arcs.append((x, y))

    for number in range(9):
        if number + 1 != int(x_number):
            y = x_letter + str(number + 1)
            if y != z:
                arcs.append((x, y))

    for square in squares:
        if x in square:
            for y in square:
                if y != x and y != z and (x, y) not in arcs:
                    arcs.append((x, y))
    return arcs


def revise(arc, x_domain):
    revised = False
    new_x_domain = []
    y_domain = domains.get(arc[1])

    for x_value in x_domain:
        consistent = False
        for y_value in y_domain:
            if x_value != y_value:
                consistent = True
        if consistent:
            new_x_domain.append(x_value)
        else:
            revised = True

    domains.update({arc[0]: new_x_domain})

    return revised


def AC3():
    arcs = []
    done = False
    iters = 0

    for x in variables:
        arcs.extend(generate_arcs(x, None))

    while not done and iters < 10:
        done = True

        while len(arcs) > 0:
            arc = arcs.pop(0)
            x_domain = domains.get(arc[0])
            if revise(arc, x_domain):
                if len(x_domain) == 0:
                    return False
                else:
                    arcs.extend(generate_arcs(arc[0], arc[1]))

        for domain in domains:
            values = domains.get(domain)
            if len(values) == 1:
                variables.update({domain: values[0]})
            else:
                arcs.extend(generate_arcs(domain, None))
                done = False
        iters += 1

    if (test()):
        write_out(dict_to_string("AC3"))
        return True
    else:
        return False


def main():
    puzzle = sys.argv[1]

    generate_squares()

    string_index = 0
    for letter in letters:
        for i in range(9):
            key = letter + str(i + 1)
            number = int(puzzle[string_index])
            variables.update({key: number})

            if number == 0:
                domains.update({key: [1, 2, 3, 4, 5, 6, 7, 8, 9]})
            else:
                domains.update({key: [number]})
            string_index += 1

    success = AC3()

    if not success:
        assignment = {}
        local_domains = {}
        string_index = 0
        for letter in letters:
            for i in range(9):
                key = letter + str(i + 1)
                number = int(puzzle[string_index])

                if number == 0:
                    local_domains.update(
                        {key: [1, 2, 3, 4, 5, 6, 7, 8, 9]})
                else:
                    assignment.update({key: number})
                    local_domains.update({key: [number]})
                string_index += 1
        varbs = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
                 "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
                 "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
                 "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
                 "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9",
                 "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9",
                 "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9",
                 "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
                 "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9",
                 ]
        constraints = [
            ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"],
            ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
            ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
            ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
            ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"],
            ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"],
            ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"],
            ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9"],
            ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "I1"],
            ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "I2"],
            ["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "I3"],
            ["A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4", "I4"],
            ["A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5", "I5"],
            ["A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6", "I6"],
            ["A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7", "I7"],
            ["A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8", "I8"],
            ["A9", "B9", "C9", "D9", "E9", "F9", "G9", "H9", "I9"],
            ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"],
            ["D1", "D2", "D3", "E1", "E2", "E3", "F1", "F2", "F3"],
            ["G1", "G2", "G3", "H1", "H2", "H3", "I1", "I2", "I3"],
            ["A4", "A5", "A6", "B4", "B5", "B6", "C4", "C5", "C6"],
            ["D4", "D5", "D6", "E4", "E5", "E6", "F4", "F5", "F6"],
            ["G4", "G5", "G6", "H4", "H5", "H6", "I4", "I5", "I6"],
            ["A7", "A8", "A9", "B7", "B8", "B9", "C7", "C8", "C9"],
            ["D7", "D8", "D9", "E7", "E8", "E9", "F7", "F8", "F9"],
            ["G7", "G8", "G9", "H7", "H8", "H9", "I7", "I8", "I9"],
        ]
        csp = {}
        csp.update({"variables": varbs})
        csp.update({"domains": local_domains})
        csp.update({"constraints": constraints})
        assignment = backtrack(assignment, csp)

        if assignment != False:
            write_out(dict_to_string2("BTS", assignment))


if __name__ == '__main__':
    main()
