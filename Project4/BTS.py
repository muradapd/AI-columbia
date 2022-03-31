
def backtrack(assignment, csp):
    update_domains(assignment, csp)
    return BTS(assignment, csp)


def BTS(assignment, csp):
    if check_constraints(assignment, csp) and len(assignment) == 81:
        return assignment

    variable = get_mrv(assignment, csp)
    neighbors = get_neighbors(variable, csp)
    values = csp.get("domains").get(variable).copy()

    for value in values:
        assignment.update({variable: value})
        # csp.get("domains").update({variable: [value]})
        if is_consistent(variable, assignment, neighbors):
            if forward_check(value, csp, neighbors):
                success = BTS(assignment, csp)
                if success != False:
                    return success
        assignment.pop(variable)
        update_domains(assignment, csp)
    return False


def update_domains(assignment, csp):
    variables = csp.get("variables")

    for variable in variables:
        assigned = []
        domain = []

        if assignment.get(variable) == None:
            neighbors = get_neighbors(variable, csp)

            for neighbor in neighbors:
                if assignment.get(neighbor) not in assigned:
                    assigned.append(assignment.get(neighbor))
            for i in range(10):
                if i > 0 and i not in assigned:
                    domain.append(i)
            csp.get("domains").update({variable: domain})


def get_mrv(assignment, csp):
    mrv = 10
    mrv_var = None
    variables = csp.get("variables")
    domains = csp.get("domains")

    for variable in variables:
        if assignment.get(variable) == None:
            values = domains.get(variable)
            if len(values) < mrv:
                mrv = len(values)
                mrv_var = variable
    return mrv_var


def is_consistent(variable, assignment, neighbors):
    for neighbor in neighbors:
        if assignment.get(neighbor) == assignment.get(variable):
            return False
    return True


def forward_check(value, csp, neighbors):
    domains = csp.get("domains")
    for neighbor in neighbors:
        if value in domains.get(neighbor) and len(domains.get(neighbor)) == 1:
            return False
    for neighbor in neighbors:
        if value in domains.get(neighbor):
            domains.get(neighbor).remove(value)
    return True


def allDiff(assignment, varbs):
    values = []
    for var in varbs:
        values.append(assignment.get(var))
    is_unique = len(set(values)) == len(values)
    return is_unique


def check_constraints(assignment, csp):
    constraints = csp.get("constraints")
    for constraint in constraints:
        if allDiff(assignment, constraint) == False:
            return False
    return True


def get_neighbors(variable, csp):
    neighbors = []
    constraints = csp.get("constraints")
    for constraint in constraints:
        if variable in constraint:
            for var in constraint:
                if var != variable and var not in neighbors:
                    neighbors.append(var)

    return neighbors
