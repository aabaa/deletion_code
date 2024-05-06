import utils
import pulp
import time
import datetime
from collections import defaultdict
import gurobipy
import math

class ArrangementOptimizer:
    VT_BOUNDS    = [0,2,2,2,4,6,10,16,30,52,94,172,316]
    LOWER_BOUNDS = [0,0,0,0,2,2, 4, 6,14,22,42, 78,144]

    def __init__(self):
        self.problem = None
        self.variables = None
        self.dim = 0
        self.binary_indices = None
    
    def prepare_model(self, dim:int, binary_indices=None):
        assert dim > 0
        self.dim = dim

        self.binary_indices = binary_indices
        self.problem = pulp.LpProblem(name='maximize_deletion_code_arrangement',
                                      sense=pulp.LpMaximize)

        # Each variable has binary expression (0-1 string) index
        indices = [utils.bin2str(i, self.dim) for i in range(2**self.dim)]
        if self.binary_indices is None:
            self.variables = pulp.LpVariable.dicts(name='P', indices=indices, cat=pulp.LpBinary)
        else:
            binary_variables = pulp.LpVariable.dicts(name='P', indices=list(self.binary_indices), cat=pulp.LpBinary)
            continuous_indices = set(indices) - set(self.binary_indices)
            continuous_variables = pulp.LpVariable.dicts(name='P', indices=continuous_indices, cat=pulp.LpContinuous, lowBound=0, upBound=1)
            self.variables = binary_variables
            self.variables.update(continuous_variables)

        # Warm start solution
        # self.set_initial_values() # VT(0)
        self.set_initial_values(offset=0)
        self.fix_at_most_one_different_character()

        # Objective function
        self.problem += pulp.lpSum(self.variables.values())

        # Constraint functions
        self.add_default_constraints()                      # constraint 0
        # self.add_vt_code_bound_constraints(exist=True)      # constraint 1
        self.add_vt_code_bound_constraints(exist=False)     # constraint 1'
        # self.add_deletion_code_inclusion_constraints()      # constraint 2
        # self.add_only_one_different_character_constraints() # constraint 3-1
        # self.add_all_the_same_characters_constraints()      # constraint 3-2
        # self.add_bit_flip_symmetry_constraints()            # constraint 4
        # self.add_bit_reversal_symmetry_constraints()        # constraint 5
        self.add_number_of_runs_constraints()               # constraint 6
        # self.add_number_of_runs_constraints2()              # constraint 6'
        # self.add_inductive_n_cube_constraints()             # constraint 7

    def set_initial_values(self, offset=1):
        vt_elements = self.generate_vt_codes(self.dim, offset=offset)
        vt_elements = set(vt_elements)
        for x in range(2**self.dim):
            xs = utils.bin2str(x, self.dim)
            if x in vt_elements:
                self.variables[xs].setInitialValue(1)
            else:
                self.variables[xs].setInitialValue(0)

    def fix_at_most_one_different_character(self):
        all_0 = '0' * self.dim
        all_1 = '1' * self.dim
        self.variables[all_0].setInitialValue(1)
        self.variables[all_1].setInitialValue(1)
        self.variables[all_0].fixValue()
        self.variables[all_1].fixValue()

        for i in range(self.dim):
            one_0 = '1' * i + '0' + '1' * (self.dim - 1-i)
            one_1 = '0' * i + '1' + '0' * (self.dim - 1-i)
            self.variables[one_0].setInitialValue(0)
            self.variables[one_1].setInitialValue(0)
            self.variables[one_0].fixValue()
            self.variables[one_1].fixValue()

    # Constraint 0
    def add_default_constraints(self):
        '''
        Each deletion code belongs to at most one insertion code.
        '''
        for x in range(2**(self.dim-1)):
            originals = utils.insert_all_components(x, self.dim-1)
            self.problem += (
                pulp.lpSum([self.variables[utils.bin2str(i,self.dim)] for i in originals]) <= 1,
                f'default_constraints_D{utils.bin2str(x, self.dim-1)}'
            )

    # Constraint 1
    def add_vt_code_bound_constraints(self, exist=False):
        '''
        VT code gives the lower bound of the solutions.
        '''
        offset = 0 if exist else 1
        self.problem += (
            pulp.lpSum(self.variables.values()) >= self.VT_BOUNDS[self.dim] + offset,
            'lower_bound_given_by_VT_code'
        )

    # Constraint 2
    def add_deletion_code_inclusion_constraints(self):
        '''
        If a pair of strings which deletion code set have inclusion relationship,
        an enclosing side does not have to be arranged.
        (∃x)(del(x) c= del(y)) ---> P[y] == 0
        '''
        inclusion_pairs = self.listup_deletion_codes_inclusion_pairs()
        inclusives = {y for x,y in inclusion_pairs}
        for y in inclusives:
            y_str = utils.bin2str(y, self.dim)
            self.problem += (
                self.variables[y_str] == 0,
                f'del({y_str})_deletion_code_inclusion_constraints'
            )
    
    # Constraint 3-1
    def add_only_one_different_character_constraints(self):
        '''
        A string with only one different character (101111 or 001000) should be 0,
        because strings with all the same characters (000000 or 111111) are 1
        (∃y)(del(x) & del(y) ∧ P[y] == 1) --> (P[x] == 0)
        '''
        for i in range(self.dim):
            one_0 = '1' * i + '0' + '1' * (self.dim - 1-i)
            one_1 = '0' * i + '1' + '0' * (self.dim - 1-i)

            self.problem += (
                self.variables[one_0] == 0,
                f'P{one_0}_only_one_different_character_constraints'
                )

            self.problem += (
                self.variables[one_1] == 0,
                f'P{one_1}_only_one_different_character_constraints'
                )
    
    # Constraint 3-2
    def add_all_the_same_characters_constraints(self):
        '''
        A string with all the same character (000000 or 111111) should be 1,
        because it has only one deletion code (00000 or 11111) so that 
        it is included in any element that has intersection with it.
        (∀y)(del(x) & del(y) --> del(x) c= del(y)) --> P[x] == 1
        '''
        all_0 = '0' * self.dim
        self.problem += (
            self.variables[all_0] == 1,
            f'P{all_0}_all_the_same_characters_constraints'
            )

        all_1 = '1' * self.dim
        self.problem += (
            self.variables[all_1] == 1,
            f'P{all_1}_all_the_same_characters_constraints'
            )
        
    # Constraint 4
    def add_bit_flip_symmetry_constraints(self):
        bit_uppers = []
        bit_lowers = []
        for x in range(2**self.dim):
            s = utils.bin2str(x, self.dim)
            if s.count('1') < self.dim/2:
                bit_lowers.append(s)
            elif s.count('1') > self.dim/2:
                bit_uppers.append(s)
        self.problem += (
            pulp.lpSum([self.variables[s] for s in bit_lowers]) >= pulp.lpSum([self.variables[s] for s in bit_uppers]),
            f'bit_flip_symmetry_constraints'
        )

    # Constraint 5
    def add_bit_reversal_symmetry_constraints(self):
        group1 = []
        group2 = []
        for x in range(2**self.dim):
            s = utils.bin2str(x, self.dim)
            t = s[::-1]
            if s < t:
                group1.append(s)
            elif t < s:
                group2.append(s)
        self.problem += (
            pulp.lpSum([self.variables[s] for s in group1]) >= pulp.lpSum([self.variables[s] for s in group2]),
            f'bit_reversal_symmetry_constraints'
        )

    # Constraint 6
    def add_number_of_runs_constraints(self):
        Wn = defaultdict(list)
        for x in range(2**self.dim):
            s = utils.bin2str(x, self.dim)
            a,b = utils.count_number_of_runs(s)
            w = s.count('1')
            Wn[(w,a,b)].append(s)
        
        indices = []
        coefficients = []
        for a in range(1,4):
            for b in range(1,3):
                wn2ab = Wn[(2,a,b)]
                indices.extend(wn2ab)
                coefficients.extend([b] * len(wn2ab))
        self.problem += (
            pulp.lpDot([self.variables[s] for s in indices], coefficients) <= self.dim - 1,
            f'number_of_runs_constraints'
        )

    # Constraint 6'
    def add_number_of_runs_constraints2(self):
        Wn = defaultdict(list)
        for x in range(2**self.dim):
            s = utils.bin2str(x, self.dim)
            a,b = utils.count_number_of_runs(s)
            w = s.count('1')
            Wn[(w,a,b)].append(s)
        
        for w in range(1, self.dim):
            indices = []
            coefficients = []
            for a in range(0, self.dim+1):
                for b in range(0, self.dim+1):
                    wn2ab = Wn[(w,a,b)]
                    indices.extend(wn2ab)
                    coefficients.extend([a] * len(wn2ab))
            for a in range(0, self.dim+1):
                for b in range(0, self.dim+1):
                    wn2ab = Wn[(w+1,a,b)]
                    indices.extend(wn2ab)
                    coefficients.extend([b] * len(wn2ab))
            self.problem += (
                pulp.lpDot([self.variables[s] for s in indices], coefficients) <= math.comb(self.dim-1, w),
                f'number_of_runs_constraints_w={w}'
            )

    # Constraint 7
    def add_inductive_n_cube_constraints(self):
        # upper bounds of number of arrangement in n-cube (0-origin)
        for n in range(3, self.dim):
            for y in range(2 ** (self.dim - n)):
                ys = utils.bin2str(y, self.dim - n)
                for i in range(len(ys)):
                    left = ys[:i]
                    right = ys[i:]
                    vars = []
                    for x in range(2**n):
                        xs = utils.bin2str(x, n)
                        vars.append(self.variables[f'{left}{xs}{right}'])
                    self.problem += (
                        pulp.lpSum(vars) <= self.VT_BOUNDS[n],
                        f'{n}-dim_{ys}_{i}th_inductive_n_cube_upper_bound_constraints'
                    )

                    if n == self.dim - 1:
                        lower_bound = self.VT_BOUNDS[n] - self.VT_BOUNDS[n-1]
                        self.problem += (
                            pulp.lpSum(vars) >= lower_bound,
                            f'{n}-dim_{ys}_{i}th_inductive_n_cube_lower_bound_constraints'
                        )


    def load_model(self, path:str):
        self.variables, self.problem = pulp.LpProblem.from_json(path)

    def save_model(self, path:str):
        self.problem.to_json(path)

    def listup_deletion_codes_inclusion_pairs(self):
        inclusion_pairs = []
        for x in range(2**self.dim):
            for y in range(2**self.dim):
                if x == y:
                    continue
                x_delset = utils.delete_all_components(x, self.dim)
                y_delset = utils.delete_all_components(y, self.dim)
                if x_delset.issubset(y_delset):
                    inclusion_pairs.append((x, y))
        return inclusion_pairs
    
    def generate_vt_codes(self, dim, offset=1):
        lattice_num = 2**dim
        codes = []
        for x in range(lattice_num):
            sum = 0
            for i in range(dim):
                sum += utils.ith_component(x, i) * (i+offset)
            if sum % (dim+1) == 0:
                codes.append(x)
        return codes
    
    def calc_variable_count(self):
        var2count = defaultdict(int)
        for c in self.problem.constraints.values():
            for var in list(c.keys()):
                var2count[var.name[2:]] += 1
        return var2count
    
    def calc_sum(self):
        sum_val = 0.0
        for key in self.variables:
            sum_val += self.variables[key].value()
        return sum_val

    def solve(self):
        # solver = pulp.getSolver('PULP_CBC_CMD', threads=16, msg=True, warmStart=True)
        solver = pulp.getSolver('GUROBI', threads=16, msg=True, warmStart=True, logPath=f'gurobi_dim={self.dim}.log')
        # solver = pulp.getSolver('GUROBI', threads=16, msg=True, warmStart=True, logPath=f'gurobi_dim={self.dim}.log', timeLimit=3600)
        self.problem.solve(solver)

if __name__ == '__main__':
    # parameters
    dim = 11
    
    # executable codes
    t0 = time.time()
    print(f"Start: {datetime.datetime.now()}")
    solver = ArrangementOptimizer()
    solver.prepare_model(dim=dim)
    t1 = time.time()
    print(f"preparation time: {t1-t0}s")

    # solver.save_model(f'model_dim={dim}.json')

    solver.solve()
    t2 = time.time()
    print(f"solve time: {t2-t1}s")
    if solver.problem.status == pulp.constants.LpStatusOptimal:
        num = 0
        solutions = []
        for key in solver.variables:
            if solver.variables[key].value() >= 0.99:
                num += 1
                solutions.append(key)

        print(f'Optimal solution: num={num}')
        for s in solutions:
            print(s)

    '''
    # Solve by Gurobi
    solver.problem.writeLP('problem.lp')
    gurobi_model = gurobipy.read('problem.lp')
    gurobi_model.setParam('PoolSearchMode', 1)
    gurobi_model.setParam('PoolSolutions', 10)
    gurobi_model.optimize()

    print(f'solution num = {gurobi_model.SolCount}')
    for i in range(gurobi_model.SolCount):
        gurobi_model.setParam('SolutionNumber', i)
        print(f'{i}-th solution was found')
        # x_val = gurobi_model.getVarByName("x").Xn
        # y_val = gurobi_model.getVarByName("y").Xn
        # print(f"Solution {i+1}: x = {x_val}, y = {y_val}")
    '''

    '''
    solver = ArrangementOptimizer()
    for i in range(15):
        vt_codes = solver.generate_vt_codes(i)
        print(len(vt_codes))
    '''