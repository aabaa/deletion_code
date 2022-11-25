import utils
import pulp
import time
import datetime

class ArrangementOptimizer:
    VT_BOUNDS    = [0,2,2,2,4,6,10,16,30,52,94,172,316]
    LOWER_BOUNDS = [0,0,0,0,2,2, 4, 6,14,22,42,78,144]

    def __init__(self, dim:int):
        assert dim > 0
        self.dim = dim
        self.problem = None
        self.variables = None
    
    def prepare_model(self):
        self.problem = pulp.LpProblem(name='maximize_deletion_code_arrangement',
                                      sense=pulp.LpMaximize)

        # Each variable has binary expression (0-1 string) index
        indices = [utils.bin2str(i, self.dim) for i in range(2**self.dim)]
        self.variables = pulp.LpVariable.dicts(name='P', indices=indices, cat=pulp.LpBinary)

        # Warm start solution
        self.set_initial_values()
        self.fix_at_most_one_different_character()

        # Objective function
        self.problem += pulp.lpSum(self.variables.values())

        # Constraint functions
        self.add_default_constraints()
        self.add_vt_code_bound_constraints()
        self.add_all_the_same_characters_constraints()
        self.add_only_one_different_character_constraints()
        self.add_deletion_code_inclusion_constraints()
        self.add_inductive_n_cube_constraints()

    def set_initial_values(self):
        vt_elements = self.generate_vt_codes(self.dim)
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

    def add_vt_code_bound_constraints(self):
        '''
        VT code gives the lower bound of the solutions.
        '''
        self.problem += (
            pulp.lpSum(self.variables.values()) >= self.VT_BOUNDS[self.dim],
            'lower_bound_given_by_VT_code'
        )
    
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
    
    def generate_vt_codes(self, dim):
        lattice_num = 2**dim
        codes = []
        for x in range(lattice_num):
            sum = 0
            for i in range(dim):
                sum += utils.ith_component(x, i) * i
            if sum % (dim+1) == 0:
                codes.append(x)
        return codes

    def solve(self):
        solver = pulp.getSolver('GUROBI', threads=16, msg=True, warmStart=True, logPath=f'gurobi_dim={self.dim}.log')
        # solver = pulp.getSolver('GUROBI', threads=16, msg=True, warmStart=True, logPath=f'gurobi_dim={self.dim}.log', timeLimit=3600)
        # solver = pulp.getSolver('PULP_CBC_CMD', threads=16, msg=True, warmStart=True, logPath=f'cbc_dim={self.dim}.log')
        self.problem.solve(solver)
        if self.problem.status == pulp.constants.LpStatusOptimal:
            num = 0
            solutions = []
            for key in self.variables:
                if self.variables[key].value() >= 0.99:
                    num += 1
                    solutions.append(key)

            print(f'Optimal solution: num={num}')
            for s in solutions:
                print(s)

if __name__ == '__main__':
    # parameters
    dim = 9

    # executable codes
    start_time = time.time()
    print(f"Start: {datetime.datetime.now()}")
    solver = ArrangementOptimizer(dim=dim)
    solver.prepare_model()
    erapsed_time = time.time() - start_time
    print(f"preparation time: {erapsed_time}s")

    # solver.save_model(f'model_dim={dim}.json')
    solver.solve()

    '''
    solver = ArrangementOptimizer(dim=9)
    solver.calc_deletion_code_inclusion()
    '''