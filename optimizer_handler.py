import time
import os
import pulp
from arrangement_optimizer import ArrangementOptimizer
import utils


class OptmizerHandler:
    VT_BOUNDS    = [0,2,2,2,4,6,10,16,30,52,94,172,316]
    LOWER_BOUNDS = [0,0,0,0,2,2, 4, 6,14,22,42, 78,144]

    def __init__(self, dim:int):
        assert dim > 0
        self.dim = dim
        self.trial_num = 0
        self.results = []
        self.result_path = 'result.log'

    def exec_albert_no_method(self):
        if os.path.exists(self.result_path):
            os.remove(self.result_path)
        with open(self.result_path, 'w') as f:
            f.write('trial, sum, total time, solve time\n')
        binary_indices = set()

        while False:
            solver = ArrangementOptimizer()
            vt_elements = solver.generate_vt_codes(self.dim)
            for e in vt_elements:
                binary_indices.add(utils.bin2str(e, self.dim))
            break

        var2count = {}
        while True:
            solver = ArrangementOptimizer()
            solver.prepare_model(dim=self.dim)
            var2count = solver.calc_variable_count()
            break

        while True:
            solver = self.exec_one_trial(binary_indices)
            if self.get_solver_status_message(solver) != 'Need Retry':
                break
            
            id2val = {}
            for id, lpval in solver.variables.items():
                id2val[id] = lpval.value()
            
            # Weighted Largest (Albert No method)
            sorted_id2val = sorted(id2val.items(), key = lambda item : item[1] * var2count[item[0]])

            # Weighted Middle
            # sorted_id2val = sorted(id2val.items(), key = lambda item : - abs(item[1] - 0.5) * var2count[item[0]])

            # Largest
            # sorted_id2val = sorted(id2val.items(), key = lambda item : item[1])

            # Middle
            # sorted_id2val = sorted(id2val.items(), key = lambda item : - abs(item[1] - 0.5))

            for id_val in reversed(sorted_id2val):
                print('id_val:', id_val)
                if id_val[0] not in binary_indices:
                    binary_indices.add(id_val[0])
                    break
        
        
    def exec_one_trial(self, binary_indices):
        start_time = time.time()
        self.trial_num += 1
        print(f"***Start***: trial={self.trial_num}, indices={binary_indices}")
        solver = ArrangementOptimizer()
        solver.prepare_model(dim=self.dim, binary_indices=binary_indices)

        solve_start_time = time.time()
        solver.solve()
        solve_erapsed_time = time.time() - solve_start_time

        status_message = self.get_solver_status_message(solver)

        erapsed_time = time.time() - start_time
        print(f"***{status_message}***: trial={self.trial_num}, status={solver.problem.status}, sum={solver.calc_sum()}, total time={erapsed_time}, solve time={solve_erapsed_time}")
        self.results.append({'trial': self.trial_num, 'status': solver.problem.status, 'sum': solver.calc_sum(), 'total_time': erapsed_time, 'solve_time': solve_erapsed_time})
        with open(self.result_path, 'a') as f:
            f.write(f'{self.trial_num}, {solver.calc_sum()}, {erapsed_time}, {solve_erapsed_time}\n')
        return solver

    def get_solver_status_message(self, solver):
        status_message = ""
        if solver.problem.status != pulp.constants.LpStatusOptimal:
            status_message = 'Failure'
        else:
            result_sum = solver.calc_sum()
            if result_sum < self.VT_BOUNDS[self.dim] + 2:
                status_message = 'Success'
            else:
                status_message = 'Need Retry'
        return status_message

if __name__ == '__main__':
    handler = OptmizerHandler(dim=11)
    start_time = time.time()
    handler.exec_albert_no_method()
    total_time = time.time() - start_time
    print(f'total time = {total_time}')
