import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
import time
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class MyProblem(Problem):
    def __init__(self, p_location, pid_need_guided, p_preference, spaces, objectives, result_length, people_wanna_move):
        self.p_location = p_location
        self.p_preference = p_preference
        self.pid_need_guided = pid_need_guided
        self.spaces = spaces
        self.objectives = objectives
        self.people_wanna_move = people_wanna_move
        # space_name_options = sorted(list(spaces.keys()))
        # index_to_space = {i: name for i, name in enumerate(space_name_options)}

        # super().__init__(n_var=len(p_preference), n_obj=len(objectives), xl=0, xu=len(spaces) - 1, vtype=int)

        super().__init__(n_var=result_length, n_obj=len(objectives), xl=0, xu=len(spaces) - 1, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        F = np.zeros((x.shape[0], len(self.objectives)))
        for i in range(x.shape[0]):
            # Iterate over each solution
            for obj_index, (obj_name, source) in enumerate(self.objectives.items()):
                if callable(source):
                    # TODO:If source is callable, use the custom method
                    value = source(x[i], self.p_location, self.pid_need_guided, self.p_preference, self.spaces, self.people_wanna_move)
                    # if obj_name == "energy_consumption":
                    #     value *= 2
                    F[i, obj_index] = value
                else:
                    raise ValueError(f"Objective '{obj_name}' must be a callable function................")
                #     # Default evaluation method
                #     F[i, obj_index] = np.sum(
                #         [np.abs(getattr(self.people[j], source) - getattr(self.spaces[x[i, j]], source)) for j in
                #          range(len(x[i]))])
        out['F'] = F


class Optimizer:
    def __init__(self, p_location, pid_need_guided, p_preference, spaces, objectives, result_length, people_wanna_move):
        self.problem = MyProblem(p_location, pid_need_guided, p_preference, spaces, objectives, result_length, people_wanna_move)

    def optimize(self, n_gen, pop_size, seed=1):
        method = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=3.0, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=0.9, eta=5.0, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        start = time.time()
        res = minimize(self.problem, method, termination=('n_gen', n_gen), seed=seed, save_history=True)
        end = time.time()

        # 使用 NonDominatedSorting 对结果进行非支配排序，并提取前两层的索引
        nds = NonDominatedSorting()
        # 参数 n_stop_if_ranked=2 意味着排序只进行到第二层
        fronts = nds.do(res.F, n_stop_if_ranked=2)
        # fronts 是一个列表，其中 fronts[0] 为第一层，fronts[1] 为第二层
        indices_first_two = np.concatenate(fronts) if len(fronts) > 0 else np.array([])

        # 提取前两层的解、目标函数值和约束违背（如果有的话）
        solutions_first_two = res.X[indices_first_two]
        function_values_first_two = res.F[indices_first_two]
        constraint_violation_first_two = res.CV[indices_first_two] if hasattr(res,
                                                                              'CV') and res.CV is not None else None

        return {
            "time_elapsed": end - start,
            "best_solution": res.X,
            "function_value": res.F,
            "constraint_violation": res.CV,
            "first_two_front_solutions": solutions_first_two,  # 前两层的解
            "first_two_front_function_values": function_values_first_two,  # 前两层对应的目标函数值
            "first_two_front_constraint_violation": constraint_violation_first_two
        }


# import random
#
# num_people = 100
# num_spaces = 6
#
# people = [Person(random.uniform(18, 28), random.uniform(100, 1000)) for _ in range(num_people)]
# spaces = [Space(random.uniform(18, 28), random.uniform(100, 1000)) for _ in range(num_spaces)]
#
# objectives = {
#     "temperature_difference": "temperature",
#     "brightness_difference": "brightness"
# }
#
# optimizer = Optimizer(people, spaces, objectives)
# result = optimizer.optimize(n_gen=1000, pop_size=20)
#
# print("Time Elapsed:", result["time_elapsed"])
# print("Best Solution:", result["best_solution"])
# print("Function Value:", result["function_value"])
