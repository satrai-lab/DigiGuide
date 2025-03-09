import numpy as np

from H2H_guidance.solver import get_position
from shortest_path import Graph


def eval_temp(solution, p_location, pid_need_guided, p_preferences, spaces, people_want_move):
    vote_sheet = {
        0: {
            16: -3, 17: -3, 18: -3, 19: -3, 20: -2, 21: -2, 22: -1, 23: -1, 24: 0, 25: 0, 26: 0, 27: 1, 28: 1,
            29: 2, 30: 2
        },
        1: {
            16: -3, 17: -3, 18: -2, 19: -2, 20: -1, 21: -1, 22: 0, 23: 0, 24: 0, 25: 0, 26: 1, 27: 1, 28: 2,
            29: 2, 30: 3
        },
        2: {
            16: -3, 17: -2, 18: -2, 19: -1, 20: -1, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 1, 27: 2, 28: 3,
            29: 3, 30: 3
        }
    }

    space_name_options = sorted(list(spaces.keys()))
    index_to_space = {i: name for i, name in enumerate(space_name_options)}

    temp_diff = 0
    for i in range(len(solution)):
        indoor_temp = round(spaces[index_to_space[solution[i]]].temperature)
        if indoor_temp < 16 or indoor_temp > 30:
            return 3
        temp_diff += abs(vote_sheet[p_preferences[i]["thermal"]][indoor_temp])
    return temp_diff


def eval_energy(solution, p_location, pid_need_guided, p_preferences, spaces, people_want_move):
    temperature_preference = {0: 21, 1: 23, 2: 25}
    diff = 0

    space_name_options = sorted(list(spaces.keys()))
    index_to_space = {i: name for i, name in enumerate(space_name_options)}

    max_diff_each_location = {}

    for i in range(len(solution)):
        if solution[i] not in max_diff_each_location:

            max_diff_each_location[solution[i]] = spaces[index_to_space[solution[i]]].setpoint - temperature_preference[
                p_preferences[i]["thermal"]]
        else:
            max_diff_each_location[solution[i]] = max(max_diff_each_location[solution[i]],
                                                      spaces[index_to_space[solution[i]]].setpoint -
                                                      temperature_preference[p_preferences[i]["thermal"]])
        if spaces[index_to_space[solution[i]]].setpoint == 30:
            # penalty to locate to an empty space
            max_diff_each_location[solution[i]] += 1

    for v in max_diff_each_location.values():
        diff += v
    return diff


def eval_crowd(solution, p_location, pid_need_guided, p_preferences, spaces, people_want_move):
    crowd_diff = 0

    space_name_options = sorted(list(spaces.keys()))
    index_to_space = {i: name for i, name in enumerate(space_name_options)}

    occupation = {}
    for s in solution:
        if s not in occupation:
            occupation[s] = 1
        else:
            occupation[s] += 1

    next_crowd = {}
    for i in range(len(solution)):
        next_crowd[i] = (spaces[index_to_space[solution[0]]].people_num + occupation[solution[i]]) / spaces[
            index_to_space[solution[0]]].capacity
        # # Penalty for over-crowded
        # if next_crowd > 1:
        #     crowd_diff += 1000 * (next_crowd - 1)

    if len(solution) == len(p_preferences):
        for i in range(len(solution)):
            if next_crowd[i] - p_preferences[i]["crowd"] > 0:
                crowd_diff += next_crowd[i] - p_preferences[i]["crowd"]
            if next_crowd[i] > 1:
                crowd_diff += 1000 * (next_crowd[i] - 1)

    else:
        for pref in p_preferences:
            if next_crowd[0] - pref["crowd"] > 0:
                crowd_diff += next_crowd[0] - pref["crowd"]

    # # diff = spaces[index_to_space[solution[i]]].crowd - p_preferences[i]["crowd"]
    #     if diff > 0:
    #         crowd_diff += diff
    if crowd_diff > 0.6:
        crowd_diff *= 2
    return crowd_diff


def eval_noise(solution, p_location, pid_need_guided, p_preferences, spaces, people_want_move):
    noise_diff = 0

    space_name_options = sorted(list(spaces.keys()))
    index_to_space = {i: name for i, name in enumerate(space_name_options)}

    occupation = {}
    for s in solution:
        if s not in occupation:
            occupation[s] = 1
        else:
            occupation[s] += 1

    next_noise = {}
    for i in range(len(solution)):
        next_noise[i] = spaces[index_to_space[solution[i]]].calculate_next_noise_level(occupation[solution[i]])

    if len(solution) == len(p_preferences):
        for i in range(len(solution)):
            diff = next_noise[i] - p_preferences[i]["noise"]
            if diff > 0:
                noise_diff += diff
    else:
        for pre in p_preferences:
            diff = next_noise[0] - pre["noise"]
            if diff > 0:
                noise_diff += diff

    if noise_diff > 0.8:
        noise_diff *= 2

    return noise_diff


def eval_distance(solution, p_location, pid_need_guided, p_preferences, spaces, people_want_move):
    graph = Graph("SmartSPEC/Updated model/Spaces.json")

    total_distance = 0

    space_name_options = sorted(list(spaces.keys()))
    index_to_space = {i: name for i, name in enumerate(space_name_options)}

    for i in range(len(solution)):
        total_distance += graph.shortest_path(
            p_location[pid_need_guided[i]] if p_location[pid_need_guided[i]] > 0 else 0,
            spaces[index_to_space[solution[i]]].room_num)

    return total_distance


def eval_re_guide_times(solution, p_location, pid_need_guided, p_preferences, spaces, people_want_move):
    total_re_guide_times = 0
    space_name_options = sorted(list(spaces.keys()))
    index_to_space = {i: name for i, name in enumerate(space_name_options)}
    for i in range(len(solution)):
        # this person is not willing to move
        if pid_need_guided[i] not in people_want_move:
            if p_location[pid_need_guided[i]] != spaces[index_to_space[solution[i]]].room_num:
                total_re_guide_times += 1
    return total_re_guide_times


def guidance_generator(p_location, pid_need_guided, p_preferences, spaces, result_length, people_wanna_move):
    objectives = {
        "temperature_difference": eval_temp,
        "energy_consumption": eval_energy,
        "crowd_difference": eval_crowd,
        "noise_difference": eval_noise,
        "move_distance": eval_distance,
        "re_guide_times": eval_re_guide_times
    }
    if len(p_preferences) == 0:
        return {}
    algorithm = "NSGA2"
    n_gens = 100
    pop_size = 200
    result = get_position(p_location, pid_need_guided, p_preferences, spaces, objectives, algorithm, n_gens=n_gens,
                          pop_size=pop_size, result_length=result_length, people_wanna_move=[])
    # if algorithm == "NSGA2":
    #     weights = np.array([1, 1, 1, 1, 0.2])
    #     result = choose_best_solution_from_nsgaii_output(result, weights)
    # weights = np.array([1, 1, 1, 1, 0])
    return choose_best_solution_from_nsgaii_output(result)


def choose_best_solution_from_nsgaii_output(nsgaii_results):
    """
    This function takes the raw NSGA-II output, normalizes the objective values,
    computes the weighted sum for each solution, and selects the best solution
    based on the lowest weighted sum.

    Args:
        nsgaii_results (dict): A dictionary containing NSGA-II output.
            It is expected to have a key 'F' that holds a numpy array of shape
            (n_solutions, n_objectives) with the raw objective values.
        weights (array-like): A list or numpy array of weights for each objective.
            It is recommended that the weights are non-negative and sum to 1.

    Returns:
        dict，包含 5 个最佳解及其对应的目标值，格式如下：
            {
                "best_solutions": <numpy 数组, shape=(5, n_var)>,
                "best_function_values": <numpy 数组, shape=(5, n_obj)>
            }
    """

    # first_two_solutions = nsgaii_results["first_two_front_solutions"]
    # first_two_objectives = nsgaii_results["first_two_front_function_values"]
    first_two_solutions = nsgaii_results["best_solution"]
    first_two_objectives = nsgaii_results["function_value"]

    # 将目标函数值归一化（min-max归一化），按列归一化
    normalized_F = np.zeros_like(first_two_objectives)
    for j in range(first_two_objectives.shape[1]):
        objective_values = first_two_objectives[:, j]
        min_val = np.min(objective_values)
        max_val = np.max(objective_values)

        # 如果所有值都相等，则直接设置为0避免除零错误
        if max_val - min_val == 0:
            normalized_F[:, j] = 0
        else:
            normalized_F[:, j] = (objective_values - min_val) / (max_val - min_val)

    # 计算每个解的加权和；加权和越低表示整体性能越好（适用于最小化问题）
    weights = np.array([1, 1, 1, 1, 1, 10])
    weighted_sums = np.dot(normalized_F, weights)

    # 获取加权和最低的前 1 个解的索引
    best_indices = np.argsort(weighted_sums)[:1]

    best_solutions = first_two_solutions[best_indices]
    best_function_values = first_two_objectives[best_indices]

    # print(nsgaii_results)

    return {
        "best_solutions_to_user": best_solutions,
        "best_function_values_to_user": best_function_values
    }


def evaluate_person_objectives(solution, person_index, p_location, pid_need_guided, p_preference, spaces):
    """
    针对一个候选解中的单个个体进行目标评价。

    参数:
        solution: 1D array，表示一个候选解，长度等于参与推荐的人数；
        person_index: int，表示当前要评价的人的索引；
        p_location, pid_need_guided, p_preference, spaces: 原来传给目标函数的其它参数；
        objectives: dict，每个目标名称对应一个评价函数，
                    每个评价函数要求能够接受单个推荐值进行评价。

    返回:
        np.array，包含当前目标下该人的各评价值，顺序与 objectives 字典中相同。
    """

    objectives = {
        "temperature_difference": eval_temp,
        "energy_consumption": eval_energy,
        "crowd_difference": eval_crowd,
        "noise_difference": eval_noise,
        "move_distance": eval_distance
    }

    person_value_vector = []
    # 假设 candidate 中每个元素都是某个人的推荐（比如一个位置编号）
    person_recommendation = solution

    for obj_name, eval_func in objectives.items():
        # 这里假设 eval_func 能够针对单个推荐计算评价值，
        # 例如：eval_temp(person_recommendation, p_location, pid_need_guided, p_preference, spaces)
        value = eval_func(person_recommendation, p_location, pid_need_guided, p_preference, spaces)
        person_value_vector.append(value)

    return np.array(person_value_vector)
