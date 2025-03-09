import random

from .NSGA2_optimizer import Optimizer
import time

# from hybrid import Hybrid_Optimizer


def get_position(p_location, pid_need_guided, p_preference, spaces, objectives, algorithm, n_gens, pop_size, result_length, people_wanna_move):
    if algorithm == "NSGA2":
        start = time.time()
        optimizer = Optimizer(p_location, pid_need_guided, p_preference, spaces, objectives, result_length, people_wanna_move)

        result = optimizer.optimize(n_gens, pop_size)

        end = time.time()
        result["time_elapsed"] = end-start
        return result

    # if algorithm == "random":
    #     return ramdom_assign(people, spaces)
    #
    # if algorithm == "dis_optimal":
    #     return distance_optimal(people, spaces)
    #
    # if algorithm == "weighted_sum":
    #     return weighted_sum(people, spaces, objectives)

    # if algorithm == "hybrid":
    #     start = time.time()
    #     result = hybrid(people, spaces, objectives)
    #     end = time.time()
    #     result["time_elapsed"] = end - start
    #     return result

    else:
        exit("no such algo")


def ramdom_assign(people, spaces):
    start = time.time()
    result = []
    for p in people:
        result.append(random.randint(0, len(spaces) - 1))
    end = time.time()
    return {
        "time_elapsed": end - start,
        "best_solution": result
    }


def distance_optimal(people, spaces):
    start = time.time()
    result = []
    for p in people.values():
        min_dis = float('inf')
        solution = -1
        for s in spaces["lounge"].values():
            p1 = spaces["gate"][p.gate].location
            p2 = s.location
            distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
            if distance < min_dis:
                min_dis = distance
                solution = s.lounge_id
        result.append(solution)
    end = time.time()
    return {
        "time_elapsed": end - start,
        "best_solution": result
    }


def calculate_scores(people, spaces, objectives):

    scores = {key: [value([s.id], {0: p}, spaces) for s in spaces.values() for p in people.values()]
              for key, value in objectives.items()}

    normalized_scores = {}
    for key in scores:
        min_score = min(scores[key])
        max_score = max(scores[key])
        range_score = max_score - min_score
        # 检查分母是否为零
        if range_score == 0:
            # 如果所有分数相等，可以将归一化分数设置为0.5或其他固定值
            normalized_scores[key] = [0.5 for _ in scores[key]]  # 这里使用0.5作为所有相同分数的归一化值
        else:
            # 正常计算归一化分数
            normalized_scores[key] = [(score - min_score) / range_score for score in scores[key]]

    return normalized_scores


def weighted_sum(people, spaces, objectives):
    start = time.time()
    my_obj = dict(objectives)
    # my_obj.pop("energy_consumption")

    normalized_scores = calculate_scores(people, spaces, my_obj)

    result = []
    for i in range(len(people)):
        min_score = float('inf')
        best_space = -1
        for s in range(len(spaces)):
            score = sum(normalized_scores[key][i * len(spaces) + s] for key in my_obj.keys())
            if score < min_score:
                best_space = s
                min_score = score
        result.append(best_space)

    end = time.time()
    return {
        "time_elapsed": end - start,
        "best_solution": result
    }


# def hybrid(people, spaces, objectives):
#     start = time.time()
#     initial_result = weighted_sum(people, spaces, objectives)["best_solution"]
#
#     optimizer = Hybrid_Optimizer(people, spaces, objectives, initial_result)
#     result = optimizer.optimize(n_gen=100, pop_size=4)
#     end = time.time()
#     result["time_elapsed"] = end - start
#     return result
