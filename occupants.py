import os.path
import random
from datetime import datetime, timedelta

import numpy as np

from H2H_guidance.multi_objectives import guidance_generator, evaluate_person_objectives
from Trajectory_data_loader import TrajectoryDataLoader
from strategies import get_next_loss
from comfort_collector import knn_train
import pandas as pd
import copy


class Participant:
    """
    Note:   p_id starts from 1,
            space_num starts from 1
    """
    num = 0
    location = {}  # -1 means not inside
    itc_loss = {}
    loss = {}
    occ_profile = {}
    result = None
    noise = {}
    flag_just_guided = {}
    trajectory_data = {}
    p_event_type = {}

    def __init__(self, space_num, occ_config, history_data, pattern, path_trajectory, collection_strategy, spaces,
                 space_zone_map, description_map, h2h):
        """

        :param occ_config: configuration file of the occupants
        :param history_data: history data of the occupants, for training the comfort collector
        :param pattern: scenario_pattern, decides the voting sys
        :param path_trajectory: path to the trajectory folder
        """

        self.collection_strategy = collection_strategy
        self.path_trajectory = path_trajectory
        self.space_num = space_num
        self.history_data = history_data
        self.spaces = spaces
        self.space_zone_map = space_zone_map
        self.description_map = description_map
        self.id_to_space = {info["id"]: name for name, info in description_map.items()}

        self.h2h = h2h

        # all preferences of all people
        self.preference = {}

        # flag, if the person is going for a meeting. 0 refers to not meeting. others refer to meeting event id
        self.is_meeting = {}

        # isdigit (has_profile = False) means it uses prior_knowledge approach for comfort collection and
        # the digit value is the number of occupants
        if str(occ_config).isdigit():
            self.has_profile = False
            self.config_occ_profile(occ_config)
        else:
            self.has_profile = True
            self.config_occ_profile(occ_config)

        if pattern is not None:
            self.all_params = ['ta', 'activity_20', 'age', 'gender', 'weight_level', "preference"]
            self.pattern_system = int(pattern[1])
            self.pattern_mode = int(pattern.split("_")[1])
            if self.pattern_system == 2 and self.pattern_mode != 100:
                if self.pattern_mode == 75:
                    self.knn_params = self.all_params[0:4]
                elif self.pattern_mode == 50:
                    self.knn_params = self.all_params[0:3]
                elif self.pattern_mode == 25:
                    self.knn_params = self.all_params[0:2]
            else:
                self.knn_params = self.all_params[0:5]
            self.knn_model = knn_train(self.knn_params)
            self.knn_model_all_params = knn_train(self.all_params[0:5])

        else:
            self.pattern_system = 0
            self.pattern_mode = 0

        self.trajectory_loader = TrajectoryDataLoader(path_trajectory)

        self.vote_sheet = {
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
        space_name_options = sorted(list(self.spaces.keys()))
        self.index_to_space = {i: name for i, name in enumerate(space_name_options)}
        self.id_to_space = {}
        for s_name, s in spaces.items():
            self.id_to_space[s.room_num] = s_name

    def config_occ_profile(self, config_file):
        if self.has_profile:
            f_config = open(config_file, "r")
            line = f_config.readline()
            params = line.strip().split(",")
            line = f_config.readline()

            occ_id = 1
            while line:
                data = line.strip().split(",")
                person = {}
                for i in range(len(params)):
                    person[params[i]] = data[i]
                self.occ_profile[occ_id] = person

                self.itc_loss[occ_id] = 0
                self.loss[occ_id] = 0
                self.location[occ_id] = -1

                occ_id += 1
                line = f_config.readline()
            f_config.close()
            self.num = occ_id - 1
        else:
            occ_id = 1
            # now config_file is the number of people, history_data is distribution like [30, 40, 30]
            cold_preferred = self.history_data[0]
            neutral_preferred = self.history_data[1]
            warm_preferred = self.history_data[2]

            comfort_list = ["crowd", "noise"]

            for i in range(config_file):
                random_number = random.uniform(0, 100)
                if random_number < cold_preferred:
                    self.occ_profile[occ_id] = 0
                elif random_number < cold_preferred + neutral_preferred:
                    self.occ_profile[occ_id] = 1
                else:
                    self.occ_profile[occ_id] = 2
                self.itc_loss[occ_id] = 0
                self.loss[occ_id] = 0
                self.location[occ_id] = -1

                other_preference = self._configure_other_objs(comfort_list)
                self.is_meeting[occ_id] = 0
                self.preference[occ_id] = {**other_preference, "thermal": self.occ_profile[occ_id]}

                # Generate noise in Decibel generated from this person randomly.
                individual_noise = np.clip(np.random.normal(40, 10), 30, 80)
                self.noise[occ_id] = individual_noise
                if individual_noise > 60 and self.preference[occ_id]["noise"] <= 2:
                    self.preference[occ_id]["noise"] += 2

                occ_id += 1
            self.num = occ_id - 1

    def _configure_other_objs(self, comfort_list):
        """
        This function is created for the H2H project for modeling multiple objectives

        :return:
        """

        # comfort_list = {"crowd": None, "noise": None}
        comfort_dict = {}
        # 拥挤舒适度正态分布参数
        crowd_mean = 50  # 拥挤舒适度需求的均值（0-100%，50为适中）
        crowd_std = 15  # 拥挤舒适度需求的标准差

        # 噪音舒适度正态分布参数
        noise_mean = 50  # 噪音舒适度需求的均值（0-100分贝，45为适中）
        noise_std = 10  # 噪音舒适度需求的标准差

        # 随机生成拥挤舒适度需求
        crowd_comfort = np.random.normal(crowd_mean, crowd_std)
        crowd_comfort = np.clip(crowd_comfort, 0, 100)  # 限制范围为 0-100
        crowd_level = np.digitize(crowd_comfort, [25, 50, 75]) + 1  # 1-4级
        comfort_dict["crowd"] = crowd_level

        # 随机生成噪音舒适度需求
        noise_comfort = np.random.normal(noise_mean, noise_std)
        noise_comfort = np.clip(noise_comfort, 0, 100)  # 限制范围为 0-100
        noise_level = np.digitize(noise_comfort, [25, 50, 75]) + 1  # 1-4级
        comfort_dict["noise"] = noise_level

        return comfort_dict

    def clean(self):
        """
        Clean participants' losses and locations.
        Used before changing to another algo while remaining participants' configuration.
        """
        for person_id in self.loss.keys():
            self.loss[person_id] = 0
            self.location[person_id] = -1

    def _locate(self, time: datetime):
        """
        Update people's current location
        :param time: datetime, simulation time
        """

        previous_location = copy.deepcopy(self.location)
        next_location = copy.deepcopy(self.location)

        # clean the old location to be -1
        for p_id in range(self.num):
            next_location[p_id + 1] = -1
            # self.p_event_type[p_id+1] = None

        # file_path = os.path.join(self.path_trajectory, time.strftime("%m"), time.strftime("%d") + ".csv")
        #
        # if not os.path.exists(file_path):
        #     return
        # if time.hour <= 7 or time.hour >= 21:
        #     return
        # df = pd.read_csv(file_path, parse_dates=["StartDateTime", "EndDateTime"])
        #
        # # **忽略年份，仅保留 "MM-DD HH:MM:SS" 格式**
        # df["TimeWithoutYear"] = df["StartDateTime"].dt.strftime("%m-%d %H:%M:%S")
        # query_time_str = time.strftime("%m-%d %H:%M:%S")
        # query_time_end_str = (time + timedelta(minutes=30)).strftime("%m-%d %H:%M:%S")
        #
        # mask = (df["TimeWithoutYear"] >= query_time_str) & (df["TimeWithoutYear"] <= query_time_end_str)
        # filtered_df = df.loc[mask]
        #
        # # **只保留每个 PersonID 最后出现的记录**
        # latest_records = filtered_df.sort_values(by="StartDateTime").groupby("PersonID").last().reset_index()

        latest_records = self.trajectory_loader.get_filtered_data(time)
        if not latest_records.empty:

            pid_need_guided_grouped = {}
            p_pref_need_guided_grouped = {}
            for index, row in latest_records.iterrows():
                if 140 <= row.EventID < 144:
                    self.is_meeting[int(row.PersonID)] = row.EventID
                    if row.EventID in pid_need_guided_grouped.keys():
                        pid_need_guided_grouped[row.EventID].append(int(row.PersonID))
                        p_pref_need_guided_grouped[row.EventID].append(self.preference[row.PersonID])
                    else:
                        pid_need_guided_grouped[row.EventID] = [int(row.PersonID)]
                        p_pref_need_guided_grouped[row.EventID] = [self.preference[row.PersonID]]
                else:
                    self.is_meeting[int(row.PersonID)] = 0

                if row.EventID > 0 and row.SpaceID > 0:
                    next_location[int(row.PersonID)] = row.SpaceID

            # The following code is for running H2H to locate people
            if self.h2h:

                # get the individuals that needs to be guided
                pid_need_guided = []
                p_pref_need_guided = []

                location_need_guided = {}
                for p_id in range(1, self.num + 1):
                    # This person needs a new place
                    if previous_location[p_id] == -1 and next_location[p_id] != -1:
                        pid_need_guided.append(p_id)
                        p_pref_need_guided.append(self.preference[p_id])

                    # this person is leaving
                    if previous_location[p_id] != -1 and next_location[p_id] == -1:
                        next_location[p_id] = -1

                # H2H guiding pid_need_guided to proper spaces
                # # guide individuals (regular work)
                #
                # solutions = guidance_generator(self.location, pid_need_guided, p_pref_need_guided, self.spaces,
                #                                len(pid_need_guided))
                # if len(solutions) != 0:
                #
                #     chosen_solution = solutions
                #
                #     for idx, pid in enumerate(pid_need_guided):
                #         next_location[pid] = self.spaces[
                #             self.index_to_space[chosen_solution["best_solutions_to_user"][0][idx]]].room_num
                #         self.result.guide_time += 1

                # for i in range(len(chosen_solution[""][0])):
                #     next_location[pid_need_guided[i]] = self.spaces[index_to_space[i]].room_num

                # # Guide one by one
                # for i in range(len(pid_need_guided)):
                #     guidance = guidance_generator(self.location, [pid_need_guided[i]], [p_pref_need_guided[i]], self.spaces)
                #     next_location[pid_need_guided[i]] = self.spaces[index_to_space[guidance[0]]].room_num


                # guide groups (meetings)
                for event_id, group in pid_need_guided_grouped.items():
                    solutions = guidance_generator(self.location, pid_need_guided_grouped[event_id],
                                                   p_pref_need_guided_grouped[event_id], self.spaces, 1, pid_need_guided_grouped[event_id])
                    if len(solutions) != 0:
                        # chosen_solution = event_work.choose_best_solution_per_person(solutions, self.location, group, p_pref_need_guided_grouped[event_id], self.spaces)
                        chosen_solution = solutions
                        for idx, pid in enumerate(group):
                            next_location[pid] = self.spaces[
                                self.index_to_space[chosen_solution["best_solutions_to_user"][0][0]]].room_num
                            self.result.guide_time += 1

                all_work_people_id = []
                all_work_people_pref = []
                for p_id, loc in next_location.items():
                    # if person is working in the office room, and their location is not -1 (outside)
                    if self.is_meeting[p_id] == 0 and loc != -1:
                        all_work_people_id.append(p_id)
                        all_work_people_pref.append(self.preference[p_id])
                solutions = guidance_generator(self.location, all_work_people_id,
                                               all_work_people_pref, self.spaces, len(all_work_people_id), pid_need_guided)
                if len(solutions) != 0:
                    for idx, pid in enumerate(all_work_people_id):
                        if next_location[pid] != self.spaces[
                            self.index_to_space[solutions["best_solutions_to_user"][0][idx]]].room_num:
                            next_location[pid] = self.spaces[
                                self.index_to_space[solutions["best_solutions_to_user"][0][idx]]].room_num
                            self.result.rearrange_time += 1
                            self.result.guide_time += 1


                # guidance = guidance_generator(self.location, pid_need_guided, p_pref_need_guided, self.spaces)
                # for i in range(len(guidance)):
                #     next_location[pid_need_guided[i]] = self.spaces[index_to_space[i]].room_num

                # Previous is for H2H

                # # relocate people that feel uncomfortable
                # # find people that feel uncomfortable
                # pid_discomfort = []
                # p_pref_discomfort = []
                # for p_id, loc in next_location.items():
                #     if loc == -1:
                #         continue
                #     if loc not in self.id_to_space:
                #         continue
                #     space_name = self.id_to_space[loc]
                #
                #     # thermal discomfort:
                #     if self.spaces[space_name].temperature < 16 or self.spaces[space_name].temperature > 30:
                #         thermal_discomfort = 3
                #     else:
                #         thermal_discomfort = abs(
                #             self.vote_sheet[self.preference[p_id]["thermal"]][round(self.spaces[space_name].temperature)])
                #
                #     # crowdedness discomfort:
                #     crowd_discomfort = self.spaces[space_name].crowd - self.preference[p_id]["crowd"]
                #     if crowd_discomfort < 0:
                #         crowd_discomfort = 0
                #
                #     # noise discomfort:
                #     noise_discomfort = self.spaces[space_name].noise - self.preference[p_id]["noise"]
                #     if noise_discomfort < 0:
                #         noise_discomfort = 0
                #
                #     # magic number...
                #     if thermal_discomfort + crowd_discomfort + noise_discomfort > 3:
                #         if self.is_meeting[p_id] == 0:
                #             pid_discomfort.append(p_id)
                #             p_pref_discomfort.append(self.preference[p_id])
                #
                # # guide these people
                # solutions = guidance_generator(self.location, pid_discomfort, p_pref_discomfort, self.spaces,
                #                                len(pid_discomfort))
                # if len(solutions) != 0:
                #     for idx, pid in enumerate(pid_discomfort):
                #         if next_location[pid] != self.spaces[
                #             self.index_to_space[solutions["best_solutions_to_user"][0][idx]]].room_num:
                #             next_location[pid] = self.spaces[
                #                 self.index_to_space[solutions["best_solutions_to_user"][0][idx]]].room_num
                #             self.result.rearrange_time += 1
                #             self.result.guide_time += 1

        self.location = copy.deepcopy(next_location)

        for s in self.spaces.values():
            s.update_crowd_level(self.location)
            s.update_noise_level(self.location, self.noise)
        self.result.update_distance(previous_location, next_location)

    def vote(self, time: datetime, temp: dict):
        """
        Votes generation by the temperature
        :param time: datetime, used to locate people
        :param temp: dictionary, indoor temperature of each space
        :return: dictionary, a dictionary of people's votes
        """

        self._locate(time)
        random_activity = random.randint(1, 2)
        # ["historical_based", "prior_knowledge"]

        # real_votes = {}
        # for s in range(1, self.space_num + 1):
        #     real_votes[s] = {}
        # for index in range(1, self.num + 1):
        #     if self.location[index] > -1:
        #         knn_data = {}
        #         for p in self.all_params[0:5]:
        #             if p == "activity_20":
        #                 knn_data[p] = [random_activity]
        #             elif p == "ta":
        #                 knn_data[p] = [temp[self.location[index]]]
        #             else:
        #                 knn_data[p] = [self.occ_profile[index][p]]
        #
        #         knn_data = pd.DataFrame(knn_data)  # Replace with your own data
        #
        #         predicted_comfort = self.knn_model_all_params.predict(knn_data)
        #
        #         real_votes[self.location[index]][index] = predicted_comfort[0].round()

        if self.collection_strategy == "historical_based":
            votes = {}
            for s in range(1, self.space_num + 1):
                votes[s] = {}
            # collect votes based on indoor temperature where people locate
            for index in range(1, self.num + 1):
                if self.location[index] > -1:
                    if self.pattern_system == 1:
                        random_number = random.randint(1, 100)
                        if random_number > self.pattern_mode:
                            continue

                    knn_data = {}
                    for p in self.knn_params:
                        if p == "activity_20":
                            knn_data[p] = [random.randint(1, 2)]
                        elif p == "ta":
                            knn_data[p] = [temp[self.location[index]]]
                        else:
                            knn_data[p] = [self.occ_profile[index][p]]

                    knn_data = pd.DataFrame(knn_data)  # Replace with your own data

                    predicted_comfort = self.knn_model.predict(knn_data)

                    votes[self.location[index]][index] = predicted_comfort[0].round()

                    if votes[self.location[index]][index] < -3:
                        votes[self.location[index]][index] = -3
                    if votes[self.location[index]][index] > 3:
                        votes[self.location[index]][index] = 3

        elif self.collection_strategy == "prior_knowledge":
            votes = {}
            for s in range(1, self.space_num + 1):
                votes[s] = {}

            for index in range(1, self.num + 1):
                # if person index is in the building
                if self.location[index] > 0 and self.location[index] in self.id_to_space:
                    zone_location = self.space_zone_map[self.id_to_space[self.location[index]]].lstrip("Thermal Zone")
                    if "Corridor" not in zone_location:
                        temperature = round(temp[int(zone_location)])
                        if temperature <= 16:
                            comfort = -3
                        elif temperature > 30:
                            comfort = 3
                        else:
                            comfort = self.vote_sheet[self.occ_profile[index]][temperature]
                        votes[int(zone_location)][index] = comfort
        else:
            raise Exception("no thermal comfort collection??")
        return votes, votes

    def update_loss(self, votes: dict, atc: dict):
        """

        :param votes: dict, votes of everyone
        :param atc: dict, atc of every space
        :return:
        """
        new_loss = {}
        for room_id, value in atc.items():
            if value == 4:
                continue
            next_loss = get_next_loss(votes[room_id], self.loss, atc[room_id])
            for p_id, loss in next_loss.items():
                if p_id not in new_loss.items():
                    new_loss[p_id] = loss

        for p_id, loss in new_loss.items():
            self.loss[p_id] = loss

        for room_id in votes.keys():
            for p_id, vote in votes[room_id].items():
                self.itc_loss[p_id] += abs(vote - atc[room_id])

# class event_weight:
#     """
#     weights for the weighted sum algorithm. The weights should be self-adapted based on occupant selection
#     """
#
#     def __init__(self, event_type, user_weight):
#         self.event_type = event_type
#         # weight is for generating options for users,
#         # self.weight = np.array([1, 1, 1, 1, 0])
#         self.weight = user_weight
#
#         # user_weight is the weight user uses to select one solution from the options
#         # this weight is like the ground truth, so it doesn't change
#         self.user_weight = user_weight
#
#     def choose_best_solution_per_person(self, best_solutions, p_location, pid_need_guided, p_preference, spaces):
#
#         n_candidates, n_persons = best_solutions["best_solutions_to_user"].shape
#         best_candidate_indices = []
#         best_recommendations = []
#         best_person_objectives = []
#
#         all_candidate_objs = []
#         for person_idx in range(n_persons):
#             candidate_scores = []
#             candidate_objs = []
#             for cand_idx in range(n_candidates):
#                 candidate_solution = best_solutions["best_solutions_to_user"][cand_idx]
#                 # 计算该候选解中该人的目标评价向量
#                 obj_vec = evaluate_person_objectives(candidate_solution, person_idx, p_location, pid_need_guided,
#                                                      p_preference, spaces)
#                 candidate_objs.append(obj_vec)
#                 # 加权求和（假设目标越低越好）
#                 score = np.dot(obj_vec, self.user_weight)
#                 candidate_scores.append(score)
#             candidate_scores = np.array(candidate_scores)
#             best_idx = np.argmin(candidate_scores)
#             best_candidate_indices.append(best_idx)
#             best_recommendations.append(best_solutions["best_solutions_to_user"][best_idx, person_idx])
#             best_person_objectives.append(candidate_objs[best_idx])
#             all_candidate_objs.append(candidate_objs)
#
#         # self.update_weights_based_on_user_choice(best_person_objectives, all_candidate_objs)
#
#         return {
#             "best_candidate_indices": best_candidate_indices,
#             "best_recommendations": best_recommendations,
#             "best_person_objectives": best_person_objectives
#         }
#
#     def update_weights_based_on_user_choice(self, selected_objectives, candidate_objectives, learning_rate=0.1):
#         """
#         根据用户选择的选项对权重进行更新。
#
#         如果用户选择的选项在某个 objective 上的值比候选方案的平均值更低，
#         则说明用户更在意这个 objective，因此该 objective 的权重将被提升。
#
#         参数:
#             old_weights: numpy 数组，形状 (n_objectives,)，表示原来的权重向量。
#             selected_objectives: numpy 数组，形状 (n_objectives,)，表示用户选择的选项的各个 objective 值。
#             candidate_objectives: numpy 数组，形状 (n_candidates, n_objectives)，表示所有候选方案的 objective 值。
#             learning_rate: float，学习率，控制更新步长，默认值 0.1。
#
#         返回:
#             new_weights: numpy 数组，更新后的权重向量（归一化后所有元素之和为 1）。
#         """
#         epsilon = 1e-6  # 防止除零
#
#         # 对多个 selected_objectives 求平均，得到全局的目标向量
#         selected_objectives_mean = np.mean(selected_objectives, axis=0)  # 结果形状为 (n_objectives,)
#
#         # 对候选方案的目标值取平均
#         candidate_objectives_mean = np.mean(candidate_objectives, axis=0)  # 结果形状为 (n_objectives,)
#
#         # 防止candidate_objectives中有多个数据，也就是说guide的是多个人的情况下
#         candidate_objectives_mean = np.mean(candidate_objectives_mean, axis=0)
#
#         # 计算差值：这里假设目标越小越好，如果用户选择的平均值比候选方案均值低，则说明该目标更在意
#         diff = candidate_objectives_mean - selected_objectives_mean
#
#         # 根据差值归一化调整权重
#         normalized_diff = diff / (np.abs(candidate_objectives_mean) + epsilon)
#         new_weights = self.weight * (1 + learning_rate * normalized_diff)
#
#         # 防止负权重，并归一化
#         new_weights = np.clip(new_weights, 0, None)
#         new_weights = new_weights / np.sum(new_weights)
#
#         self.weight = new_weights
#
#
# event_work = event_weight("work", np.array([1, 1, 1, 1, 0]))
# event_meeting = event_weight("meeting", np.array([1.5, 1, 0, 0, 0.5]))
