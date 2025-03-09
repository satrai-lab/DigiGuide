import pandas as pd
from datetime import datetime
import pytz
import numpy as np
import os

from shortest_path import Graph


def _get_itc_increment(votes: dict, atc):
    itc = 0
    for p_id, v in votes.items():
        itc += abs(v - atc)
    return itc


class Result:
    crowd_zero_time = 0
    noise_zero_time = 0
    his_ec = {}
    pointer_ec = 0
    his_itc = {}
    pointer_itc = 0
    his_tce = {}
    pointer_tce = 0
    total_distance = 0
    moved_times = 0
    graph = Graph("SmartSPEC/Updated model/Spaces.json")

    carbon_file_mapping = {"mumbai": "models/carbon_intensity/IN-WE_2023_hourly.csv",
                           "la": "models/carbon_intensity/US-CAL-CISO_2023_hourly.csv",
                           "paris": "models/carbon_intensity/FR_2023_hourly.csv",
                           "scranton": "models/carbon_intensity/US-MIDA-PJM_2023_hourly.csv"}

    time_zone = {"mumbai": "Asia/Kolkata",
                 "la": "America/Los_Angeles",
                 "paris": "Europe/Paris",
                 "scranton": "America/New_York"}

    def __init__(self, occupants, city, spaces, comment):
        # TODO user define region from input
        self.ec_clg = 0
        self.ec_htg = 0
        self.ec_fan = 0
        self.itc = 0
        self.tce = 0
        self.carbon_emission = 0
        self.occupants = occupants

        self.local_time_zone = pytz.timezone(self.time_zone[city])
        self.path_intensity_file = self.carbon_file_mapping[city]

        self.carbon_intensity = pd.read_csv(self.path_intensity_file)
        self.carbon_intensity['Datetime (UTC)'] = pd.to_datetime(self.carbon_intensity['Datetime (UTC)'])

        # Other metrics for H2H
        self.other_metrics = {"crowd": 0, "noise": 0, "thermal": 0}

        self.spaces = spaces

        # A comment that will add at the end of the output file
        self.comment = comment

        self.rearrange_time = 0
        self.guide_time = 0

    def add_consumption(self, consumption_clg, consumption_htg, consumption_fan, demand):
        if demand == "clg":
            self.ec_clg += consumption_clg
        else:
            self.ec_htg += consumption_htg
        self.ec_fan += consumption_fan

    def update_itc(self, votes: dict, atc: dict):
        for zone_id, votes_space in votes.items():

            # Update Thermal Comfort that defined in H2H
            for v in votes_space.values():
                self.other_metrics["thermal"] += abs(v)
            # Update ITC
            if atc[zone_id] == 4 or len(votes_space) == 0:
                continue
            self.itc += _get_itc_increment(votes_space, atc[zone_id])

    def update_co2_emission(self, energy_consumption, date):
        local_dt = self.local_time_zone.localize(date)
        utc_dt = local_dt.astimezone(pytz.utc)
        utc_dt = np.datetime64(utc_dt)
        row = self.carbon_intensity.loc[self.carbon_intensity['Datetime (UTC)'] == utc_dt]
        if not row.empty:
            self.carbon_emission += row['Carbon Intensity gCO₂eq/kWh (direct)'].values[0] * energy_consumption
        else:
            data_sorted = self.carbon_intensity.sort_values(by='Datetime (UTC)')
            # find the nearest one before
            nearest_before = data_sorted[data_sorted['Datetime (UTC)'] <= utc_dt]
            nearest_before = nearest_before.iloc[-1] if not nearest_before.empty else None

            # find the nearest one after
            nearest_after = data_sorted[data_sorted['Datetime (UTC)'] > utc_dt]
            nearest_after = nearest_after.iloc[0] if not nearest_after.empty else None

            if nearest_before is not None and nearest_after is not None:

                avg_intensity = (nearest_before['Carbon Intensity gCO₂eq/kWh (direct)'] + nearest_after[
                    'Carbon Intensity gCO₂eq/kWh (direct)']) / 2
                self.carbon_emission += avg_intensity * energy_consumption
            elif nearest_before is not None:

                self.carbon_emission += nearest_before['Carbon Intensity gCO₂eq/kWh (direct)'] * energy_consumption
            elif nearest_after is not None:

                self.carbon_emission += nearest_after['Carbon Intensity gCO₂eq/kWh (direct)'] * energy_consumption
            else:
                raise Exception("Cannot find carbon intensity data")

    def update_other_metrics(self):
        environment = {}
        for space_name, space in self.spaces.items():
            # occupant_count = 0
            # for loc in self.occupants.location.values():
            #     if loc > 0 and loc == int(space.room_num):
            #         occupant_count += 1
            crowd_level = space.crowd
            noise_level = space.noise
            env_space = {"crowd": crowd_level, "noise": noise_level}
            environment[space.room_num] = env_space

        for occ_id, loc in self.occupants.location.items():
            if loc != -1 and loc < 300:
                if loc == 101 or loc == 204:
                    continue
                # set weights for each metrics based on event type
                # if the value (event id) is more than 0 (in a meeting)
                weight = [1, 1, 1, 1, 0]
                if self.occupants.is_meeting[occ_id] > 0:
                    weight = [1, 1, 0, 0, 1]
                diff_crowd = environment[loc]["crowd"] - self.occupants.preference[occ_id]["crowd"]
                self.other_metrics["crowd"] += (diff_crowd * weight[2]) if diff_crowd > 0 else 0
                if weight[2] == 0:
                    self.crowd_zero_time += 1
                diff_noise = environment[loc]["noise"] - self.occupants.preference[occ_id]["noise"]
                self.other_metrics["noise"] += (diff_noise * weight[3]) if diff_noise > 0 else 0
                if weight[3] == 0:
                    self.noise_zero_time += 1

    # def _calculate_crowd_level(self, room_id, occupant_count, space):
    #
    #     ratio = (occupant_count / space.capacity) * 100
    #
    #     return np.digitize(ratio, [0.25, 0.5, 0.75]) + 1
    #
    # def _calculate_noise_level(self, room_id, occupant_count, space):
    #     """基于声学公式的噪音等级计算"""
    #     # 将基础噪音转换为声强
    #     total_intensity = 10 ** (space.initial_noise / 10)
    #
    #     # 叠加每个人的噪音贡献
    #     for person in range(occupant_count):
    #         # 每个人的噪音贡献需保证非负
    #         individual_noise = max(np.random.normal(40, 10), 0)
    #         total_intensity += 10 ** (individual_noise / 10)
    #
    #     # 转换回分贝（避免除以零）
    #     total_db = 10 * np.log10(total_intensity) if total_intensity > 0 else -np.inf
    #
    #     # 分级标准（可自定义）
    #     return np.digitize(total_db, [30, 50, 70]) + 1

    def update_distance(self, previous_location: dict, next_location: dict):
        for p_id, loc in next_location.items():
            if 0 < loc < 300:
                # not the same means the person has been moved
                if previous_location[p_id] != loc:
                    pre_loc = previous_location[p_id] if previous_location[p_id] > 0 else 0
                    self.moved_times += 1
                    self.total_distance += self.graph.shortest_path(loc, pre_loc)

    def reset(self):
        self.ec_clg = 0
        self.itc = 0
        self.other_metrics = {key: 0 for key in self.other_metrics}
        self.moved_times = 0
        self.total_distance = 0

    def save_result(self, timestamp, identify, total_itc_count):
        current_time = datetime.now()
        # timestamp_str = current_time.strftime("%Y%m%d%H%M%S")
        result_path = "./results/" + f"result_{timestamp}_" + self.comment
        if not os.path.exists(result_path):
            try:
                os.mkdir(result_path)
                print(f"Result Directory '{result_path}' created successfully.")
            except OSError as error:
                print(f"Error creating directory: {error}")

        with open(result_path + "/result-" + identify + ".txt", "a") as file:
            file.write("energy cooling (kwh): \n")
            file.write(str(self.ec_clg / 3600000) + "\n")
            file.write("crowd: \n")
            file.write(str(self.other_metrics["crowd"] / (total_itc_count-self.crowd_zero_time)) + "\n")
            file.write("noise: \n")
            file.write(str(self.other_metrics["noise"] / (total_itc_count-self.noise_zero_time)) + "\n")
            file.write("avg distance: \n")
            file.write(str(self.total_distance / self.moved_times) + "\n")
            file.write("thermal discomfort: \n")
            file.write(str(self.other_metrics["thermal"]/total_itc_count) + "\n")
            file.write("rearrange rate:")
            if self.guide_time == 0:
                rearrange = 0
            else:
                rearrange = self.rearrange_time/self.guide_time
            file.write(str(rearrange) + "\n")

            file.write("\n\n\n")

            file.write("energy heating (kwh): \n")
            file.write(str(self.ec_htg / 3600000) + "\n")
            file.write("energy fan (kwh): \n")
            file.write(str(self.ec_fan / 3600000) + "\n")
            file.write("energy total (kwh): \n")
            file.write(str((self.ec_clg + self.ec_htg + self.ec_fan) / 3600000) + "\n")
            file.write("co2:\n")
            file.write(str(self.carbon_emission) + "\n")
            file.write("itc: \n")
            file.write(str(self.itc) + "\n")
            file.write("avg itc: \n")
            file.write(str(self.itc / total_itc_count) + "\n")
            file.write("tce:\n")
            for loss in self.occupants.loss.values():
                file.write(str(loss) + "\n")
            file.write("std tce: \n")
            file.write(str(np.std(list(self.occupants.loss.values()))) + "\n")
            file.write("\n")
        self.reset()
