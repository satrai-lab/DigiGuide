import math

import numpy as np


class Space:
    def __init__(self, room_num, temperature, setpoint, capacity):

        self.room_num = room_num
        self.temperature = temperature
        self.setpoint = setpoint
        self.capacity = capacity
        if self.capacity == 0:
            self.capacity = 1
        self.initial_noise = np.random.normal(40, 10)
        self.crowd = 0
        self.noise = 0
        self.people_num = 0
        self.type = None
        self.sound_pressure = self.initial_noise

    def _count_people(self, occupant_location):
        occupant_count = 0
        for loc in occupant_location.values():
            if loc == self.room_num:
                occupant_count += 1
        self.people_num = occupant_count

    def update_crowd_level(self, occupant_location):
        self._count_people(occupant_location)
        crowd_ratio = (self.people_num / self.capacity)

        self.crowd = np.digitize(crowd_ratio, [0.25, 0.5, 0.75]) + 1
        return self.crowd

    def update_noise_level(self, occupant_location, occupant_noise_level):
        """基于声学公式的噪音等级计算"""
        # 将基础噪音转换为声强
        total_intensity = 10 ** (self.initial_noise / 10)

        # 叠加每个人的噪音贡献
        for p_id, loc in occupant_location.items():
            if loc == self.room_num:
                total_intensity += 10 ** (occupant_noise_level[p_id] / 10)

        # 叠加每个人的噪音贡献
        # for person in range(self.people_num):
        #     # 每个人的噪音贡献需保证非负
        #     individual_noise = max(np.random.normal(30, 15), 0)
        #     total_intensity += 10 ** (individual_noise / 10)

        # 转换回分贝（避免除以零）
        total_db = 10 * np.log10(total_intensity) if total_intensity > 0 else -np.inf

        # 分级标准（可自定义）
        self.sound_pressure = total_db
        self.noise = np.digitize(total_db, [40, 55, 60]) + 1

        return self.noise

    def calculate_next_noise_level(self, num_people):
        """基于声学公式的噪音等级计算"""
        # 将基础噪音转换为声强
        total_intensity = 10 ** (self.sound_pressure / 10)

        # 叠加每个人的噪音贡献
        for person in range(num_people):
            # 每个人的噪音贡献需保证非负
            individual_noise = max(np.random.normal(30, 0), 0)
            total_intensity += 10 ** (individual_noise / 10)

        # 转换回分贝（避免除以零）
        total_db = 10 * np.log10(total_intensity) if total_intensity > 0 else -np.inf

        # 分级标准（可自定义）
        return np.digitize(total_db, [40, 55, 60]) + 1

# class Space:
#     def __init__(self):
#         self.id = ''
#         self.setpoint = -1
#         self.temperature = -1
#         self.cap = -1
#         self.luminosity = -1
#
#     def getID(self):
#         return self.id
#
#     def setID(self, id):
#         self.id = id
#
#     def setCap(self,cap):
#         self.cap = cap
#
#     def getCap(self):
#         return self.cap
#
#     def setSetpoint(self, setpoint):
#         self.setpoint = setpoint
#
#     def getSetpoint(self):
#         return self.setpoint
#
#     def setTemperature(self, temp):
#         self.temperature = temp
#
#     def getTemperature(self):
#         return self.temperature
#
#     def setLuminosity(self, luminosity):
#         self.luminosity = luminosity
#
#     def getLuminosity(self):
#         return self.luminosity
