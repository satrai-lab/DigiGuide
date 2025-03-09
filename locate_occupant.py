# import os
# from datetime import datetime
# import random
#
#
# def locate_by_time(time: datetime, people_num, occupant_locations, path_trajectory):
#     old_occupant_locations = occupant_locations
#
#     # clean the old location to be -1
#     for p_id in range(people_num):
#         occupant_locations[p_id+1] = -1
#
#     f_p = open(os.path.join(path_trajectory, str(time.month) + ".txt"))
#
#     line = f_p.readline()
#     while line:
#         data = line.strip().split(";")
#         if data[1][4:] == str(time)[4:]:
#             people = data[-1].split(",")[:-1]
#             for p_id in people:
#                 if int(data[0]) > 120:
#                     occupant_locations[int(p_id)] = random.randint(1, 120)
#                 else:
#                     occupant_locations[int(p_id)] = int(data[0])
#         line = f_p.readline()
#
#     return occupant_locations
#
#
