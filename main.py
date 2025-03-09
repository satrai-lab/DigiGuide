import json
import subprocess
import datetime
import argparse
from datetime import datetime, timedelta
from time import sleep

from colorama import Fore, Back, Style, init
from tqdm import tqdm

from occupants import Participant
from result_collector import Result
from sim_ep import CoSimulation
from eppy.modeleditor import IDF

from spaces import Space
from strategies import SensationAggregator, generate_set_point
import platform


def main(args):
    path_ep_model = "models/drahix/drahix.idf"
    path_trajectories = "SmartSPEC/post-processing/trajectories_split_data"
    path_profile = 160

    repeat_time = 1
    # TODO
    scenario_system = [0]
    percentage = [100]
    # distribution_scenarios = {"MW80": [20, 0, 80], "MW70": [30, 0, 70], "NM": [40, 20, 40], "MC70": [70, 0, 30], "MC80": [80, 0, 20]}
    # sensation_collector = ["historical_based", "prior_knowledge"]
    # hvac_controller = ["fixed_rule", "preference_estimation"]
    hvac_controller = ["fixed_rule"]
    # algorithms = ["majority", "fair", "drift"]
    # algorithms = ["majority"]
    algorithms = [args.algorithm]
    cities = ["paris"]
    weather_file_mapping = {"mumbai": "models/weathers/mumbai", "la": "models/weathers/la",
                            "paris": "models/weathers/paris", "scranton": "models/weathers/scranton"}

    start_date = datetime.strptime("2010-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2010-12-31", "%Y-%m-%d")

    # scenario_system = [1]
    # percentage = [100]
    # view for required processing time

    total_simulation_iteration = len(algorithms) * repeat_time * len(hvac_controller) * len(cities)
    current_simulation_iteration = 0
    start_time = datetime.now()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    for r_time in range(repeat_time):
        for city in cities:
            for algo in algorithms:
                for hvac_control_strategy in hvac_controller:
                    model_path = path_ep_model

                    epw_file_path = weather_file_mapping[city] + ".epw"

                    system = platform.system()
                    if system == "Windows":
                        idd_file = r"D:\Programming\EnergyPlus\EnergyPlusV22-2-0\Energy+.idd"
                    else:
                        idd_file = r"/home/jun/EnergyPlus-22.2.0/Energy+.idd"
                    IDF.setiddname(idd_file)
                    idf_origin = IDF(model_path)

                    ddy_file_path = weather_file_mapping[city] + ".ddy"
                    ddy_idf = IDF(ddy_file_path)

                    design_day_classes = ['SizingPeriod:DesignDay']
                    for cls in design_day_classes:
                        idf_origin.removeallidfobjects(cls)

                    design_days = ddy_idf.idfobjects["SizingPeriod:DesignDay".upper()]
                    design_days = design_days[0]

                    idf_origin.copyidfobject(design_days)

                    # Update simulation period
                    run_periods = idf_origin.idfobjects["RUNPERIOD"]
                    if run_periods:
                        run_period = run_periods[0]
                        run_period.Begin_Month = start_date.month
                        run_period.Begin_Day_of_Month = start_date.day
                        run_period.Begin_Year = start_date.year

                        run_period.End_Month = end_date.month
                        run_period.End_Day_of_Month = end_date.day
                        run_period.End_Year = end_date.year

                    idf_origin.save("models/temp/temp.idf")

                    idf = IDF("models/temp/temp.idf")

                    thermal_zones = []
                    for zone in idf.idfobjects['ZONEHVAC:EQUIPMENTCONNECTIONS']:
                        thermal_zones.append(zone.Zone_Name)
                    space_num = len(thermal_zones)

                    # Space-> Zone map
                    space_zone_map = get_space_zone_mapping(idf)

                    # map from the Spaces.json file, space_name -> {}
                    description_map = extract_description_mapping("SmartSPEC/Updated model/Spaces.json")

                    spaces = {}
                    for space_name, space_data in description_map.items():
                        if "Space" in space_name:
                            if not "Space 1 - 1" == space_name and not "Space 2 - 4" == space_name:
                                spaces[space_name] = Space(space_data["id"], 23, 23, space_data["capacity"])
                    # # objects of all zone_objs, where includes environmental conditions and properties. Created for H2H
                    # area_by_zone_name = cal_ep_zone_area(idf)
                    # zone_objs = {}
                    # for index, zone in enumerate(thermal_zones):
                    #     if "Corridor" not in zone:
                    #         zone_objs[int(zone.lstrip("Thermal Zone"))] = Space(int(zone.lstrip("Thermal Zone")), 23,
                    #                                                             None, area_by_zone_name[zone])
                    # zone_objs[44] = Space(44, 23, 40, 20)
                    # occupants = Participant(space_num, path_profile, "knn/ashrae_comfort_data.csv", "s1_100", path_trajectories, "historical_based")
                    occupants = Participant(space_num, path_profile, [30, 40, 30], "s1_100",
                                            path_trajectories, "prior_knowledge", spaces, space_zone_map, description_map, args.h2h)

                    occupants.clean()

                    result = Result(occupants, city, spaces, args.comment)

                    occupants.result = result

                    input_param = {}
                    output_param = {}
                    for index, zone in enumerate(thermal_zones):
                        input_param['sch_clg_' + str(index + 1)] = ["Zone Temperature Control", "Cooling Setpoint",
                                                                    zone, 50]
                        input_param['sch_htg_' + str(index + 1)] = ["Zone Temperature Control", "Heating Setpoint",
                                                                    zone, 0]
                        output_param['temp' + str(index + 1)] = ["Zone Air Temperature", zone]

                    # handles for energy consumption:
                    for index, zone in enumerate(thermal_zones):
                        output_param["ec_clg_tz" + str(index + 1)] = ["Zone Air System Sensible Cooling Energy", zone]
                        output_param["ec_htg_tz" + str(index + 1)] = ["Zone Air System Sensible Heating Energy", zone]
                    output_param.update(
                        {'ec_clg_coil': ["Cooling Coil Electricity Energy", "COIL COOLING DX TWO SPEED 1"],
                         'ec_htg_coil': ["Heating Coil Electricity Energy", "1 SPD DX HTG COIL"],
                         'temp_out': ["Site Outdoor Air Drybulb Temperature", "Environment"]})

                    idf_fans = idf.idfobjects["Fan:OnOff"]
                    for fan in idf_fans:
                        output_param["ec_fan_" + fan.Name] = ["Fan Electricity Energy", fan.Name]


                    if algo == "const":
                        turn_off_when_empty = False
                    else:
                        turn_off_when_empty = True

                    co_sim = CoSimulation("models/temp/temp.idf", start_date, end_date, input_param, output_param,
                                          occupants.vote,
                                          SensationAggregator(algo, occupants), generate_set_point, result,
                                          occupants.update_loss, epw_file_path, turn_off_when_empty,
                                          weather_file_mapping[city] + ".epw", hvac_control_strategy, spaces, space_zone_map, description_map)
                    co_sim.run()

                    result.save_result(start_time.strftime("%Y%m%d%H%M%S"),
                                       city + "_" + algo + "_" + str(hvac_control_strategy) + "_" + str(repeat_time),
                                       co_sim.total_people_count)

                    end_time = datetime.now()
                    used_time = end_time - start_time
                    current_simulation_iteration += 1
                    avg_time = used_time / current_simulation_iteration
                    total_time = avg_time * total_simulation_iteration
                    required_time = total_time - used_time
                    sleep(1)
                    print(Fore.CYAN + Back.MAGENTA + "total time" + str(total_time) + Style.RESET_ALL)
                    print(Fore.CYAN + Back.MAGENTA + "used time" + str(used_time) + Style.RESET_ALL)
                    print(Fore.CYAN + Back.MAGENTA + "require time" + str(required_time) + Style.RESET_ALL)

                    print("Done:" + city + "_" + algo + "_" + str(hvac_control_strategy) + "\n\n\n")

                    # is_finished = True
                    # while is_finished:
                    #     sim_time = datetime(2010, 1, 1) + timedelta(seconds=co_sim.time)


def cal_ep_zone_area(idf):
    # 几何计算工具：通过顶点坐标计算多边形面积
    def calculate_polygon_area(vertices):
        n = len(vertices)
        area = 0
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            area += x1 * y2 - y1 * x2
        return abs(area) / 2

    # 获取所有 ZONEHVAC:EQUIPMENTCONNECTIONS 对象
    equipment_connections = idf.idfobjects['ZONEHVAC:EQUIPMENTCONNECTIONS']

    # 初始化存储区域面积的字典
    zone_areas = {}

    # 遍历每个 BuildingSurface:Detailed 对象
    surfaces = idf.idfobjects['BUILDINGSURFACE:DETAILED']
    for surface in surfaces:
        zone_name = surface.Zone_Name
        vertices = []

        # 动态检测顶点坐标的数量（跳过空值）
        for i in range(1, 100):  # 假设最多有 100 个顶点
            x_attr = f"Vertex_{i}_Xcoordinate"
            y_attr = f"Vertex_{i}_Ycoordinate"
            if hasattr(surface, x_attr) and hasattr(surface, y_attr):
                x = getattr(surface, x_attr, None)
                y = getattr(surface, y_attr, None)
                if x is not None and y is not None and x and y:
                    vertices.append((float(x), float(y)))
                else:
                    break
            else:
                break

        # 如果顶点数不足以构成多边形，跳过此表面
        if len(vertices) < 3:
            # print(f"Warning: Surface {surface.Name} has insufficient vertices. Skipping...")
            continue

        # 计算当前表面的面积
        area = calculate_polygon_area(vertices)

        # 累加到对应的 ZONE
        if zone_name in zone_areas:
            zone_areas[zone_name] += area
        else:
            zone_areas[zone_name] = area
    area = {}
    # 输出每个 ZONE 的总面积
    for connection in equipment_connections:
        zone_name = connection.Zone_Name

        area[zone_name] = round(zone_areas.get(zone_name, 0.0), 2)
    return area


def get_space_zone_mapping(idf):
    """
    解析 EnergyPlus IDF 文件，提取 Space 和 Thermal Zone 之间的对应关系，并返回字典格式。

    :param idf: str, IDF 文件路径
    :return: dict, {Space_Name: Thermal_Zone_Name}
    """

    # 存储映射关系的字典
    space_zone_dict = {}

    # 遍历 Space 对象
    for space in idf.idfobjects["SPACE"]:
        space_name = space.Name  # 空间名称
        thermal_zone = space.Zone_Name  # 关联的 Thermal Zone 名称

        # 存入字典
        space_zone_dict[space_name] = thermal_zone

    return space_zone_dict


def extract_description_mapping(json_file):
    """
    解析 JSON 文件，提取每个 description 对应的 id 和 room_type。

    :param json_file: str, JSON 文件路径
    :return: dict, {description: {"id": id, "room_type": room_type}}
    """
    # 读取 JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构造映射字典
    description_mapping = {
        item["description"]: {"id": item["id"], "room_type": item["room_type"], "capacity": item["capacity"], "description": item["description"]}
        for item in data
    }

    return description_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-zyBench + H2H")
    parser.add_argument("-f", "--h2h", type=lambda x: x.lower() == 'true',
                        help="Running H2H True or False", default=True)
    parser.add_argument("-c", "--comment", type=str,
                        help="A comment followed by the output file name", default="")
    parser.add_argument("-a", "--algorithm", type=str,
                        help="TCPS Algorithm that will be accessed (majority, fair, drift)", default="majority")

    main(parser.parse_args())
