import json
import math
import re

from eppy import modeleditor
from eppy.modeleditor import IDF
import numpy as np
import csv
import matplotlib.pyplot as plt
import networkx as nx

def name_to_id(name):
    """
    根据数据点名称生成节点 id：
      - "Space 1 - X"  => 100 + X
      - "Space 2 - Y"  => 200 + Y
      - "Corridor 1 - X"  => 300 + X
      - "Corridor 2 - Y"  => 400 + Y
    """
    name = name.strip()
    if name == "outside":
        return 0
    if not name:
        return None
    m = re.match(r"(Space|Corridor)\s+(\d)\s*-\s*(\d+)", name)
    if m:
        typ, level, num = m.groups()
        level = int(level)
        num = int(num)

        if typ == "Gate":
            return 100 + num

        elif typ == "Space":
            if level == 2:
                return 200 + num
            elif level == 3:
                return 300 + num
            else:
                raise Exception("No this space? type is: " + typ + "level is: " + str(level))
        elif typ == "Corridor":
            if level == 1:
                return 400 + num
            elif level == 2:
                return 500 + num
    return None


def compute_polygon_area(vertices):
    """
    根据二维多边形顶点列表计算面积（鞋带公式）
    vertices: [[x1, y1], [x2, y2], ..., [xn, yn]]
    """
    if len(vertices) < 3:
        return 0
    area = 0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def get_space_area_from_idf(space_name, idf):
    """
    根据给定的 space_name 在 IDF 文件中查找对应的
    Floor 类型的 BuildingSurface:Detailed 对象，并计算面积。

    计算方法：
      1. 遍历 idf.idfobjects["BuildingSurface:Detailed"]
      2. 如果该对象的 "Space Name" 与 space_name 匹配，
         且 "Surface Type" 为 "Floor"（忽略大小写），
         则尝试读取该对象的顶点数据（字段 "Vertex 1 X-coordinate", "Vertex 1 Y-coordinate", ...）。
      3. 利用二维顶点（仅取 X 和 Y 坐标）利用鞋带公式计算多边形面积，
         累加所有 Floor 面的面积即为该 Space 的总面积。

    如果找不到对应数据，则返回 0。
    """
    total_area = 0.0
    try:
        surfaces = idf.idfobjects["BuildingSurface:Detailed"]
    except KeyError:
        return 0

    for surface in surfaces:
        # 判断对象是否属于当前空间，并且 Surface Type 为 Floor

        if surface["Space_Name"].strip().lower() == space_name.strip().lower() and \
                surface["Surface_Type"].strip().lower() == "floor":
            vertices = []
            i = 1
            while True:
                # 字段名与 IDF 中保持一致，这里直接用原始名称
                x_key = f"Vertex_{i}_Xcoordinate"
                y_key = f"Vertex_{i}_Ycoordinate"
                try:
                    x = float(surface[x_key])
                    y = float(surface[y_key])
                except Exception:
                    break
                vertices.append([x, y])
                i += 1
            if len(vertices) >= 3:
                area = compute_polygon_area(vertices)
                total_area += area
    return total_area

def generate_rooms_json_and_csv():
    # 设置 EnergyPlus IDD 文件路径 (需要与你的 IDF 版本匹配)
    iddfile = "D:\Programming\EnergyPlus\EnergyPlusV22-2-0\Energy+.idd"  # 需要确保这个文件路径正确
    idf_file = "models/orly/orly.idf"  # 你的 EnergyPlus IDF 文件

    # 加载 IDD 文件
    IDF.setiddname(iddfile)

    # 读取 IDF 文件
    idf = IDF(idf_file)

    # 存储房间数据
    rooms_data = {}

    # 遍历所有建筑表面
    for surface in idf.idfobjects["BUILDINGSURFACE:DETAILED"]:
        room_name = surface.Space_Name  # 这里 Zone_Name 可能表示房间

        if room_name not in rooms_data:
            rooms_data[room_name] = {"x": [], "y": [], "z": []}
            rooms_data[room_name]["area"] = get_space_area_from_idf(room_name, idf)

        # 提取顶点坐标
        for coord in surface.coords:
            rooms_data[room_name]["x"].append(coord[0])
            rooms_data[room_name]["y"].append(coord[1])
            rooms_data[room_name]["z"].append(coord[2])

    # 计算房间中心点并存入 JSON
    output_data = []
    for room_name, coords in rooms_data.items():
        if coords["x"] and coords["y"]:
            center_x = sum(coords["x"]) / len(coords["x"])
            center_y = sum(coords["y"]) / len(coords["y"])
            center_z = min(coords["z"]) if coords["z"] else 0  # 取最低点

            if "Space" in room_name:
                capacity = rooms_data[room_name]["area"]/1.5
            else:
                capacity = -1

            room_data = {
                "name": room_name,
                "id": name_to_id(room_name),
                "coordinates": [center_x, center_y, center_z],
                "neighbors": [],  # 可扩展
                "description": room_name,
                "area": rooms_data[room_name]["area"],
                "capacity": None

            }

            output_data.append(room_data)

    # 保存 JSON 文件
    with open("orly_rooms.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    # json_filename = "orly_rooms.json"
    # csv_filename = "updated_orly_rooms.csv"
    #
    # with open(json_filename, "r", encoding="utf-8") as json_file:
    #     data_list = json.load(json_file)
    #
    # # Open the CSV file and write data
    # with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #
    #     # Write the header
    #     writer.writerow(["name", "x", "y", "z", "neighbors"])
    #
    #     # Write the row data
    #     for data in data_list:
    #         writer.writerow([
    #             data["name"],
    #             data["coordinates"][0],
    #             data["coordinates"][1],
    #             data["coordinates"][2],
    #             ",".join(map(str, data["neighbors"]))  # Convert list to comma-separated string
    #         ])
    #
    # print(f"CSV file '{csv_filename}' has been created successfully.")


def draw_all_rooms_as_points(json_file):

    # 读取 JSON 数据
    with open(json_file, "r") as f:
        data = json.load(f)

    # 初始化图
    G = nx.Graph()

    # 解析数据
    nodes = {}  # 存储节点及其坐标
    edges = []  # 存储边信息

    # 读取所有节点
    for node in data:
        if "1 -" in node["name"]:
            continue
        node_id = node["name"]
        x, y, _ = node["coordinates"]  # 取前两个坐标
        nodes[node_id] = (x, y)
        G.add_node(node_id, pos=(x, y))  # 添加节点

    # 读取所有边（双向连接）
    for node in data:
        node_id = node["name"]
        for neighbor in node["neighbors"]:
            if neighbor in nodes:  # 确保邻居存在于数据中
                edges.append((node_id, neighbor))
                G.add_edge(node_id, neighbor)

    # 绘图
    plt.figure(figsize=(24, 20))

    # 获取节点位置
    pos = nx.get_node_attributes(G, "pos")

    # 画节点
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue", edgecolors="black")

    # 画边
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, alpha=0.6)

    # 画标签
    # **画标签（倾斜显示）**
    for node, (x, y) in pos.items():
        if x < 0:  # **左侧的节点**
            angle = -0
            ha = "right"
        else:  # **右侧的节点**
            angle = 0
            ha = "left"

        plt.text(
            x, y, node, fontsize=8, fontweight="bold", color="black",
            horizontalalignment=ha, verticalalignment="center",
            rotation=angle
        )

    # 显示图像
    plt.title("Graph Visualization with Coordinates and Neighbors")
    plt.axis("off")
    plt.show()




# -----------------------------------------------

def load_idf(idf_filename, idd_filename):
    """
    利用 eppy 加载 EnergyPlus 的 IDF 文件，需要先指定 IDD 文件。
    """
    try:
        IDF.setiddname(idd_filename)
        idf = IDF(idf_filename)
        return idf
    except Exception as e:
        print(f"Error loading IDF file: {e}")
        return None

def csv_to_json(csv_filename, json_filename, idf, outside_name="outside"):
    """
    1. 读取 CSV 文件，构建所有节点及双向邻居关系（邻居保存为集合，后转换为列表）。
    2. 对于 CSV 中每个节点，其名称由规则转换为 id。
    3. 若 CSV 中某节点列出了邻居名称，则同时构造邻居的双向关系；
       如果某个邻居在 CSV 中没有单独记录，则创建默认节点（坐标为空列表，capacity=-1）。
    4. 对于名称中以 "Space" 开头的节点，通过 eppy 从 IDF 中计算其面积，并将面积赋给 capacity；
       对于 Corridor 以及 outside 节点，capacity 保持 -1。
    5. 最后将所有节点以列表形式写入 JSON 文件。
    """
    # 用字典存储所有节点，key 为节点 id
    nodes_dict = {0: {
                    "id": 0,
                    "description": "outside",
                    "capacity": -1,  # 后续针对 Space 节点更新
                    "neighbors": set(),
                    "coordinates": [0, 0, 0]
                }}

    with open(csv_filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # 如果 CSV 有表头，则跳过

        for row in reader:
            if not row or len(row) < 4:
                continue

            name = row[0].strip()
            if not name:
                continue

            node_id = name_to_id(name)
            if node_id is None:
                continue

            try:
                x = float(row[1])
                y = float(row[2])
                z = float(row[3])
            except ValueError:
                continue

            # 如果节点已存在（可能之前作为邻居被添加），更新基本信息
            if node_id not in nodes_dict:
                nodes_dict[node_id] = {
                    "id": node_id,
                    "description": name,
                    "capacity": -1,  # 后续针对 Space 节点更新
                    "neighbors": set(),
                    "coordinates": [x, y, z]
                }
            else:
                nodes_dict[node_id]["description"] = name
                nodes_dict[node_id]["coordinates"] = [x, y, z]

            # 处理 CSV 中后续的邻居字段（可能为空）
            neighbor_names = [n.strip() for n in row[4:] if n.strip() != '']
            for nb_name in neighbor_names:
                nb_id = name_to_id(nb_name)
                if nb_id is None:
                    continue

                # 将 nb_id 加入当前节点的邻居中
                nodes_dict[node_id]["neighbors"].add(nb_id)

                # 确保邻居节点也存在；如果不存在则创建默认节点
                if nb_id not in nodes_dict:
                    nodes_dict[nb_id] = {
                        "id": nb_id,
                        "description": nb_name,
                        "capacity": -1,  # 默认值
                        "neighbors": {node_id},
                        "coordinates": []  # 未提供坐标信息
                    }
                else:
                    nodes_dict[nb_id]["neighbors"].add(node_id)

    # 如果 CSV 中没有定义 "outside" 节点，则单独创建一个 outside 节点，其 capacity 为 -1
    outside_id = 0
    if outside_id not in nodes_dict:
        nodes_dict[outside_id] = {
            "id": outside_id,
            "description": outside_name,
            "capacity": -1,
            "neighbors": set(),
            "coordinates": []
        }


    # 根据 IDF 数据更新节点信息
    for node in nodes_dict.values():
        desc = node["description"]
        # 针对 Space 节点：利用 IDF 数据计算面积，并根据面积确定房间类型及人员容量

        area = get_space_area_from_idf(desc, idf)
        node["area"] = area
        # 如果面积无效，则标记为未知
        capacity = node["capacity"]
        if desc.lower().startswith("gate"):
            room_type = "gate"
            node["room_type"] = room_type
            node["capacity"] = capacity
        elif desc.lower().startswith("space") and "3 -" not in desc:
            room_type = "waiting area"
            capacity = node["area"] / 1.5
            node["room_type"] = room_type
            node["capacity"] = capacity

        elif desc.lower().startswith("space") and "3 -" in desc:
            room_type = "lounge"
            capacity = node["area"] / 3

            node["room_type"] = room_type
            node["capacity"] = capacity
        # 对于 Corridor 节点
        elif desc.lower().startswith("corridor"):
            node["area"] = 0
            node["room_type"] = "Corridor"
            node["capacity"] = -1
        # 对于 outside 节点
        elif desc.lower() == "outside":
            node["area"] = 0
            node["room_type"] = "outside"
            node["capacity"] = -1
        else:
            node["area"] = 0
            node["room_type"] = None
            node["capacity"] = -1

    # 将邻居集合转换为列表，并整理输出数据
    output_nodes = []
    for node in nodes_dict.values():
        node["neighbors"] = sorted(list(node["neighbors"]))
        output_nodes.append(node)

    # 可选：按照 id 排序输出
    output_nodes.sort(key=lambda n: n["id"])

    with open(json_filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(output_nodes, jsonfile, indent=4, ensure_ascii=False)

def summarize_json(json_filename):
    """
    读取 JSON 文件后，按照 room_type 对所有节点的 area 和 capacity 进行汇总统计。
    返回一个字典，key 为 room_type，value 为包含节点数量、总面积和总容量的字典。
    """
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary = {}
    for node in data:
        # 取得节点的 room_type，如果没有则设为 "未知"
        room_type = node.get("room_type", "未知")
        # 如果该 room_type 不在统计字典中，初始化
        if room_type not in summary:
            summary[room_type] = {"count": 0, "total_area": 0, "total_capacity": 0}
        # 累加节点数、面积和容量（注意：如果 capacity 是 -1，则仍然会累加进去，可根据需要处理）
        summary[room_type]["count"] += 1
        summary[room_type]["total_area"] += node.get("area", 0)
        summary[room_type]["total_capacity"] += node.get("capacity", 0)

    print("按照 room_type 的汇总统计：")
    for room_type, stats in summary.items():
        print(
            f"{room_type}: 节点数量 = {stats['count']}, 总面积 = {stats['total_area']}, 总容量 = {stats['total_capacity']}")


def update_meta_events(spaces_filename, meta_events_filename, output_filename):
    # 加载 Spaces.json 文件，得到所有空间数据
    with open(spaces_filename, 'r', encoding='utf-8') as f:
        spaces = json.load(f)

    # 加载 MetaEvents.json 文件，支持文件中为单个对象或列表
    with open(meta_events_filename, 'r', encoding='utf-8') as f:
        meta_events = json.load(f)

    # 定义函数：根据 MetaEvent 描述过滤出符合要求的空间 id
    def filter_space_ids(desc):
        desc_lower = desc.lower()
        selected_ids = []
        for space in spaces:
            # 获取房间类型与容量
            room_type = space.get("room_type", "")
            cap = space.get("capacity", -1)
            # 规则1：如果 description 包含 "employee"
            if "employee" in desc_lower:
                # 选择所有 "中办公室" 和 "大办公室"
                if room_type in ["Mid Office", "Large Office"]:
                    selected_ids.append(space["id"])
            # 规则2：如果 description 等于 "manager"
            elif desc.lower() == "manager":
                # 选择 "小办公室" 和 (room_type=="中办公室" 且 capacity==2)
                if room_type == "Small Office":
                    selected_ids.append(space["id"])
                elif room_type == "Mid Office" and cap == 2:
                    selected_ids.append(space["id"])
            # 规则3：如果 description 包含 "meeting"
            elif "meeting" in desc_lower:
                if room_type == "Meeting Room":
                    selected_ids.append(space["id"])
            # 规则4：如果 description 等于 "lunch"
            elif desc.lower() == "lunch":
                if room_type == "outside":
                    selected_ids.append(space["id"])
        return selected_ids

    # 更新 MetaEvents 的 "spaces" 字段
    # 如果 meta_events 为列表，则逐个更新；如果为字典，则直接更新
    if isinstance(meta_events, list):
        for event in meta_events:
            description = event.get("description", "")
            new_ids = filter_space_ids(description)
            # 更新 event 的 spaces 部分
            if "spaces" not in event or not isinstance(event["spaces"], dict):
                event["spaces"] = {}
            event["spaces"]["space-ids"] = new_ids
            event["spaces"]["number"] = len(new_ids)
    elif isinstance(meta_events, dict):
        description = meta_events.get("description", "")
        new_ids = filter_space_ids(description)
        if "spaces" not in meta_events or not isinstance(meta_events["spaces"], dict):
            meta_events["spaces"] = {}
        meta_events["spaces"]["space-ids"] = new_ids
        meta_events["spaces"]["number"] = len(new_ids)

    # 将修改后的 MetaEvents 数据写入 output_filename
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(meta_events, f, indent=4, ensure_ascii=False)


def update_event_space_ids(mateevents_file, events_file, output_file):
    # 加载 MateEvents.json 文件
    with open(mateevents_file, 'r', encoding='utf-8') as f:
        mate_events = json.load(f)

    # 将 mate_events 按照 "id" 构造为字典，便于查找
    mate_events_by_id = {}
    if isinstance(mate_events, list):
        for me in mate_events:
            if "id" in me:
                mate_events_by_id[me["id"]] = me
    elif isinstance(mate_events, dict):
        # 如果是单个对象，也直接存入字典
        if "id" in mate_events:
            mate_events_by_id[mate_events["id"]] = mate_events

    # 加载 Event.json 文件
    with open(events_file, 'r', encoding='utf-8') as f:
        events = json.load(f)

    # 根据事件数据结构进行处理：如果事件数据是列表，则逐个处理；如果是单个对象，则直接处理
    def process_event(event):
        metaevent_id = event.get("metaevent-id")
        if metaevent_id is None:
            return  # 如果没有 metaevent-id 则不处理
        meta_event = mate_events_by_id.get(metaevent_id)
        if meta_event is None:
            return  # 没有找到对应的 meta event
        # 如果在 meta event 中存在 spaces 字段，并且有 space-ids，则更新
        spaces = meta_event.get("spaces")
        if spaces and "space-ids" in spaces:
            event["space-ids"] = spaces["space-ids"]

    if isinstance(events, list):
        for event in events:
            process_event(event)
    elif isinstance(events, dict):
        process_event(events)

    # 将更新后的事件数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # CSV 与输出 JSON 文件路径
    csv_filename = "updated_orly_rooms.csv"
    json_filename = "orly_output_space.json"

    # EnergyPlus 文件（IDF）及其对应的 IDD 文件路径，请根据实际情况修改
    idf_filename = "models/orly/orly.idf"
    idd_filename = "D:\Programming\EnergyPlus\EnergyPlusV22-2-0\Energy+.idd"

    # TO create Space JSON file based on csv and idf files.
    idf = load_idf(idf_filename, idd_filename)
    if idf is None:
        print("加载 IDF 文件失败！")
    else:
        csv_to_json(csv_filename, json_filename, idf)

    # Summarize the json file by numbers, area and capacity of space types.
    summarize_json(json_filename)

    # # Update MetaEvents with new generated Spaces.json
    # spaces_filename = "output_space.json"
    # meta_events_filename = "SmartSPEC/Previous model/MetaEvents.json"
    # output_filename = "SmartSPEC/Previous model/MetaEvents.json"
    # update_meta_events(spaces_filename, meta_events_filename, output_filename)
    # print(f"Updated MetaEvents is saved as {output_filename}")


    # # Update Event based on updated MetaEvents
    # mateevents_file = "SmartSPEC/Updated model/MetaEvents.json"
    # events_file = "SmartSPEC/Previous model/Events.json"
    # output_file = "SmartSPEC/Updated model/Events.json"
    # update_event_space_ids(mateevents_file, events_file, output_file)
    # print(f"Updated Events is saved as {output_file}")



# --------------------------------------------------
# json_filename = "orly_rooms.json"
# csv_filename = "updated_orly_rooms.csv"
# generate_rooms_json_and_csv()
# # draw_all_rooms_as_points(json_filename)
