import json
import heapq
from math import sqrt


class Graph:
    def __init__(self, json_file):
        """从 JSON 文件初始化图结构"""
        self.graph = {}  # 邻接表
        self.coordinates = {}  # 坐标信息
        self._load_data(json_file)

    def _load_data(self, json_file):
        """从 JSON 文件加载数据"""
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        for node in data:
            node_id = node["id"]
            self.graph[node_id] = node["neighbors"]
            self.coordinates[node_id] = tuple(node["coordinates"]) if node["coordinates"] else None

    def euclidean_distance(self, coord1, coord2):
        """计算两个三维坐标之间的欧几里得距离"""
        return sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

    def shortest_path(self, start_id, end_id):
        """使用 Dijkstra 算法计算最短路径，并返回路径和距离"""
        if start_id not in self.graph or end_id not in self.graph:
            return float('inf')  # 无法到达

        # 优先队列（最小堆），存储 (当前距离, 节点ID)
        pq = [(0, start_id)]
        distances = {node_id: float('inf') for node_id in self.graph}
        distances[start_id] = 0

        # 记录路径来源，用于回溯
        predecessors = {start_id: None}

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            # 目标点找到，返回最短距离和路径
            if current_node == end_id:
                # path = self._reconstruct_path(predecessors, start_id, end_id)
                # return current_dist, path
                return current_dist

            for neighbor in self.graph[current_node]:
                if self.coordinates[current_node] and self.coordinates[neighbor]:
                    dist = self.euclidean_distance(self.coordinates[current_node], self.coordinates[neighbor])
                else:
                    continue  # 跳过无坐标的点

                new_distance = current_dist + dist
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node  # 记录前驱节点
                    heapq.heappush(pq, (new_distance, neighbor))

        return float('inf')  # 如果无法到达，返回无穷大

    # def _reconstruct_path(self, predecessors, start_id, end_id):
    #     """回溯路径"""
    #     path = []
    #     current = end_id
    #     while current is not None:
    #         path.append(current)
    #         current = predecessors.get(current)
    #     return path[::-1]  # 逆序返回正确路径



# # 创建图实例（从 JSON 文件加载）
# graph = Graph("SmartSPEC/Updated model/Spaces.json")
#
# # 多次查询最短路径
# print(graph.shortest_path(104, 314))  # 查询 101 到 321
# # print(graph.shortest_path(101, 320))  # 查询 101 到 320
# # print(graph.shortest_path(320, 321))  # 查询 320 到 321
