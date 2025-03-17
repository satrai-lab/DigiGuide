import json
import heapq
from math import sqrt


class Graph:
    def __init__(self, json_file):
        """Initialize graph structure from JSON file"""
        self.graph = {}  # Adjacency list
        self.coordinates = {}  # Coordinate information
        self._load_data(json_file)

    def _load_data(self, json_file):
        """Load data from JSON file"""
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        for node in data:
            node_id = node["id"]
            self.graph[node_id] = node["neighbors"]
            self.coordinates[node_id] = tuple(node["coordinates"]) if node["coordinates"] else None

    def euclidean_distance(self, coord1, coord2):
        """Calculate Euclidean distance between two 3D coordinates"""
        return sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

    def shortest_path(self, start_id, end_id):
        """Calculate shortest path using Dijkstra's algorithm and return path and distance"""
        if start_id not in self.graph or end_id not in self.graph:
            return float('inf')  # Cannot reach

        # Priority queue (min-heap), store (current distance, node ID)
        pq = [(0, start_id)]
        distances = {node_id: float('inf') for node_id in self.graph}
        distances[start_id] = 0

        # Record path source for backtracking
        predecessors = {start_id: None}

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            # Target point found, return shortest distance and path
            if current_node == end_id:
                # path = self._reconstruct_path(predecessors, start_id, end_id)
                # return current_dist, path
                return current_dist

            for neighbor in self.graph[current_node]:
                if self.coordinates[current_node] and self.coordinates[neighbor]:
                    dist = self.euclidean_distance(self.coordinates[current_node], self.coordinates[neighbor])
                else:
                    continue  # Skip points without coordinates

                new_distance = current_dist + dist
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node  # Record predecessor node
                    heapq.heappush(pq, (new_distance, neighbor))

        return float('inf')  # If cannot reach, return infinity

    # def _reconstruct_path(self, predecessors, start_id, end_id):
    #     """回溯路径"""
    #     path = []
    #     current = end_id
    #     while current is not None:
    #         path.append(current)
    #         current = predecessors.get(current)
    #     return path[::-1]  # Reverse to return correct path



# # 创建图实例（从 JSON 文件加载）
# graph = Graph("SmartSPEC/Updated model/Spaces.json")
#
# # 多次查询最短路径
# print(graph.shortest_path(104, 314))  # 查询 101 到 321
# # print(graph.shortest_path(101, 320))  # 查询 101 到 320
# # print(graph.shortest_path(320, 321))  # 查询 320 到 321
