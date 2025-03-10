import json
import os
import time
from collections import defaultdict


def generate_smartspec_model(people, start_points, end_points):
    # 创建带有时间戳的文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    people_data = []
    # 示例输入数据: p_id 从 s_i 到 s_j
    for i in range(len(people)):
        people_data.append((people[i], start_points[i], end_points[i]))

    # 生成 MetaPeople.json 数据
    meta_people = []
    for p_id, s_i, s_j in people_data:
        meta_people.append({
            "id": p_id,
            "description": "upper-div-ugrad",
            "probability": 2000.0,
            "event-affinity": [
                {"metaevent-id": 1, "probability": 10.0}
            ],
            "time-profiles": [
                {
                    "probability": 1.0,
                    "profile": [
                        {
                            "pattern": {
                                "start-date": "2020-01-01",
                                "end-date": "2020-12-01",
                                "period": "day",
                                "period-details": {"repeat-every": 1}
                            },
                            "duration": {
                                "start-time": ["00:00:00", "00:00:00"],
                                "end-time": ["23:00:00", "00:10:00"],
                                "required": ["20:00:00", "00:10:00"]
                            }
                        }
                    ]
                }
            ]
        })

    with open(os.path.join(output_dir, "MetaPeople.json"), "w") as f:
        json.dump(meta_people, f, indent=4)

    # 生成 People.json 数据
    people = []
    for p_id, s_i, s_j in people_data:
        people.append({
            "id": p_id,
            "metaperson-id": p_id,
            "description": "a",
            "profile-index": 0,
            "initial-location": s_i
        })

    with open(os.path.join(output_dir, "People.json"), "w") as f:
        json.dump(people, f, indent=4)

    # 生成 MetaEvents.json 数据
    meta_events = []
    space_groups = defaultdict(list)

    for p_id, s_i, s_j in people_data:
        space_groups[s_j].append(p_id)

    for space_id, p_ids in space_groups.items():
        meta_events.append({
            "id": len(meta_events) + 1,
            "description": "lower-div-lect-large",
            "probability": 5.0,
            "spaces": {"space-ids": [space_id], "number": 1},
            "capacity": [
                {
                    "metaperson-id": 0,
                    "lo": [90.0, 5.0],
                    "hi": [225.0, 5.0]
                }
            ],
            "time-profiles": [
                {
                    "probability": 1.0,
                    "profile": [
                        {
                            "pattern": {
                                "start-date": "2020-01-01",
                                "end-date": "2020-12-01",
                                "period": "day",
                                "period-details": {"repeat-every": 1}
                            },
                            "duration": {
                                "start-time": ["00:00:00", "00:02:00"],
                                "end-time": ["23:50:00", "00:02:00"],
                                "required": ["00:50:00", "00:02:00"]
                            }
                        }
                    ]
                }
            ]
        })

    with open(os.path.join(output_dir, "MetaEvents.json"), "w") as f:
        json.dump(meta_events, f, indent=4)

    # 生成 Events.json 数据
    events = []
    for idx, (p_id, s_i, s_j) in enumerate(people_data, start=1):
        events.append({
            "id": idx,
            "metaevent-id": 1,
            "description": "research-project-meeting",
            "profile-index": 0,
            "space-ids": [s_j],
            "capacity": [{"metaperson-id": -1, "range": [20, 12]}]
        })

    with open(os.path.join(output_dir, "Events.json"), "w") as f:
        json.dump(events, f, indent=4)

    print(f"JSON 文件已生成在文件夹: {output_dir}")
