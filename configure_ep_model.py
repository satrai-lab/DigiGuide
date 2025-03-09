from eppy import modeleditor
from eppy.modeleditor import IDF

# 设置EnergyPlus IDD文件的路径
idd_file = r"D:\Programming\EnergyPlus\EnergyPlusV22-2-0\Energy+.idd"

# 设置要修改的IDF文件的路径
idf_path = "./models/drahix/drahix.idf"

# 使用Eppy加载IDD文件
IDF.setiddname(idd_file)

# 打开IDF文件
idf = IDF(idf_path)


# for i, people in enumerate(idf.idfobjects["PEOPLE"]):
#     schedule_name = "People_Schedule_" + str(i)
#
#     # 创建新的Schedule:Compact对象
#     schedule = idf.newidfobject("SCHEDULE:COMPACT",
#                                 Name=schedule_name,
#                                 Schedule_Type_Limits_Name="Fraction",
#                                 Field_1="Through: 12/31",
#                                 Field_2="For: AllDays",
#                                 Field_3="Until: 24:00",
#                                 Field_4="0.0")
#
#     # 将生成的Schedule名称分配给当前的People对象
#     people.Number_of_People_Schedule_Name = schedule_name
#     people.Number_of_People_Calculation_Method = "People"
#     people.Number_of_People = 100


# for space in idf.idfobjects["SPACE"]:
#     # 创建新的ThermalZone对象
#     thermal_zone_name = f"ThermalZone_{space.Name}"
#     thermal_zone = idf.newidfobject("ZONE",
#                                     Name=thermal_zone_name)
#
#     # 将Space对象的热区属性设置为新创建的热区
#     space.Zone_Name = thermal_zone_name
#
#
# # 保存修改后的IDF文件
# idf.save("./models/configured/output_withzones.idf")

print(len(idf.idfobjects["SPACE"]))
