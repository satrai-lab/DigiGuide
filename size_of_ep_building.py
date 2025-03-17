from eppy import modeleditor
from eppy.modeleditor import IDF

# Set IDF and IDD file paths
idf_path = r"C:\Users\XiaoMA\Desktop\DoE buildings\FRA_Paris.Orly.071490_IWEC\RefBldgSecondarySchoolNew2004_v1.4_7.2_1A_USA_FL_MIAMI.idf"
# Read IDF file content
with open(idf_path, 'r') as file:
    idf_content = file.readlines()

# Find all Zone objects
zone_count = 0
zone_names = []
inside_zone_block = False

for line in idf_content:
    line = line.strip()
    if line.lower().startswith("zone,"):
        zone_count += 1
        inside_zone_block = True
    elif inside_zone_block:
        zone_name = line.split(',')[0]  # Extract Zone name
        zone_names.append(zone_name)
        inside_zone_block = False

# Print number of Zones
print(f"Number of Zones in the IDF file: {zone_count}")

# Optional: Print each Zone's name
for zone_name in zone_names:
    print(f"Zone Name: {zone_name}")