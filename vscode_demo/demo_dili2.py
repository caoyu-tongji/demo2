import math

def haversine(coord1, coord2):
    # 地球平均半径，单位为米
    R = 6371 * 1000  # 地球平均半径（公里）转换成米
    
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # 将角度转换为弧度
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine公式
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    
    return d

# 当前位置的经纬度
current_position = (31.2272, 121.4925)  # 示例位置，假设您当前的位置
# 旅游景区的经纬度
destination = (31.2283, 121.4730)  # 豫园位置

distance = haversine(current_position, destination)
print(f"与景点的距离是 {distance:.2f} 米")  # 输出距离