import requests

def search_poi_near_yuyuan(api_key, radius=500, max_pages=5):
    # 豫园中心坐标（高德经纬度，需先通过地理编码获取准确值）
    center_lnglat = "121.492512,31.227174"  # 经度,纬度
    
    # POI类型：风景名胜（可根据需求修改，如餐饮=050000）
    poi_type = "风景名胜"
    
    base_url = "https://restapi.amap.com/v3/place/around"
    
    pois = []
    page = 1
    
    while page <= max_pages:
        params = {
            "key": api_key,
            "location": center_lnglat,
            "radius": radius,
            "types": poi_type,
            "offset": 20,  # 每页记录数（最大值20）
            "page": page,
            "extensions": "base"  # 返回基础信息
        }
        
        try:
            response = requests.get(base_url, params=params)
            result = response.json()
            
            if result["status"] == "1":
                current_pois = result["pois"]
                if not current_pois:
                    break  # 无更多数据
                pois.extend(current_pois)
                print(f"已获取第{page}页，共{len(current_pois)}条POI")
                page += 1
            else:
                print(f"请求失败: {result.get('info', '未知错误')}")
                break
                
        except Exception as e:
            print(f"请求异常: {str(e)}")
            break
    
    return pois

# 使用示例
if __name__ == "__main__":
    amap_key = "2ae214ee34f906b9a61aa11e9ad38a90"  # 替换为你的高德Key
    
    # 搜索豫园500米内的风景名胜POI，最多10页（200条）
    poi_list = search_poi_near_yuyuan(amap_key, radius=500, max_pages=10)
    
    # 打印前10条结果
    print("\n=== 周边POI结果（示例） ===")
    for idx, poi in enumerate(poi_list[:]):
        print(f"{idx + 1}. {poi['name']}")
        print(f"   地址：{poi['address']}")
        print(f"   距离：{poi['distance']}米")
        print(f"   类型：{poi['type']}\n")