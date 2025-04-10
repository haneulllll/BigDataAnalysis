import folium
import pandas as pd
import random

# CSV 파일 읽기
csv_file_path = 'Dynamic_20200310.csv' 
df = pd.read_csv(csv_file_path, skiprows=2, encoding='cp949')

# 랜덤으로 선박 하나의 MMSI 추출
random_mmsi = random.choice(df['mmsi'].unique())

# 특정 MMSI의 데이터 필터링
ship_data = df[df['mmsi'] == random_mmsi]

########################## 기초 전처리

# [데이터 정리] 위도(-90~90) & 경도(-180~180) 범위를 벗어난 데이터 삭제
ship_data = ship_data[(ship_data["latitude"].between(-90, 90)) & (ship_data["longitude"].between(-180, 180))]

# 위도, 경도 0인 이상치 제거
ship_data  = ship_data [ship_data ['latitude'] != 0]
ship_data  = ship_data [ship_data ['longitude'] != 0]

##########################

# 지도 생성 (선박이 처음 있는 위치)
start_lat = ship_data['latitude'].iloc[0]
start_lon = ship_data['longitude'].iloc[0]
map = folium.Map(location=[start_lat, start_lon], zoom_start=12)

# 선박의 경로를 선으로 연결
for i in range(len(ship_data)-1):
    lat1, lon1 = ship_data.iloc[i]['latitude'], ship_data.iloc[i]['longitude']
    lat2, lon2 = ship_data.iloc[i+1]['latitude'], ship_data.iloc[i+1]['longitude']
    folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="blue", weight=2.5, opacity=0.5).add_to(map)

# 지도 저장
output_map_path = 'ship_route_map2.html'
map.save(output_map_path)

print(f"선박 경로 지도는 {output_map_path}로 저장되었습니다.")
