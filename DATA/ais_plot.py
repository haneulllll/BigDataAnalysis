import pandas as pd
import folium
import random
from sqlalchemy import create_engine

# ✅ PostgreSQL 연결 설정
DB_USER = "postgres"
DB_PASSWORD = "ky76018500"
DB_HOST = "localhost"  # ✅ 외부 서버의 공인 IP 또는 도메인
DB_PORT = "5432"
DB_NAME = "ais_data"

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 🔹 특정 일시에 데이터가 있는 MMSI 중 하나를 랜덤 선택
def get_random_mmsi(date):
    query = f"""
    SELECT mmsi
    FROM ais
    WHERE timestamp = '{date} 00:00:00'
    LIMIT 100
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return None  # 해당 날짜에 데이터가 없는 경우
    
    return random.choice(df["mmsi"].tolist())  # 랜덤 MMSI 반환

# 🔹 특정 MMSI와 날짜의 AIS 데이터 불러오기
def fetch_ship_data(mmsi, date):
    query = f"""
    SELECT mmsi, timestamp, latitude, longitude
    FROM ais
    WHERE timestamp >='{date} 00:00:00'
    AND timestamp < '{date} 03:00:00'
    AND mmsi = '{mmsi}'
    ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, engine)
    return df

# 🔹 Folium을 이용한 지도 시각화
def plot_ship_map(df):
    if df.empty:
        print("❌ 해당 날짜에 데이터가 없습니다.")
        return
    # ❌ [데이터 정리] 위도(-90~90) & 경도(-180~180) 범위를 벗어난 데이터 삭제
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    # 위도, 경도 0인 이상치 제거
    df = df[df['latitude'] != 0]
    df = df[df['longitude'] != 0]

    # 지도 생성 (선박이 처음 있는 위치)
    start_lat = df['latitude'].iloc[0]
    start_lon = df['longitude'].iloc[0]
    map = folium.Map(location=[start_lat, start_lon], zoom_start=12)

    # 선박의 경로를 선으로 연결
    for i in range(len(df)-1):
        lat1, lon1 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
        lat2, lon2 = df.iloc[i+1]['latitude'], df.iloc[i+1]['longitude']
        folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="blue", weight=2.5, opacity=0.5).add_to(map)

    return map

# 🔹 실행 코드
if __name__ == "__main__":
    selected_date = "2020-03-10"  # 원하는 날짜 (YYYY-MM-DD 형식)

    # 랜덤 MMSI 선택
    selected_mmsi = get_random_mmsi(selected_date)

    if selected_mmsi:
        print(f"✅ 선택된 MMSI: {selected_mmsi}")
        df = fetch_ship_data(selected_mmsi, selected_date)  # 데이터 가져오기

        if not df.empty:
            ship_map = plot_ship_map(df)  # 지도 시각화
            output_file = f"ship_route_{selected_date}_{selected_mmsi}.html"
            ship_map.save(output_file)  # HTML 파일로 저장
            print(f"✅ {output_file} 파일을 브라우저에서 확인하세요.")
        else:
            print("❌ 해당 MMSI에 대한 데이터가 없습니다.")
    else:
        print("❌ 선택한 날짜에 데이터가 없습니다.")
