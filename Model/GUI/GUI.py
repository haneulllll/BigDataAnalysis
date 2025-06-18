import sys, os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt6.QtCore import QTimer
import folium
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox, QComboBox, QLabel, QSizePolicy
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
import threading
from sklearn.preprocessing import MinMaxScaler
from math import atan2, sqrt, degrees, radians, cos, sin, asin
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json

def haversine(lat1, lon1, lat2, lon2):
    # 지구 반지름 (km)
    R = 6371.0
    # 라디안 변환
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    return R * c  # km 단위 거리
class AISPreprocessor:
    # data_dir: 경로 데이터 파일(csv)이 존재하는 폴더 이름 
    # input_seq_len: 참조할 이전 과거 정보(default: 20분)
    # output_seq_len: 예측할 미래 정보(default: 30초)
    def __init__(self, data_dir, input_seq_len=20, output_seq_len=1):
        self.data_dir = data_dir
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        # 수동 설정된 범위로 MinMaxScaler 초기화
        # 정규화 범위 설정
        lat_range = (33.0, 38.0)
        lon_range = (124.0, 132.0)
        sog_range = (0.0, 100.0)
        cog_range = (0.0, 360.0)
        
        # MinMaxScaler 수동 설정
        self.scaler = MinMaxScaler()
        self.scaler.min_ = np.array([
            -lat_range[0] / (lat_range[1] - lat_range[0]),
            -lon_range[0] / (lon_range[1] - lon_range[0]),
            -sog_range[0] / (sog_range[1] - sog_range[0]),
            -cog_range[0] / (cog_range[1] - cog_range[0])
        ])
        self.scaler.scale_ = np.array([
            1 / (lat_range[1] - lat_range[0]),
            1 / (lon_range[1] - lon_range[0]),
            1 / (sog_range[1] - sog_range[0]),
            1 / (cog_range[1] - cog_range[0])
        ])
        self.scaler.feature_names_in_ = np.array(['위도', '경도', 'SOG', 'COG'])


    def load_and_preprocess(self):
        input_seqs = []
        output_seqs = []
        count = 1
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                print(f"---------- {count}번째 파일 진행 중 ----------")
                count += 1
                df = pd.read_csv(os.path.join(self.data_dir, file), encoding='cp949')
                df = self._preprocess_single_file(df)
                in_seqs, out_seqs = self._extract_sequences(df)
                input_seqs.extend(in_seqs)
                output_seqs.extend(out_seqs)

        return np.array(input_seqs), np.array(output_seqs)

    def _preprocess_single_file(self, df):
        df = df[['일시', '위도', '경도', 'SOG', 'COG']].copy()
        df['일시'] = pd.to_datetime(df['일시'])
        # 일시 기준 데이터 sorting
        df = df.sort_values('일시')
        # NA 데이터 drop
        df = df.dropna()
        
        # 각 데이터를, 일시 기준 30초 별 보간법을 적용해 transform 적용 ------------------------------------
        # 평균을 적용하는 데이터는 위도, 경도, sog, cog 데이터 
        df = df.set_index('일시').resample('30s').mean().interpolate()
        df = df.reset_index()
        # -----------------------------------------------------------------------------------------------
        
        # 목적지 좌표: 마지막 위치
        dest_lat = df['위도'].iloc[-1]
        dest_lon = df['경도'].iloc[-1]
        
        # feature engineering을 위한 feature 저장 
        df['dest_lat'] = dest_lat
        df['dest_lon'] = dest_lon
        return df

    def _extract_sequences(self, df):
        input_seqs = []
        output_seqs = []
    
        total_len = self.input_seq_len + self.output_seq_len
        for i in range(len(df) - total_len):
            input_window = df.iloc[i:i+self.input_seq_len]
            output_window = df.iloc[i+self.input_seq_len:i+total_len]
    
            # 입력: 위도, 경도, SOG, COG (정규화)
            input_scaled = self.scaler.transform(input_window[['위도', '경도', 'SOG', 'COG']])
            
             # 목적지 좌표 가져오기 (하나만, 파일 마지막 지점 기준)
            dest_lat = input_window['dest_lat'].iloc[0]
            dest_lon = input_window['dest_lon'].iloc[0]
    
            #  Δlat, Δlon 계산 (목적지 - 현재 위치) 계산 - 이 값들은 정규화 진행 x
            delta_lat = dest_lat - input_window['위도']
            delta_lon = dest_lon - input_window['경도']
            
            # 목적지까지 거리 계산 
            distance = np.sqrt(delta_lat ** 2 + delta_lon ** 2)

            # 입력 최종: [정규화된 위도, 경도, SOG, COG, distance]
            delta_coords = np.stack([distance], axis=1)
            input_seq = np.hstack([input_scaled, delta_coords])
            
            # 출력: 위도, 경도, SOG, COG (정규화)
            output_seq = self.scaler.transform(output_window[['위도', '경도', 'SOG', 'COG']])
    
            input_seqs.append(input_seq)
            output_seqs.append(output_seq)
    
        return input_seqs, output_seqs
    
# =======================
# 👇 콤보 박스 리스트, 시나리오 좌표, 초기 시퀀스 정의
# =======================
start_list = ["인천항", "동해항", "여수항"]
end_list = ["제주항", "포항항", "울산항"]

scenarios = {
    "인천항->제주항": ((37.4535, 126.6056), (33.5176, 126.5186)),
    "동해항->포항항": ((37.5474, 129.1164), (36.0322, 129.3650)),
    "여수항->울산항": ((34.7365, 127.7456), (35.5066, 129.3735)),
}
seq_names = [
    'rou/ij_route_1_202002.csv',
    'rou/dp_route_1_202002.csv',
    'rou/yu_route_1_202003.csv'
]

# MinMaxScaler 수동 설정 ----------------------------------------------------------
# 정규화 범위 설정
lat_range = (33.0, 38.0)
lon_range = (124.0, 132.0)
sog_range = (0.0, 100.0)
cog_range = (0.0, 360.0)

input_scaler = MinMaxScaler()
input_scaler.min_ = np.array([
    -lat_range[0] / (lat_range[1] - lat_range[0]),
    -lon_range[0] / (lon_range[1] - lon_range[0]),
    -sog_range[0] / (sog_range[1] - sog_range[0]),
    -cog_range[0] / (cog_range[1] - cog_range[0])
])
input_scaler.scale_ = np.array([
    1 / (lat_range[1] - lat_range[0]),
    1 / (lon_range[1] - lon_range[0]),
    1 / (sog_range[1] - sog_range[0]),
    1 / (cog_range[1] - cog_range[0])
])
input_scaler.feature_names_in_ = np.array(['위도', '경도', 'SOG', 'COG'])
# --------------------------------------------------------------------------------

# =======================
# 👇 지도 생성 함수
# =======================
def create_live_map(coord_list):
    m = folium.Map(location=[35.5, 128.0], zoom_start=7)
    if coord_list:
        folium.Marker(coord_list[0], tooltip="출발지", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(coord_list[-1], tooltip="현재 위치", icon=folium.Icon(color="red")).add_to(m)
        folium.PolyLine(coord_list, color="blue", weight=2.5, opacity=1).add_to(m)
    m.save("map.html")


# =======================
# 👇 자가회귀 함수
# =======================

class AutoRegressivePredictor:
    def __init__(self, model, initial_seq, dest_lat, dest_lon, scaler, map_view,
                 max_steps=2400, distance_threshold=0.15, num_mc_samples=30, interval_ms=1000,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        
        self.model = model.to(device)
        self.model.train()  # Dropout 활성화
        self.input_seq = initial_seq.clone()
        self.dest_lat = dest_lat
        self.dest_lon = dest_lon
        self.scaler = scaler
        self.map_view = map_view
        self.max_steps = max_steps
        self.distance_threshold = distance_threshold
        self.num_mc_samples = num_mc_samples
        self.device = device
        self.step = 0
        self.route_coords = []
        self.dist_list = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.step_predict)
        self.timer.start(interval_ms)

        # 실제 Vessel의 위치를 업데이트하는 카운터 -> 5초마다 한 step 이동한다고 가정
        self.timer2 = QTimer()
        self.timer2.timeout.connect(self.update_vessel)
        self.timer2.start(4*interval_ms)
        self.update_count = 0

    def update_vessel(self):
        simulate_vessel_route(self.route_coords[self.update_count], self.map_view)
        self.update_count += 1
        if(self.update_count >= len(self.route_coords)):
            # 도착 알림 띄우기
            msg = QMessageBox(self.map_view)
            msg.setWindowTitle("선박 도착 알림")
            msg.setText(f"✅ 선박이 목적지까지 도착 완료했습니다!")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.exec()

            self.timer2.stop()
            return
        
    def step_predict(self):
        if self.step >= self.max_steps:
            print("⚠️ 최대 step 도달. 예측 종료.")
            self.timer.stop()
            return

        input_tensor = self.input_seq.to(self.device)
        preds = []

        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                pred = self.model(input_tensor).squeeze(0).cpu().numpy()
                preds.append(pred)

        pred = np.mean(preds, axis=0)
        pred_denorm = self.scaler.inverse_transform(pred.reshape(1, -1))[0][:4]
        pred_lat, pred_lon = pred_denorm[:2]

        self.route_coords.append((pred_lat, pred_lon))
        self.route_coords = [(float(lat), float(lon)) for lat, lon in self.route_coords]
        simulate_autoregressive_route(self.route_coords, self.map_view)
        if(self.update_count == 0):
            simulate_vessel_route(self.route_coords[0], self.map_view)
            self.update_count += 1
        
        dist = haversine(pred_lat, pred_lon, self.map_view.original_route[self.step][0],
                         self.map_view.original_route[self.step][1])
        self.dist_list.append(dist)
        update_dist(self.dist_list, self.map_view)
        # 경로 출력 로그
        if self.step % 20 == 0:
            print(f"[Step {self.step+1}] Pred: ({pred_lat:.5f}, {pred_lon:.5f}) | "
                  f"Target: ({self.dest_lat:.5f}, {self.dest_lon:.5f}) | "
                  f"ΔLat: {abs(pred_lat - self.dest_lat):.5f}, ΔLon: {abs(pred_lon - self.dest_lon):.5f}")

        # 목적지 도달 판정
        if (abs(pred_lat - self.dest_lat) < self.distance_threshold) and (abs(pred_lon - self.dest_lon)) < self.distance_threshold:
            print(f"✅ 목적지 도달 - Step: {self.step + 1}, {int(self.step/120)}시간 {int((self.step%120)/2)}분 소요")
            
            # 도착 알림 띄우기
            msg = QMessageBox(self.map_view)
            msg.setWindowTitle("도착 알림")
            msg.setText(f"✅ 경로 생성 완료했습니다!\n예상 소요 시간: {int(self.step/120)}시간 {int((self.step%120)/2)}분")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.exec()

            self.timer.stop()
            self.timer2.setInterval(100)
            return

        # 다음 입력 시퀀스 준비
        delta_lat = self.dest_lat - pred_lat
        delta_lon = self.dest_lon - pred_lon
        distance = np.sqrt(delta_lat ** 2 + delta_lon ** 2)

        pred_norm = pred.reshape(1, -1)[0]
        pred_norm = np.concatenate([pred_norm, [distance]])
        next_input = pred_norm

        input_seq_np = self.input_seq.squeeze(0).numpy()
        input_seq_np = np.vstack([input_seq_np[1:], next_input])
        self.input_seq = torch.from_numpy(input_seq_np).unsqueeze(0).float()

        self.step += 1
    
    def stop_prediction(self):
        self.timer.stop()
        print("예측 중단!")


# =======================
# 👇 예측 경로 시각화 함수
# =======================
def simulate_autoregressive_route(route_coords, map_view):
    map_view.draw_route(route_coords)  # 현재 누적 경로 그리기

def simulate_vessel_route(route_coords, map_view):
    map_view.draw_vessel(route_coords)

def update_dist(list, map_view):
    avg_dist = sum(list) / len(list)
    percent = round((avg_dist / 100) * 100, 2) # 100km 이탈시 100% 이탈로 간주

    map_view.draw_dist(percent)
# =======================
# 👇 메인 윈도우 클래스
# =======================

class ShipRoutePredictor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("선박 경로 예측 GUI")
        self.resize(1000, 700)

        main_layout = QVBoxLayout()

        top_input_layout = QVBoxLayout()

        # 출발지 콤보박스
        self.start_label = QLabel("출발지 선택:")
        self.start_combo = QComboBox()
        self.start_combo.addItems(start_list)
        
        # 목적지 콤보박스
        self.end_label = QLabel("목적지 선택:")
        self.end_combo = QComboBox()
        self.end_combo.addItems(end_list)
        
        # 실행 버튼
        self.run_button = QPushButton("예측 경로 탐색")
        self.run_button.clicked.connect(self.handle_scenario_selection)

        # 이탈률
        self.deviation_label = QLabel("평균 이탈률: -")
        self.deviation_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 5px;")

        # 위젯들 레이아웃에 추가
        top_input_layout.addWidget(self.start_label)
        top_input_layout.addWidget(self.start_combo)
        top_input_layout.addWidget(self.end_label)
        top_input_layout.addWidget(self.end_combo)
        top_input_layout.addWidget(self.run_button)
        top_input_layout.addWidget(self.deviation_label)

        # 상단 입력부를 고정 크기로 유지
        input_container = QWidget()
        input_container.setLayout(top_input_layout)
        input_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(input_container)

        self.map_view = QWebEngineView()
        self.map_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.map_view)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 예측 중단 버튼 생성
        stop_button = QPushButton("예측 중단")
        stop_button.clicked.connect(self.stop_prediction)
        main_layout.addWidget(stop_button)

        # map.html 파일 경로
        self.map_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "map.html")

        # 로컬 HTTP 서버 시작 (스레드로 백그라운드 실행)
        self.start_http_server()

        # QWebEngineView에 HTTP URL로 로드
        self.map_view.load(QUrl("http://localhost:8000/map.html"))

    def stop_prediction(self):
        if hasattr(self, 'predictor'):
            self.predictor.stop_prediction()
        else:
            print("⚠️ 현재 실행 중인 예측이 없습니다.")

    def start_http_server(self):
        """현재 디렉토리에서 HTTP 서버를 8000 포트로 실행"""
        os.chdir(os.path.dirname(self.map_file))

        handler = SimpleHTTPRequestHandler
        self.httpd = HTTPServer(("localhost", 8000), handler)

        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()

    def closeEvent(self, event):
        # 프로그램 종료 시 HTTP 서버도 종료
        self.httpd.shutdown()
        super().closeEvent(event)

    def draw_dist(self, deviation_percent):
        self.deviation_label.setText(f"평균 이탈률: {deviation_percent:.2f}%")

        # 색상 시각화 옵션 (선택)
        if deviation_percent < 5:
            color = "green"
        elif deviation_percent < 20:
            color = "orange"
        else:
            color = "red"
        self.deviation_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px; margin: 5px;")
    
    def draw_route(self, route_coords):
        if not route_coords:
            return
        coords_js = [[lat, lon] for lat, lon in route_coords]
        js_code = f"updateRoute({coords_js})"
        self.map_view.page().runJavaScript(js_code)

    def draw_vessel(self, route_coords):
        if not route_coords:
            return
        coords_js = [list(route_coords)]
        js_code = f"updateVessel({coords_js})"
        self.map_view.page().runJavaScript(js_code)
    
    def frequency_route(self, route):
        coords_js = [[float(lat), float(lon)] for lat, lon in route]
        js_code = f"updatefreRoute({coords_js})"
        self.map_view.page().runJavaScript(js_code)

    def handle_scenario_selection(self):
        start_idx = self.start_combo.currentText()
        end_idx = self.end_combo.currentText()
        if(start_idx == "인천항"):
            scenario_name = "인천항->제주항"
        elif(start_idx == "동해항"):
            scenario_name = "동해항->포항항"
        else:
            scenario_name = "여수항->울산항"

        if(end_idx == "제주항"):
            scenario_name_end = "인천항->제주항"
        elif(end_idx == "포항항"):
            scenario_name_end = "동해항->포항항"
        else:
            scenario_name_end = "여수항->울산항"
        
        start = scenarios[scenario_name][0]
        end = scenarios[scenario_name_end][1]

        # 출발항 기준으로 초기 CSV 시퀀스 결정
        if(scenario_name == "인천항->제주항"):
            df = pd.read_csv(seq_names[0], encoding='cp949', parse_dates=['일시'])
        elif(scenario_name == "동해항->포항항"):
            df = pd.read_csv(seq_names[1], encoding='cp949', parse_dates=['일시'])
        else:
            df = pd.read_csv(seq_names[2], encoding='cp949', parse_dates=['일시'])
        
        pre = AISPreprocessor('rou/', 40, 1)
        df = pre._preprocess_single_file(df)
        initial_seq = df.iloc[:40]

        self.frequency_route(df[['위도', '경도']].values)
        self.original_route = df[['위도', '경도']].iloc[40:].values
        self.original_route = [[float(lat), float(lon)] for lat, lon in self.original_route]

        # 입력 시퀀스 생성 - 도착항 기준으로 DISTANCE 계산
        test_input_seq = []
        for _, row in initial_seq.iterrows():
            # 1. 정규화된 위도, 경도, SOG, COG
            scaled = input_scaler.transform([[row['위도'], row['경도'], row['SOG'], row['COG']]])[0]
            # 2. Δlat, Δlon 계산
            delta_lat = end[0] - row['위도']
            delta_lon = end[1] - row['경도']
            
            # 3. distance 계산
            distance = sqrt(delta_lat ** 2 + delta_lon ** 2)
            # 4. 최종 입력 벡터 구성 (5차원)
            input_row = list(scaled) + [distance]
            test_input_seq.append(input_row)

        initial_seq = torch.tensor([test_input_seq], dtype=torch.float32)  

        scaler = input_scaler 

        # ✅ 실제 학습된 모델로 교체
        model = torch.load("route_predictor.pth", weights_only=False, map_location=torch.device('cpu'))
        max_steps = 4800

        self.predictor = AutoRegressivePredictor(
            model=model, 
            initial_seq=initial_seq, 
            dest_lat=end[0], 
            dest_lon=end[1], 
            scaler=scaler, 
            map_view=self, 
            max_steps=max_steps, 
            interval_ms=1000
        )


# =======================
# 👇 Dummy 클래스들 (교체 필요)
# =======================
def inverse_transform(preds, scaler):
    preds_unscaled = preds.copy()  # 예측된 값을 복사

    # 위도, 경도, SOG, COG를 역정규화
    preds_unscaled[:, :4] = scaler.inverse_transform(preds_unscaled[:, :4])  # 역정규화
    return preds_unscaled

## 위치 정보 전달을 위한 정적 포지셔널 인코딩
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerPredictor(nn.Module):
    def __init__(self, input_size=5, output_size=4, d_model=256, nhead=16, num_layers=5, dim_feedforward=1024, dropout=0.2, use_attention_pool=True):
        super(TransformerPredictor, self).__init__()
        # Attention pooling ------------------------------------------------------------------
        self.use_attention_pool = use_attention_pool
        if self.use_attention_pool:
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.Tanh(),
                nn.Linear(128, 1)  # 각 time step에 대한 score 출력
            )
        # --------------------------------------------------------------------------------------
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model) # 포지셔널 인코딩을 통해 순서 정보를 추가
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, # input의 특징들
            nhead=nhead, # 멀티 헤드 어텐션 헤드 수
            dim_feedforward=dim_feedforward, # FFN 차원 수, 기본 2048
            dropout=dropout,
            batch_first=True,
            activation="gelu", # default="relu"
        )
        # 인코더
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # MLP 기반 디코더
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        # Attention pooling 사용 ---------------------------------------------
        if self.use_attention_pool:
            # Attention score 계산
            attn_weights = self.attn_pool(x)  # (batch_size, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)  # normalize
            x_last = (attn_weights * x).sum(dim=1)  # 가중 평균
        else:
            # 기본 평균 풀링 사용
            x_last = x[:, -40:, :].mean(dim=1)
        out = self.decoder(x_last)  # (batch_size, output_size)
        # ---------------------------------------------------------------------
        # 차원을 맞추기 위해 seq_len=1 축을 다시 추가
        out = out.unsqueeze(1)  # (batch_size, 1, output_size)
        return out


# =======================
# 👇 실행부
# =======================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ShipRoutePredictor()
    window.show()
    sys.exit(app.exec())