# Flask 관련 모듈 불러오기
from flask import Flask, render_template, request, jsonify

# Flask 애플리케이션 인스턴스 생성
app = Flask(__name__)

# 임시 경로 예측 함수
def predict_route(current_lat, current_lon, sog, cog, dest_lat, dest_lon):
    # 선박 경로를 단순한 직선 경로로 구성
    route = [
        [current_lat, current_lon],  # 시작 위치
        [(current_lat + dest_lat)/2, (current_lon + dest_lon)/2],  # 중간 지점
        [dest_lat, dest_lon]  # 목적지
    ]
    # 임의의 결과 데이터 생성
    eta = "3시간 45분"  # 예상 도착 시간
    recommended_sog = sog + 1.0  # 추천 속력 (SOG)
    recommended_cog = (cog + 10) % 360  # 추천 방향 (COG), 360도 넘지 않도록 보정
    return {
        "route": route,
        "eta": eta,
        "recommended_sog": recommended_sog,
        "recommended_cog": recommended_cog
    }

# 메인 페이지 렌더링 (index.html을 클라이언트에 전달)
@app.route('/')
def index():
    return render_template('index.html')

# 경로 예측 요청을 처리하는 API 엔드포인트
@app.route('/api/predict', methods=['POST'])
def api_predict():
    # 클라이언트로부터 JSON 데이터 받기
    data = request.json
    # 데이터 파싱 및 형 변환
    current_lat = float(data['current_lat'])
    current_lon = float(data['current_lon'])
    sog = float(data['sog'])
    cog = float(data['cog'])
    dest_lat = float(data['dest_lat'])
    dest_lon = float(data['dest_lon'])

    # 경로 예측 함수 호출
    result = predict_route(current_lat, current_lon, sog, cog, dest_lat, dest_lon)
    # JSON 형식으로 결과 반환
    return jsonify(result)

# 애플리케이션 실행 (디버그 모드)
if __name__ == '__main__':
    app.run(debug=True)
