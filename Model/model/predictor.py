def predict_route(data):
    # 현재 위치, 속력, 방향, 목적지 등 처리
    # model = torch.load(...) 등으로 모델 불러오기
    # 예측 로직 수행 (단순화 예시)
    return {
        "eta": "3시간 20분",
        "recommended_sog": 14.3,
        "recommended_cog": 270,
        "route": [
            {"lat": 37.5, "lng": 126.9},
            {"lat": 35.0, "lng": 128.0},
        ]
    }
