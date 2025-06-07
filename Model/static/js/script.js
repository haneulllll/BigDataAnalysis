// 지도 객체 생성 및 초기 뷰 설정 (서울 근처 좌표와 줌레벨 7)
let map = L.map('map').setView([37.5, 126.9], 7);

// OpenStreetMap 타일 레이어 추가 (지도 배경)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

/**
 * 예측 요청을 처리하는 함수
 * 사용자가 입력한 값들을 모아 서버로 POST 요청을 보낸 후,
 * 결과를 받아 지도에 경로를 표시하고, ETA 및 추천 SOG/COG를 출력함.
 */
function predict() {
    // 사용자 입력값 수집 및 실수형으로 변환
    const data = {
        lat: parseFloat(document.getElementById('lat').value),              // 현재 위도
        lng: parseFloat(document.getElementById('lng').value),              // 현재 경도
        sog: parseFloat(document.getElementById('sog').value),              // 현재 속력 (Speed Over Ground)
        cog: parseFloat(document.getElementById('cog').value),              // 현재 방향 (Course Over Ground)
        dest_lat: parseFloat(document.getElementById('dest-lat').value),    // 목적지 위도
        dest_lng: parseFloat(document.getElementById('dest-lng').value)     // 목적지 경도
    };

    // 예측 API로 POST 요청 전송
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }, // JSON 형식 명시
        body: JSON.stringify(data) // 사용자 입력값을 JSON 문자열로 변환하여 전송
    })
    .then(res => res.json()) // 응답을 JSON 객체로 변환
    .then(res => {
        // 예측 결과를 HTML 요소에 표시
        document.getElementById('eta').innerText = res.eta;                         // 예상 도착 시간
        document.getElementById('sog-result').innerText = res.recommended_sog;      // 추천 속력
        document.getElementById('cog-result').innerText = res.recommended_cog;      // 추천 방향

        // 서버로부터 받은 경로 데이터를 지도에 그리기
        const route = res.route; // 경로는 여러 좌표들의 배열
        let latlngs = route.map(p => [p.lat, p.lng]); // {lat, lng} 객체를 [lat, lng] 배열로 변환

        // 지도에 경로 선(polyline) 추가 (파란색)
        L.polyline(latlngs, {color: 'blue'}).addTo(map);

        // 지도 중심을 출발 지점으로 이동
        map.setView(latlngs[0], 7);
    });
}
