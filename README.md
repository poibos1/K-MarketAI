# K-MarketAI: 한국 주식시장(코스피) 방향 예측 베이스라인

이 프로젝트는 한국 주식시장(기본값: 코스피 지수 `^KS11`)의 **다음 거래일 상승/하락 방향**을 예측하는 간단한 베이스라인 프로그램입니다.

## 구성
- `stock_predictor_kr.py`
  - Yahoo Finance 데이터 다운로드
  - 기술지표 기반 피처 생성
  - 로지스틱 회귀 학습
  - 테스트 정확도 및 최신 예측 출력

## 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행
```bash
python stock_predictor_kr.py
```

옵션:
```bash
python stock_predictor_kr.py --ticker ^KS11 --period 10y --test-size 0.2
```

## 주의
- 본 코드는 교육/연구용 베이스라인입니다.
- 실제 투자에 사용하기 전, 추가 피처/검증/리스크 관리가 필요합니다.
