# K-MarketAI: 한국 주식시장 예측 (미국시장 + 뉴스/매체/포럼 반영)

요청사항에 맞춰, 한국 주식시장이 미국시장 영향을 받는 점을 반영해 다음을 함께 분석하는 베이스라인을 제공합니다.

- 한국시장 가격 데이터 (기본: KOSPI `^KS11`)
- 미국시장 영향 지표 (`^GSPC`, `^IXIC`, `^DJI`, `^SOX`, `KRW=X`)
- 인터넷 뉴스/매체/포럼 관련 RSS 텍스트의 일별 감성 점수

## 파일
- `stock_predictor_kr.py`
  - 가격 데이터 수집 (Yahoo Finance)
  - 한국시장 기술지표 생성
  - 미국시장 파생 피처 생성 (한국장 반영을 위해 1일 shift)
  - 뉴스/매체/포럼 키워드 RSS 수집 및 감성 점수화(VADER)
  - 랜덤포레스트 분류로 다음 거래일 상승/하락 예측

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

옵션 예시:
```bash
python stock_predictor_kr.py --ticker ^KS11 --period 10y --test-size 0.2
```

## 해석 팁
- 출력의 `Top Feature Importances`로 어떤 미국지표/뉴스요인이 모델에 크게 작용했는지 확인할 수 있습니다.
- 본 코드는 연구용 베이스라인입니다. 실거래에는 추가 검증(워크포워드, 비용/슬리피지, 리스크 관리)이 필수입니다.
