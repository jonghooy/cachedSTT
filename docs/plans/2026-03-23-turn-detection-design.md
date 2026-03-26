# Turn Detection 설계: 한국어 형태소 분석 + 경량 분류기

## 목표

1. **중간 끊김 방지**: 발화 중 짧은 쉼 구간에서 잘리는 문제 해결
2. **빠른 발화 종료 감지**: 화자가 말을 끝내면 최대한 빨리 Final 확정
3. **사용 시나리오**: 전화 상담/콜센터 (두 화자 번갈아 대화)
4. **점진적 개선**: 엣지케이스 발견 시 규칙 추가 or 학습 데이터 추가로 즉시 개선

## 아키텍처

```
ASR partial text ─→ Kiwi 형태소분석 ─→ 어미 유형 + 피처 추출
                                              │
        음향 피처 (silence, blank, energy) ────┤
                                              ▼
                                     EOU 분류기 (Phase 1: 규칙, Phase 2: ML)
                                              │
                                              ▼
                                     EOU 확률 (0~1)
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                        EOU > 0.8       0.3~0.8          EOU < 0.3
                     silence 0.5초    silence 1.2초    silence 2.5초
                      (빠른 종료)      (기본값)        (끊김 방지)
```

## Phase 1: 규칙 기반 (먼저 구현)

### 한국어 어미 분류 체계

Kiwi 형태소 분석 결과에서 **마지막 어미 태그**를 기반으로 3단계 분류:

| 분류 | Kiwi POS 태그 | 예시 | EOU 점수 |
|------|-------------|------|---------|
| **종결어미** (turn 완료) | EF | -요, -다, -까, -세요 | 0.8~0.95 |
| **연결어미** (계속 발화) | EC | -는데, -고, -어서, -면, -니까 | 0.1~0.3 |
| **전성어미/명사형** (미완성) | ETN, ETM | -는, -을, -기 | 0.2~0.4 |
| **조사/보조사** (미완성) | JX, JKS, JKO | -은, -는, -를, -에 | 0.1~0.2 |
| **체언 단독** (미완성) | NNG, NNP | 주문번호, 서울 | 0.1~0.2 |
| **감탄사/맞장구** (짧은 turn) | IC | 네, 아, 응 | 0.9 |

### 규칙 기반 EOU 계산

```python
def compute_eou_score(text, silence_sec, blank_count, energy):
    """규칙 기반 EOU(End-of-Utterance) 점수 계산"""
    morphemes = kiwi.tokenize(text)
    last_tag = get_last_meaningful_tag(morphemes)

    # 1. 어미 유형에 따른 기본 점수
    if last_tag == 'EF':           # 종결어미
        base = 0.85
    elif last_tag == 'IC':         # 감탄사
        base = 0.9
    elif last_tag == 'EC':         # 연결어미
        base = 0.15
    elif last_tag in ('ETN', 'ETM'):  # 전성어미
        base = 0.2
    elif last_tag in ('JX', 'JKS', 'JKO', 'JKB'):  # 조사
        base = 0.15
    else:                          # 체언/동사 어간 등
        base = 0.2

    # 2. 구두점 보너스
    if text.rstrip().endswith(('.', '?', '!')):
        base = max(base, 0.9)

    # 3. 무음 시간 보정
    if silence_sec > 1.5:
        base = min(base + 0.3, 1.0)
    elif silence_sec > 0.8:
        base = min(base + 0.15, 1.0)

    # 4. blank 스텝 보정
    if blank_count >= 5:
        base = min(base + 0.2, 1.0)
    elif blank_count >= 3:
        base = min(base + 0.1, 1.0)

    return base
```

### 동적 silence 임계값 매핑

```python
def get_silence_threshold(eou_score):
    """EOU 점수에 따른 무음 임계값 (초)"""
    if eou_score >= 0.8:
        return 0.5    # 종결어미 + 무음 → 빠른 종료
    elif eou_score >= 0.5:
        return 1.0    # 중간 확신
    elif eou_score >= 0.3:
        return 1.5    # 낮은 확신 → 좀 더 기다림
    else:
        return 2.5    # 연결어미 → 오래 기다림 (끊김 방지)
```

## Phase 2: ML 분류기 (데이터 축적 후)

Phase 1에서 로깅한 데이터를 기반으로 학습:

### 피처 벡터 (9차원)

| # | 피처 | 타입 | 설명 |
|---|------|------|------|
| 1 | ending_type | categorical (0-5) | 종결/연결/전성/조사/체언/감탄 |
| 2 | has_punctuation | binary | 구두점 여부 |
| 3 | silence_seconds | float | 무음 지속 시간 |
| 4 | blank_step_count | int | 연속 blank 스텝 수 |
| 5 | rms_energy | float | 현재 오디오 에너지 |
| 6 | utterance_duration | float | 현재 발화 길이 (초) |
| 7 | token_count | int | 디코딩된 토큰 수 |
| 8 | text_length | int | 텍스트 글자 수 |
| 9 | last_energy_slope | float | 최근 에너지 추세 (하강=발화 끝) |

### 학습 데이터 수집

Phase 1 운영 중 자동 로깅:
```json
{
  "text": "네 알겠습니다",
  "features": [0, 1, 0.8, 5, 0.002, 2.1, 12, 7, -0.3],
  "actual_eou": true,
  "timestamp": "2026-03-24T10:30:00"
}
```

- `actual_eou`: 운영자가 나중에 라벨링 (또는 다음 발화 시작까지 시간으로 자동 추정)
- 목표: 500건 이상 모이면 XGBoost/LightGBM 학습

### 분류기 선택

- **Phase 2a**: scikit-learn LogisticRegression (데이터 적을 때)
- **Phase 2b**: XGBoost/LightGBM (데이터 1000건+)
- 추론 <1ms (CPU), 모델 크기 <1MB

## 서버 통합 설계

### TurnDetector 클래스

```python
class TurnDetector:
    def __init__(self):
        self.kiwi = Kiwi()
        self.classifier = None  # Phase 2에서 로드
        self.log_buffer = []    # 학습 데이터 수집용

    def predict_eou(self, text, silence_sec, blank_count, energy,
                    utterance_duration) -> float:
        """EOU 확률 반환 (0~1)"""
        # Phase 1: 규칙 기반
        if self.classifier is None:
            return self._rule_based_eou(text, silence_sec, blank_count, energy)
        # Phase 2: ML 분류기
        features = self._extract_features(text, silence_sec, blank_count,
                                          energy, utterance_duration)
        return self.classifier.predict_proba(features)[0][1]

    def get_silence_threshold(self, eou_score) -> float:
        """EOU 점수 → 동적 무음 임계값"""
        ...

    def log_decision(self, text, features, eou_score, actual_endpoint):
        """학습 데이터 자동 수집"""
        ...
```

### server.py 수정 포인트

현재 endpoint 판단 로직:
```python
# 현재 (고정 임계값)
if silence_seconds >= SILENCE_DURATION and blank_count >= 5:
    endpoint_reason = "silence+blank"
```

변경 후 (동적 임계값):
```python
# Turn Detection 적용
eou_score = turn_detector.predict_eou(
    text=session.last_text,
    silence_sec=silence_seconds,
    blank_count=blank_count,
    energy=rms,
    utterance_duration=buffer_seconds
)
dynamic_threshold = turn_detector.get_silence_threshold(eou_score)

if silence_seconds >= dynamic_threshold and blank_count >= 2:
    endpoint_reason = f"turn_detect(eou={eou_score:.2f})"
```

## 파일 구조

```
realtime_demo/
├── server.py                    # 기존 서버 (TurnDetector 통합)
├── turn_detector.py             # TurnDetector 클래스 (NEW)
├── turn_detection_rules.py      # 한국어 어미 분류 규칙 (NEW)
├── static/index.html            # 프론트엔드
└── models/                      # Phase 2 학습된 분류기 저장 (NEW)
    └── eou_classifier.pkl
```

## 구현 순서

1. **kiwipiepy 설치** + 형태소 분석 동작 확인
2. **turn_detection_rules.py** — 한국어 어미 분류 규칙 구현 + 단위 테스트
3. **turn_detector.py** — TurnDetector 클래스 구현 (Phase 1 규칙 기반)
4. **server.py 통합** — 기존 endpoint 로직을 TurnDetector로 교체
5. **로깅 추가** — Phase 2 학습 데이터 자동 수집
6. **통합 테스트** — 실제 음성으로 동작 확인

## 의존성

- `kiwipiepy` (pip install) — 한국어 형태소 분석, CPU, ~50MB
- `scikit-learn` (이미 설치됨) — Phase 2 분류기
- `xgboost` (Phase 2에서 설치) — 고급 분류기

## 향후 확장 (접근법 C 연계)

Phase 3에서 FastConformer 인코더 임베딩을 TurnDetector의 추가 피처로 결합:
- 인코더 출력 1024차원 → Linear(1024→64) 압축
- 기존 9차원 피처 + 64차원 임베딩 = 73차원 입력
- 음향(피치/억양) + 텍스트(어미) + 모델(blank) 신호 종합
