# Turn Detection Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Kiwi 형태소 분석 기반 동적 silence 임계값으로 중간 끊김 방지 + 빠른 발화 종료 감지

**Architecture:** ASR partial 텍스트를 Kiwi로 형태소 분석하여 마지막 어미 유형(종결/연결/조사 등)을 판별하고, 무음·blank·에너지와 결합하여 EOU 점수를 계산. 점수에 따라 silence 임계값을 0.5초~2.5초로 동적 조정.

**Tech Stack:** kiwipiepy (형태소 분석), Python 3.10, FastAPI WebSocket, pytest

---

### Task 1: 한국어 어미 분류 모듈 — 테스트

**Files:**
- Create: `realtime_demo/tests/test_turn_detector.py`

**Step 1: Write failing tests for Korean ending classification**

```python
"""Turn Detector 단위 테스트"""
import pytest


class TestClassifyEnding:
    """한국어 어미 유형 분류 테스트"""

    def test_final_ending_ef(self):
        """종결어미(EF) → 'final'"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        assert td.classify_ending("네 알겠습니다") == "final"
        assert td.classify_ending("감사합니다") == "final"
        assert td.classify_ending("확인해 드릴게요") == "final"
        assert td.classify_ending("잠시만요") == "final"

    def test_connective_ending_ec(self):
        """연결어미(EC) → 'connective'"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        assert td.classify_ending("그래서 제가 말하는데") == "connective"
        assert td.classify_ending("주문을 하고") == "connective"
        assert td.classify_ending("배송이 되어서") == "connective"
        assert td.classify_ending("확인하시면") == "connective"

    def test_interjection_ic(self):
        """감탄사(IC) → 'interjection'"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        assert td.classify_ending("네") == "interjection"
        assert td.classify_ending("여보세요") == "interjection"

    def test_particle_incomplete(self):
        """조사로 끝남 → 'incomplete'"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        assert td.classify_ending("주문번호가") == "incomplete"
        assert td.classify_ending("주문하신 상품은") == "incomplete"

    def test_connective_adverb_maj(self):
        """접속부사(MAJ) 단독 → 'connective'"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        assert td.classify_ending("그런데") == "connective"
        assert td.classify_ending("그래서") == "connective"

    def test_noun_ending(self):
        """명사로 끝남 → 'incomplete'"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        assert td.classify_ending("주문번호") == "incomplete"

    def test_empty_text(self):
        """빈 텍스트 → 'unknown'"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        assert td.classify_ending("") == "unknown"


class TestComputeEou:
    """EOU 점수 계산 테스트"""

    def test_final_ending_high_score(self):
        """종결어미 → 높은 EOU 점수 (>= 0.7)"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        score = td.compute_eou("감사합니다", silence_sec=0.0,
                               blank_count=0, energy=0.01)
        assert score >= 0.7

    def test_connective_ending_low_score(self):
        """연결어미 → 낮은 EOU 점수 (<= 0.3)"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        score = td.compute_eou("주문을 하고", silence_sec=0.0,
                               blank_count=0, energy=0.01)
        assert score <= 0.3

    def test_silence_boosts_score(self):
        """무음이 길면 EOU 점수 증가"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        score_no_silence = td.compute_eou("주문을 하고", silence_sec=0.0,
                                          blank_count=0, energy=0.01)
        score_with_silence = td.compute_eou("주문을 하고", silence_sec=2.0,
                                            blank_count=5, energy=0.001)
        assert score_with_silence > score_no_silence

    def test_punctuation_boosts_score(self):
        """구두점 → 높은 EOU 점수"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        score = td.compute_eou("감사합니다.", silence_sec=0.0,
                               blank_count=0, energy=0.01)
        assert score >= 0.85


class TestGetSilenceThreshold:
    """동적 silence 임계값 테스트"""

    def test_high_eou_short_threshold(self):
        """높은 EOU → 짧은 silence 임계값"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        threshold = td.get_silence_threshold(0.9)
        assert threshold <= 0.6

    def test_low_eou_long_threshold(self):
        """낮은 EOU → 긴 silence 임계값"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        threshold = td.get_silence_threshold(0.1)
        assert threshold >= 2.0

    def test_medium_eou_medium_threshold(self):
        """중간 EOU → 중간 silence 임계값"""
        from realtime_demo.turn_detector import TurnDetector
        td = TurnDetector()
        threshold = td.get_silence_threshold(0.5)
        assert 0.8 <= threshold <= 1.5
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_turn_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'realtime_demo.turn_detector'`

---

### Task 2: TurnDetector 구현

**Files:**
- Create: `realtime_demo/turn_detector.py`
- Create: `realtime_demo/__init__.py` (빈 파일, 패키지화)
- Create: `realtime_demo/tests/__init__.py` (빈 파일)

**Step 1: Implement TurnDetector class**

```python
"""
Turn Detector — 한국어 형태소 분석 기반 발화 종료 감지 (Phase 1: 규칙 기반)

Kiwi 형태소 분석으로 마지막 어미 유형을 판별하고,
음향 피처(무음, blank, 에너지)와 결합하여 EOU 점수를 산출.
점수에 따라 silence 임계값을 동적으로 조정.
"""
import json
import logging
import time
from pathlib import Path

from kiwipiepy import Kiwi

logger = logging.getLogger(__name__)

# 어미 유형별 기본 EOU 점수
_BASE_SCORES = {
    "final": 0.85,        # 종결어미 (EF)
    "interjection": 0.9,  # 감탄사 (IC)
    "connective": 0.15,   # 연결어미 (EC), 접속부사 (MAJ)
    "transformative": 0.2, # 전성어미 (ETN, ETM)
    "incomplete": 0.15,   # 조사, 명사 단독
    "unknown": 0.3,       # 판별 불가
}

# Kiwi POS 태그 → 어미 유형 매핑
_TAG_TO_ENDING = {
    "EF": "final",         # 종결어미
    "EC": "connective",    # 연결어미
    "ETN": "transformative", # 명사형 전성어미
    "ETM": "transformative", # 관형형 전성어미
    "IC": "interjection",  # 감탄사
    "MAJ": "connective",   # 접속부사 (그런데, 그래서, 그리고...)
}

# incomplete로 분류할 태그 집합
_INCOMPLETE_TAGS = {
    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ",  # 격조사
    "JX",   # 보조사
    "JC",   # 접속조사
    "NNG", "NNP", "NNB", "NR", "NP",  # 명사류
    "XSV", "XSA", "XSN",  # 접미사 (파생)
    "VV", "VA", "VX", "VCP", "VCN",  # 용언 어간 (어미 없이 끝남)
    "EP",   # 선어말어미 (-시-, -겠-)
    "MM", "MAG",  # 관형사, 부사
    "SF",   # 마침표 등 (단독으로 올 때)
}


class TurnDetector:
    """한국어 형태소 분석 기반 Turn Detection (Phase 1: 규칙)"""

    def __init__(self, log_dir=None):
        self.kiwi = Kiwi()
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def classify_ending(self, text: str) -> str:
        """텍스트의 마지막 형태소 태그 기반 어미 유형 분류.

        Returns:
            'final' | 'connective' | 'interjection' | 'transformative'
            | 'incomplete' | 'unknown'
        """
        text = text.strip()
        if not text:
            return "unknown"

        tokens = self.kiwi.tokenize(text)
        if not tokens:
            return "unknown"

        # 뒤에서부터 의미 있는 태그 탐색 (공백/기호 건너뜀)
        for tok in reversed(tokens):
            tag = tok.tag
            # 공백, 기호 건너뜀
            if tag in ("SS", "SP", "SE", "SO", "SW", "SH", "SL", "SN"):
                continue
            if tag == "SF":  # 마침표/물음표 → 그 앞 형태소로 판단
                continue
            if tag in _TAG_TO_ENDING:
                return _TAG_TO_ENDING[tag]
            if tag in _INCOMPLETE_TAGS:
                return "incomplete"
            return "unknown"

        return "unknown"

    def compute_eou(self, text: str, silence_sec: float,
                    blank_count: int, energy: float) -> float:
        """EOU(End-of-Utterance) 점수 계산 (0~1).

        Args:
            text: 현재까지의 ASR partial 텍스트
            silence_sec: 마지막 음성 이후 무음 시간 (초)
            blank_count: 연속 blank 스텝 수
            energy: 현재 오디오 RMS 에너지
        """
        ending_type = self.classify_ending(text)
        base = _BASE_SCORES[ending_type]

        # 구두점 보너스
        stripped = text.rstrip()
        if stripped.endswith(('.', '?', '!')):
            base = max(base, 0.9)

        # 무음 시간 보정
        if silence_sec > 1.5:
            base = min(base + 0.3, 1.0)
        elif silence_sec > 0.8:
            base = min(base + 0.15, 1.0)

        # blank 스텝 보정
        if blank_count >= 5:
            base = min(base + 0.2, 1.0)
        elif blank_count >= 3:
            base = min(base + 0.1, 1.0)

        return round(base, 3)

    def get_silence_threshold(self, eou_score: float) -> float:
        """EOU 점수 → 동적 무음 임계값 (초)."""
        if eou_score >= 0.8:
            return 0.5
        elif eou_score >= 0.5:
            return 1.0
        elif eou_score >= 0.3:
            return 1.5
        else:
            return 2.5

    def log_decision(self, text: str, eou_score: float,
                     silence_sec: float, blank_count: int,
                     energy: float, endpoint: bool,
                     endpoint_reason: str = ""):
        """Phase 2 학습 데이터 자동 수집."""
        if not self.log_dir:
            return
        entry = {
            "ts": time.time(),
            "text": text,
            "ending": self.classify_ending(text),
            "eou": eou_score,
            "silence": round(silence_sec, 3),
            "blanks": blank_count,
            "energy": round(energy, 5),
            "endpoint": endpoint,
            "reason": endpoint_reason,
        }
        log_file = self.log_dir / "turn_decisions.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Step 2: Create __init__.py files**

Create empty `realtime_demo/__init__.py` and `realtime_demo/tests/__init__.py`.

**Step 3: Run tests to verify they pass**

Run: `cd /home/jonghooy/work/cachedSTT && python -m pytest realtime_demo/tests/test_turn_detector.py -v`
Expected: All 12 tests PASS

**Step 4: Commit**

```bash
git add realtime_demo/turn_detector.py realtime_demo/__init__.py \
        realtime_demo/tests/__init__.py realtime_demo/tests/test_turn_detector.py
git commit -m "feat: add TurnDetector with Korean morpheme-based EOU scoring"
```

---

### Task 3: server.py에 TurnDetector 통합

**Files:**
- Modify: `realtime_demo/server.py`

**Step 1: Import and initialize TurnDetector at module level**

At top of server.py, add import:
```python
from realtime_demo.turn_detector import TurnDetector
```

Add global:
```python
turn_detector = None
```

In `startup()`, after model warmup, add:
```python
# 6. Turn Detector 초기화
log_dir = Path(__file__).parent / "logs" / "turn_decisions"
turn_detector = TurnDetector(log_dir=log_dir)
logger.info("Turn Detector initialized (Phase 1: rule-based)")
```

**Step 2: Replace endpoint logic in websocket_endpoint**

Replace the current endpoint block (lines ~379-395) with:

```python
            # 끝점 판단: Turn Detection 기반 동적 임계값
            endpoint_reason = ""
            if session.is_speaking and buffer_seconds >= 0.5 and silence_seconds > 0:
                blank_count = session.blank_step_count
                eou_score = turn_detector.compute_eou(
                    text=session.last_text,
                    silence_sec=silence_seconds,
                    blank_count=blank_count,
                    energy=rms,
                )
                dynamic_threshold = turn_detector.get_silence_threshold(eou_score)

                if silence_seconds >= dynamic_threshold and blank_count >= 2:
                    endpoint_reason = f"turn(eou={eou_score:.2f},th={dynamic_threshold})"
                elif buffer_seconds >= MAX_BUFFER_SECONDS:
                    endpoint_reason = "buffer_max"
```

**Step 3: Add logging after endpoint decision**

After the final message is sent, add:
```python
                turn_detector.log_decision(
                    text=session.last_text,
                    eou_score=eou_score if 'eou_score' in dir() else 0,
                    silence_sec=silence_seconds,
                    blank_count=blank_count,
                    energy=rms,
                    endpoint=True,
                    endpoint_reason=endpoint_reason,
                )
```

**Step 4: Test server starts without errors**

Run:
```bash
cd /home/jonghooy/work/cachedSTT/realtime_demo
source /home/jonghooy/miniconda3/etc/profile.d/conda.sh && conda activate nemo-asr
python -c "from realtime_demo.turn_detector import TurnDetector; print('OK')"
```

**Step 5: Commit**

```bash
git add realtime_demo/server.py
git commit -m "feat: integrate TurnDetector into streaming server endpoint logic"
```

---

### Task 4: 서버 실행 + 통합 테스트

**Step 1: Kill existing server and restart**

```bash
lsof -ti:3000 | xargs kill -9 2>/dev/null; sleep 2
cd /home/jonghooy/work/cachedSTT/realtime_demo
CUDA_VISIBLE_DEVICES=0 python server.py
```

**Step 2: Verify Turn Detector is loaded**

Check server.log for: `Turn Detector initialized (Phase 1: rule-based)`

**Step 3: Test in browser**

Open http://localhost:3000 and test:
1. "감사합니다" → 빠른 Final (짧은 silence)
2. "그래서 제가 말씀드린 것처럼..." (pause) "...이 부분을 확인해주세요" → 중간에 안 끊기고 하나로 연결
3. "네" → 빠른 Final

**Step 4: Verify logs are being written**

```bash
cat realtime_demo/logs/turn_decisions/turn_decisions.jsonl
```

**Step 5: Commit final state**

```bash
git add -A
git commit -m "feat: Turn Detection Phase 1 complete - Korean morpheme-based dynamic endpointing"
```
