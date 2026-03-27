"""
Turn Detector — 한국어 형태소 분석 기반 발화 종료 감지 (Phase 1: 규칙 기반)

Kiwi 형태소 분석으로 마지막 어미 유형을 판별하고,
음향 피처(무음, blank, 에너지)와 결합하여 EOU 점수를 산출.
점수에 따라 silence 임계값을 동적으로 조정.

사용자 정의 규칙(turn_rules.json)을 Kiwi 분석보다 우선 적용.
"""
import json
import logging
import threading
import time
from pathlib import Path

from kiwipiepy import Kiwi

logger = logging.getLogger(__name__)

# 어미 유형별 기본 EOU 점수
_BASE_SCORES = {
    "final": 0.85,         # 종결어미 (EF)
    "interjection": 0.4,   # 감탄사 (IC) — 후속 발화 가능성 고려, silence 1.0초 대기
    "connective": 0.15,    # 연결어미 (EC), 접속부사 (MAJ)
    "transformative": 0.2, # 전성어미 (ETN, ETM)
    "incomplete": 0.15,    # 조사, 명사 단독
    "unknown": 0.3,        # 판별 불가
}

# Kiwi POS 태그 → 어미 유형 매핑
_TAG_TO_ENDING = {
    "EF": "final",          # 종결어미
    "EC": "connective",     # 연결어미
    "ETN": "transformative", # 명사형 전성어미
    "ETM": "transformative", # 관형형 전성어미
    "IC": "interjection",   # 감탄사
    "MAJ": "connective",    # 접속부사 (그런데, 그래서, 그리고...)
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
}

# 건너뛸 태그 (기호류)
_SKIP_TAGS = {"SS", "SP", "SE", "SO", "SW", "SH", "SL", "SN", "SF"}

# 유효한 ending 유형
_VALID_ENDINGS = set(_BASE_SCORES.keys())


class TurnDetector:
    """한국어 형태소 분석 기반 Turn Detection (Phase 1: 규칙)"""

    def __init__(self, log_dir=None, rules_path=None):
        self.kiwi = Kiwi()
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # 사용자 정의 규칙
        self.rules_path = Path(rules_path) if rules_path else None
        self._rules = []          # [{suffix, ending, note}, ...]
        self._rules_lock = threading.Lock()
        if self.rules_path and self.rules_path.exists():
            self._load_rules()

    # ── 사용자 규칙 관리 ──

    def _load_rules(self):
        """JSON 파일에서 규칙 로드."""
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rules = data.get("rules", [])
            # suffix 길이 내림차순 정렬 (긴 패턴 우선 매칭)
            rules.sort(key=lambda r: len(r.get("suffix", "")), reverse=True)
            with self._rules_lock:
                self._rules = rules
            logger.info(f"Turn rules loaded: {len(rules)} rules from {self.rules_path}")
        except Exception as e:
            logger.error(f"Failed to load turn rules: {e}")

    def _save_rules(self):
        """현재 규칙을 JSON 파일에 저장. 호출 전 lock이 잡혀있어야 함."""
        if not self.rules_path:
            return
        try:
            self.rules_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"rules": list(self._rules)}
            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save turn rules: {e}")

    def get_rules(self) -> list:
        """현재 규칙 목록 반환."""
        with self._rules_lock:
            return list(self._rules)

    def add_rule(self, suffix: str, ending: str, note: str = "") -> dict:
        """규칙 추가. 중복이면 업데이트."""
        suffix = suffix.strip()
        ending = ending.strip()
        if not suffix:
            return {"error": "suffix가 비어있습니다"}
        if ending not in _VALID_ENDINGS:
            return {"error": f"잘못된 ending: {ending}. 가능한 값: {sorted(_VALID_ENDINGS)}"}

        with self._rules_lock:
            # 기존 동일 suffix 규칙 업데이트
            for rule in self._rules:
                if rule["suffix"] == suffix:
                    rule["ending"] = ending
                    rule["note"] = note
                    self._rules.sort(key=lambda r: len(r.get("suffix", "")), reverse=True)
                    self._save_rules()
                    return {"ok": True, "action": "updated"}
            # 새 규칙 추가
            self._rules.append({"suffix": suffix, "ending": ending, "note": note})
            self._rules.sort(key=lambda r: len(r.get("suffix", "")), reverse=True)

        self._save_rules()
        return {"ok": True, "action": "added"}

    def delete_rule(self, suffix: str) -> dict:
        """규칙 삭제."""
        suffix = suffix.strip()
        with self._rules_lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r["suffix"] != suffix]
            if len(self._rules) == before:
                return {"error": f"규칙을 찾을 수 없습니다: '{suffix}'"}

        self._save_rules()
        return {"ok": True}

    def reload_rules(self):
        """파일에서 규칙 다시 로드."""
        if self.rules_path and self.rules_path.exists():
            self._load_rules()

    # ── 분류 로직 ──

    def _check_user_rules(self, text: str):
        """사용자 정의 suffix 규칙으로 분류 시도. 매칭 시 ending 반환, 아니면 None."""
        with self._rules_lock:
            for rule in self._rules:
                if text.endswith(rule["suffix"]):
                    return rule["ending"]
        return None

    def classify_ending(self, text: str) -> str:
        """텍스트의 어미 유형 분류.

        1. 사용자 정의 규칙 (suffix 매칭) — 최우선
        2. Kiwi 형태소 분석 — 폴백

        Returns:
            'final' | 'connective' | 'interjection' | 'transformative'
            | 'incomplete' | 'unknown'
        """
        text = text.strip()
        if not text:
            return "unknown"

        # 1. 사용자 정의 규칙 우선 적용
        user_result = self._check_user_rules(text)
        if user_result:
            return user_result

        # 2. Kiwi 형태소 분석
        tokens = self.kiwi.tokenize(text)
        if not tokens:
            return "unknown"

        # 연결어미 + 요(JX) 패턴 감지
        if len(tokens) >= 2:
            last = tokens[-1]
            prev = tokens[-2]
            if (last.tag == "JX" and last.form == "요"
                    and prev.tag in ("EC", "MAJ")):
                return "final"

        # 뒤에서부터 의미 있는 태그 탐색 (기호 건너뜀)
        for tok in reversed(tokens):
            tag = tok.tag
            if tag in _SKIP_TAGS:
                continue
            if tag in _TAG_TO_ENDING:
                return _TAG_TO_ENDING[tag]
            if tag in _INCOMPLETE_TAGS:
                return "incomplete"
            return "unknown"

        return "unknown"

    def compute_eou(self, text: str, silence_sec: float,
                    blank_count: int, energy: float) -> float:
        """EOU(End-of-Utterance) 점수 계산 (0~1)."""
        ending_type = self.classify_ending(text)
        base = _BASE_SCORES.get(ending_type, 0.3)

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
