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
