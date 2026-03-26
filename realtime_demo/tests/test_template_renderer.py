"""Test template renderer."""
from realtime_demo.dialogue.template_renderer import render_template


class TestRenderTemplate:
    def _ctx(self, **kwargs):
        return {"slots": kwargs.get("slots", {}), "variables": kwargs.get("variables", {})}

    def test_simple_slot(self):
        result = render_template("카드번호 {{slots.card_number}}", self._ctx(slots={"card_number": "1234-5678"}))
        assert result == "카드번호 1234-5678"

    def test_multiple_slots(self):
        ctx = self._ctx(slots={"card_number": "1234", "loss_type": "분실"})
        result = render_template("{{slots.card_number}}, {{slots.loss_type}}", ctx)
        assert result == "1234, 분실"

    def test_variable(self):
        result = render_template("결과: {{variables.block_result}}", self._ctx(variables={"block_result": "OK"}))
        assert result == "결과: OK"

    def test_missing_value_empty_string(self):
        result = render_template("{{slots.missing}}", self._ctx())
        assert result == ""

    def test_no_template(self):
        result = render_template("안녕하세요", self._ctx())
        assert result == "안녕하세요"

    def test_render_dict(self):
        """Render templates inside a dict (for API body)."""
        from realtime_demo.dialogue.template_renderer import render_dict
        body = {"card_no": "{{slots.card_number}}", "type": "{{slots.loss_type}}"}
        ctx = self._ctx(slots={"card_number": "1234", "loss_type": "분실"})
        result = render_dict(body, ctx)
        assert result == {"card_no": "1234", "type": "분실"}
