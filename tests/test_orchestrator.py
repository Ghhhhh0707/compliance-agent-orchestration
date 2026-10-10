"""智能体编排器自动化测试。"""

import sys
import unittest
from pathlib import Path

# 让测试可以找到 src 目录中的 Python 文件
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from orchestrator import Orchestrator, recognize_intent
from schemas import Answer, Evidence


class IntentRouterTests(unittest.TestCase):
    """意图路由器测试。"""

    def test_recognize_risk_diagnosis(self):
        plan = recognize_intent(
            "我们公司不签劳动合同有什么法律风险？"
        )

        self.assertEqual(plan.intent, "risk_diagnosis")
        self.assertTrue(plan.need_law_search)
        self.assertTrue(plan.need_enterprise_profile)
        self.assertTrue(plan.need_risk_scoring)

    def test_recognize_general_question(self):
        plan = recognize_intent("请介绍一下你自己")

        self.assertEqual(plan.intent, "general")
        self.assertFalse(plan.need_law_search)
        self.assertFalse(plan.need_risk_scoring)


class OrchestratorTests(unittest.TestCase):
    """智能体编排流程测试。"""

    def test_run_complete_workflow(self):
        orchestrator = Orchestrator()

        answer = orchestrator.run(
            question="我们公司没有签订劳动合同有什么风险？"
        )

        self.assertIsInstance(answer, Answer)
        self.assertEqual(answer.risk_level, "high")
        self.assertTrue(answer.requires_human_review)
        self.assertIn("ev_001", answer.citation_ids)
        self.assertGreaterEqual(answer.confidence, 0.85)

    def test_reject_unknown_citation(self):
        orchestrator = Orchestrator()

        answer = Answer(
            conclusion="测试结论",
            risk_level="low",
            reasoning_summary="测试理由",
            recommendations=["测试建议"],
            citation_ids=["citation_does_not_exist"],
            confidence=1.0,
            model_route="local",
            requires_human_review=False,
        )

        with self.assertRaises(ValueError):
            orchestrator.validate_answer(
                answer=answer,
                evidence=[],
            )

    def test_accept_valid_citation(self):
        orchestrator = Orchestrator()

        evidence = [
            Evidence(
                evidence_id="ev_test",
                source_type="law",
                title="测试法规",
                article="第一条",
                text="测试法规内容",
                document_id="doc_test",
                chunk_id="chunk_001",
                score=0.9,
            )
        ]

        answer = Answer(
            conclusion="测试结论",
            risk_level="low",
            reasoning_summary="测试理由",
            recommendations=[],
            citation_ids=["ev_test"],
            confidence=1.0,
            model_route="local",
            requires_human_review=False,
        )

        # 不应该抛出异常
        orchestrator.validate_answer(
            answer=answer,
            evidence=evidence,
        )

    def test_reject_empty_conclusion(self):
        orchestrator = Orchestrator()

        answer = Answer(
            conclusion="   ",
            risk_level="low",
            reasoning_summary="测试理由",
            recommendations=[],
            citation_ids=[],
            confidence=1.0,
            model_route="local",
            requires_human_review=False,
        )

        with self.assertRaises(ValueError):
            orchestrator.validate_answer(
                answer=answer,
                evidence=[],
            )


if __name__ == "__main__":
    unittest.main()