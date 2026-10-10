"""合规智能体编排器：控制意图识别、工具调用、模型生成和出口校验。"""

import json
from dataclasses import asdict

from schemas import (
    Answer,
    EnterpriseProfile,
    Evidence,
    IntentPlan,
    RiskResult,
)
from tools import (
    calculate_risk,
    get_enterprise_profile,
    rag_search,
)


def recognize_intent(question: str) -> IntentPlan:
    """
    第一阶段：理解用户意图并决定需要调用哪些工具。

    当前使用规则识别，后续可以替换为大模型结构化输出。
    """

    text = question.lower()

    need_law_search = any(
        word in text
        for word in [
            "法规",
            "法律",
            "法条",
            "合同",
            "风险",
            "处罚",
            "合规",
        ]
    )

    need_enterprise_profile = any(
        word in text
        for word in [
            "我们公司",
            "本公司",
            "企业",
            "员工",
            "历史风险",
        ]
    )

    need_risk_scoring = any(
        word in text
        for word in [
            "风险",
            "评分",
            "等级",
            "诊断",
            "处罚",
        ]
    )

    if need_risk_scoring:
        intent = "risk_diagnosis"
    elif need_law_search:
        intent = "consultation"
    else:
        intent = "general"

    return IntentPlan(
        intent=intent,
        domain="compliance",
        need_law_search=need_law_search,
        need_enterprise_profile=need_enterprise_profile,
        need_risk_scoring=need_risk_scoring,
    )


class DemoModel:
    """
    演示模型。

    正式版本可以替换为：
    - Ollama 本地模型
    - vLLM 本地模型
    - Qwen
    - DeepSeek
    - OpenAI 兼容接口
    """

    def generate(
        self,
        question: str,
        evidence: list[Evidence],
        profile: EnterpriseProfile | None,
        risk: RiskResult | None,
    ) -> dict:
        print(f"[模型生成] 处理问题：{question}")

        if not evidence:
            return {
                "conclusion": "没有检索到足够证据，暂时不能形成确定结论。",
                "reasoning_summary": "需要补充法规、制度或企业事实。",
                "recommendations": [
                    "补充相关材料后再次咨询",
                ],
                "citation_ids": [],
            }

        citation_ids = [
            item.evidence_id
            for item in evidence
        ]

        reasoning = "根据检索到的法规证据，形成了初步合规判断。"

        if profile and profile.historical_risks:
            reasoning += " 企业存在历史风险记录，需要重点核查。"

        return {
            "conclusion": "存在相关合规风险，需要进一步核查和整改。",
            "reasoning_summary": reasoning,
            "recommendations": [
                "核对相关业务、合同和用工记录",
                "根据法规要求制定整改计划",
                "保留整改过程和完成证明",
            ],
            "citation_ids": citation_ids,
        }


class Orchestrator:
    """控制智能体执行流程的核心状态机。"""

    def __init__(self, model: DemoModel | None = None):
        self.model = model or DemoModel()

    def run(
        self,
        question: str,
        enterprise_id: str = "demo_enterprise",
    ) -> Answer:
        print("\n========== 智能体开始执行 ==========")

        # 状态 1：识别意图
        plan = recognize_intent(question)
        print(
            "[意图识别]",
            f"意图={plan.intent}",
            f"法规检索={plan.need_law_search}",
            f"企业档案={plan.need_enterprise_profile}",
            f"风险评分={plan.need_risk_scoring}",
        )

        evidence: list[Evidence] = []
        profile: EnterpriseProfile | None = None
        risk: RiskResult | None = None

        # 状态 2：按照计划调用工具
        if plan.need_law_search:
            evidence.extend(rag_search(question))

        if plan.need_enterprise_profile:
            profile = get_enterprise_profile(enterprise_id)

        if plan.need_risk_scoring:
            risk = calculate_risk(evidence, profile)

        # 状态 3：调用模型生成答案
        generated = self.model.generate(
            question=question,
            evidence=evidence,
            profile=profile,
            risk=risk,
        )

        # 状态 4：计算答案置信度
        confidence = self.calculate_confidence(
            generated=generated,
            evidence=evidence,
        )
        print(f"[置信度] {confidence}")

        risk_level = risk.level if risk else "low"

        # 状态 5：高风险或低置信度进入人工复核
        requires_human_review = (
            confidence < 0.85
            or risk_level == "high"
        )

        answer = Answer(
            conclusion=generated["conclusion"],
            risk_level=risk_level,
            reasoning_summary=generated["reasoning_summary"],
            recommendations=generated["recommendations"],
            citation_ids=generated["citation_ids"],
            confidence=confidence,
            model_route="local",
            requires_human_review=requires_human_review,
        )

        # 状态 6：执行最终出口校验
        self.validate_answer(
            answer=answer,
            evidence=evidence,
        )

        print("[出口校验] 通过")
        print("========== 智能体执行完成 ==========\n")

        return answer

    def calculate_confidence(
        self,
        generated: dict,
        evidence: list[Evidence],
    ) -> float:
        """根据证据和模型结果计算置信度。"""

        score = 0.40

        if evidence:
            score += 0.30

        if generated.get("citation_ids"):
            score += 0.20

        if generated.get("reasoning_summary"):
            score += 0.10

        return round(min(score, 1.0), 2)

    def validate_answer(
        self,
        answer: Answer,
        evidence: list[Evidence],
    ) -> None:
        """
        出口校验：

        1. 结论不能为空
        2. 引用证据必须真实存在
        3. 置信度必须在 0 到 1 之间
        """

        if not answer.conclusion.strip():
            raise ValueError("结论不能为空")

        valid_evidence_ids = {
            item.evidence_id
            for item in evidence
        }

        for citation_id in answer.citation_ids:
            if citation_id not in valid_evidence_ids:
                raise ValueError(
                    f"发现不存在的证据引用：{citation_id}"
                )

        if not 0 <= answer.confidence <= 1:
            raise ValueError("置信度必须在 0 到 1 之间")


if __name__ == "__main__":
    orchestrator = Orchestrator()

    result = orchestrator.run(
        question="我们公司没有和部分员工签订劳动合同，有什么风险？"
    )

    print(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
        )
    )