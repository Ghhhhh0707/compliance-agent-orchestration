"""智能体工具层：RAG 检索、企业档案查询和风险评分。"""

from __future__ import annotations

from schemas import EnterpriseProfile, Evidence, RiskResult


def rag_search(query: str) -> list[Evidence]:
    """
    演示版 RAG 检索工具。

    正式版本中，这里将连接：
    - 向量数据库
    - 全文关键词检索
    - RRF 结果融合
    - Rerank 精排
    """

    print(f"[工具调用] RAG 检索：{query}")

    return [
        Evidence(
            evidence_id="ev_001",
            source_type="law",
            title="中华人民共和国劳动合同法",
            article="第八十二条",
            text=(
                "用人单位自用工之日起超过一个月不满一年"
                "未与劳动者订立书面劳动合同的，"
                "应当向劳动者每月支付二倍的工资。"
            ),
            document_id="doc_labor_contract_law",
            chunk_id="chunk_082",
            score=0.93,
        )
    ]


def get_enterprise_profile(
    enterprise_id: str,
) -> EnterpriseProfile:
    """
    演示版企业档案查询工具。

    正式版本中必须增加：
    - 企业权限校验
    - 多租户隔离
    - 敏感字段脱敏
    - 审计日志
    """

    print(f"[工具调用] 企业档案查询：{enterprise_id}")

    return EnterpriseProfile(
        enterprise_id=enterprise_id,
        industry="software",
        employee_count=320,
        region="上海市",
        historical_risks=["未签订书面劳动合同"],
    )


def calculate_risk(
    evidence: list[Evidence],
    profile: EnterpriseProfile | None = None,
) -> RiskResult:
    """
    使用确定性规则计算风险。

    风险分数不应由大模型直接决定，大模型只负责解释风险结果。
    """

    print("[工具调用] 计算合规风险")

    score = 20
    reasons: list[str] = []

    if evidence:
        score += 35
        reasons.append("检索到相关法规依据")

    if profile and profile.historical_risks:
        score += 25
        reasons.append("企业存在历史风险记录")

    if score >= 75:
        level = "high"
    elif score >= 45:
        level = "medium"
    else:
        level = "low"

    return RiskResult(
        level=level,
        score=score,
        reasons=reasons,
    )