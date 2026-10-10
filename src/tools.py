"""智能体工具层：RAG 检索、企业档案查询和风险评分。"""

from __future__ import annotations

from rag import search as bm25_search
from schemas import EnterpriseProfile, Evidence, RiskResult


def rag_search(query: str) -> list[Evidence]:
    """
    BM25 关键词检索工具。

    当前委托 src/rag.py 的纯标准库 BM25 实现（字符二元组分词，零第三方依赖）。
    正式版本可扩展为：向量检索 + 关键词检索 + RRF 融合 + Rerank。
    """

    print(f"[工具调用] BM25 检索：{query}")
    return bm25_search(query, top_k=3)


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
