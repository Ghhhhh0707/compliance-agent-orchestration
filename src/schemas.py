"""智能体各模块之间传递的数据结构。"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class IntentPlan:
    """意图识别和工具选择结果。"""

    intent: str
    domain: str
    need_law_search: bool
    need_enterprise_profile: bool
    need_risk_scoring: bool


@dataclass
class Evidence:
    """法规、制度或其他知识证据。"""

    evidence_id: str
    source_type: str
    title: str
    article: str
    text: str
    document_id: str
    chunk_id: str
    score: float


@dataclass
class EnterpriseProfile:
    """经过授权访问的企业档案。"""

    enterprise_id: str
    industry: str
    employee_count: int
    region: str
    historical_risks: list[str] = field(default_factory=list)


@dataclass
class RiskResult:
    """由确定性规则产生的风险结果。"""

    level: Literal["low", "medium", "high"]
    score: int
    reasons: list[str]


@dataclass
class Answer:
    """智能体编排系统的最终结构化答案。"""

    conclusion: str
    risk_level: str
    reasoning_summary: str
    recommendations: list[str]
    citation_ids: list[str]
    confidence: float
    model_route: str
    requires_human_review: bool