"""BM25 关键词检索层（纯标准库实现，零第三方依赖）。

当前实现：
- BM25Okapi 算法手写实现（与 rank-bm25 库等价），仅用 math / collections。
- 中文分词采用字符二元组（bigram），无需 jieba，保证零依赖可在本机运行。

后续可选升级（不改变接口）：
- 安装 jieba 后替换 _tokenize 为词级分词，检索质量更高。
- 安装 rank-bm25 后替换 BM25Okapi 为库实现。
- 接入向量检索 + RRF 融合 + Rerank。
"""

from __future__ import annotations

import math
import re
from collections import Counter

from schemas import Evidence


# 演示版法规语料库。
# 正式版本应从数据库或文件加载，并支持增量更新与版本管理。
CORPUS = [
    {
        "evidence_id": "ev_001",
        "source_type": "law",
        "title": "中华人民共和国劳动合同法",
        "article": "第八十二条",
        "text": (
            "用人单位自用工之日起超过一个月不满一年"
            "未与劳动者订立书面劳动合同的，"
            "应当向劳动者每月支付二倍的工资。"
        ),
        "document_id": "doc_labor_contract_law",
        "chunk_id": "chunk_082",
    },
    {
        "evidence_id": "ev_002",
        "source_type": "law",
        "title": "中华人民共和国劳动合同法",
        "article": "第十条",
        "text": (
            "建立劳动关系，应当订立书面劳动合同。"
            "已建立劳动关系，未同时订立书面劳动合同的，"
            "应当自用工之日起一个月内订立书面劳动合同。"
        ),
        "document_id": "doc_labor_contract_law",
        "chunk_id": "chunk_010",
    },
    {
        "evidence_id": "ev_003",
        "source_type": "law",
        "title": "中华人民共和国劳动合同法",
        "article": "第八十五条",
        "text": (
            "用人单位未按照劳动合同约定或者国家规定"
            "及时足额支付劳动者劳动报酬的，"
            "由劳动行政部门责令限期支付劳动报酬、加班费或者经济补偿。"
        ),
        "document_id": "doc_labor_contract_law",
        "chunk_id": "chunk_085",
    },
    {
        "evidence_id": "ev_004",
        "source_type": "law",
        "title": "中华人民共和国社会保险法",
        "article": "第五十八条",
        "text": (
            "用人单位应当自用工之日起三十日内"
            "为其职工向社会保险经办机构申请办理社会保险登记。"
            "未办理社会保险登记的，由社会保险经办机构核定其应当缴纳的社会保险费。"
        ),
        "document_id": "doc_social_insurance_law",
        "chunk_id": "chunk_058",
    },
    {
        "evidence_id": "ev_005",
        "source_type": "law",
        "title": "中华人民共和国个人信息保护法",
        "article": "第十三条",
        "text": (
            "符合下列情形之一的，个人信息处理者方可处理个人信息："
            "取得个人的同意；"
            "为订立、履行个人作为一方当事人的合同所必需。"
        ),
        "document_id": "doc_pipl",
        "chunk_id": "chunk_013",
    },
    {
        "evidence_id": "ev_006",
        "source_type": "law",
        "title": "中华人民共和国民法典",
        "article": "第四百六十九条",
        "text": (
            "当事人订立合同，可以采用书面形式、口头形式或者其他形式。"
            "书面形式是合同书、信件、电报等可以有形地表现所载内容的形式。"
        ),
        "document_id": "doc_civil_code",
        "chunk_id": "chunk_469",
    },
]


def _tokenize(text: str) -> list[str]:
    """零依赖中文分词：保留中文/字母/数字，中文按字符二元组。"""

    tokens: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text):
        if re.fullmatch(r"[A-Za-z0-9]+", seg):
            tokens.append(seg.lower())
        else:
            for i in range(len(seg) - 1):
                tokens.append(seg[i : i + 2])
            if len(seg) == 1:
                tokens.append(seg)
    return tokens


class BM25Okapi:
    """BM25Okapi 检索器（纯标准库实现）。"""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.n = len(corpus_tokens)
        self.doc_lens = [len(toks) for toks in corpus_tokens]
        self.avgdl = (sum(self.doc_lens) / self.n) if self.n else 0.0
        self.tf = [Counter(toks) for toks in corpus_tokens]

        self.df: dict[str, int] = {}
        for toks in corpus_tokens:
            for term in set(toks):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for q in query_tokens:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for i in range(self.n):
                f = self.tf[i].get(q, 0)
                if f == 0:
                    continue
                norm = self.doc_lens[i] / self.avgdl if self.avgdl else 0.0
                denom = f + self.k1 * (1 - self.b + self.b * norm)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


class BM25Retriever:
    """基于 BM25 的关键词检索器，封装语料库与索引。"""

    def __init__(self, corpus: list[dict]):
        self._corpus = list(corpus)
        self._tokenized = [
            _tokenize(doc["title"] + " " + doc["article"] + " " + doc["text"])
            for doc in self._corpus
        ]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 3) -> list[Evidence]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        max_score = max((s for _, s in ranked), default=0.0) or 1.0

        results: list[Evidence] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            doc = self._corpus[idx]
            results.append(
                Evidence(
                    evidence_id=doc["evidence_id"],
                    source_type=doc["source_type"],
                    title=doc["title"],
                    article=doc["article"],
                    text=doc["text"],
                    document_id=doc["document_id"],
                    chunk_id=doc["chunk_id"],
                    score=round(score / max_score, 2),
                )
            )
        return results


_retriever = BM25Retriever(CORPUS)


def search(query: str, top_k: int = 3) -> list[Evidence]:
    """对外检索入口，返回按相关性排序的证据列表。"""

    return _retriever.search(query, top_k=top_k)


if __name__ == "__main__":
    for q in [
        "我们公司没有签订劳动合同有什么风险？",
        "拖欠工资会怎么处罚？",
        "员工社保没有交怎么办？",
    ]:
        print("\n查询：" + q)
        for ev in search(q, top_k=3):
            print("  [" + str(ev.score) + "] " + ev.evidence_id + " " + ev.title + " " + ev.article)
