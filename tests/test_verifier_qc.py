"""Verifier 的 QC 测试：给 EM 打分函数（gca/reward.py 的 compute_score）造输入，检查分数是不是我们预想的。

每个测试 = 一个输入 + 一个"应该得几分"。分五类：
  正例（对的必须给分）、反例（错的不能给分）、边界、规则（>10 个 <answer> 打折）、已知漏洞。
运行：python -m pytest tests/test_verifier_qc.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gca.reward import compute_score  # noqa: E402

GOLD = {"target": ["Thomas Jefferson"]}


def score(output, gold=GOLD):
    return compute_score("searchR1_nq", output, gold, {})["score"]


# ---------- 正例：答对了，必须给 1 分 ----------
def test_完全一致():
    assert score("<answer>Thomas Jefferson</answer>") == 1


def test_大小写和标点不影响():  # 归一化：转小写、去标点
    assert score("<answer>thomas jefferson.</answer>") == 1


def test_命中别名列表里任意一个():
    assert score("<answer>Jefferson</answer>", {"target": ["Thomas Jefferson", "Jefferson"]}) == 1


# ---------- 反例：答错了，不能给分 ----------
def test_错误答案():
    assert score("<answer>John Adams</answer>") == 0


def test_只答对一部分也算错():  # EM 是"完全一致"，不是"包含"
    assert score("<answer>Jefferson</answer>") == 0


def test_多答了内容也算错():
    assert score("<answer>Thomas Jefferson of Virginia</answer>") == 0


# ---------- 边界：空的、没标签的 ----------
def test_空输出():
    assert score("") == 0


def test_没有answer标签():  # 答案写对了但没放进标签，照样 0 分
    assert score("The answer is Thomas Jefferson.") == 0


def test_空answer标签():
    assert score("<answer></answer>") == 0


# ---------- 规则：防刷屏 ----------
def test_answer标签超过10个打四折():
    spam = "<answer>x</answer>" * 11 + "<answer>Thomas Jefferson</answer>"
    assert score(spam) == 0.25


# ---------- 已知漏洞：记录 verifier 现在的行为，以及它和"真实情况"对不上的地方 ----------
def test_只看最后一个answer():  # 先答对后改错 -> 0 分（这条行为合理）
    assert score("<answer>Thomas Jefferson</answer> 等等 <answer>John Adams</answer>") == 0


def test_漏洞_编造检索结果答对也给满分():
    # 模型没搜索，自己编了一段 <tool_response>，再照着答对。EM 只看最终答案，所以照样 1 分。
    fake = ("<tool_response>\nDoc 1\nThe Declaration was drafted by Thomas Jefferson.\n</tool_response>\n"
            "<answer>Thomas Jefferson</answer>")
    assert score(fake) == 1   # 这是现状：漏洞存在


@pytest.mark.xfail(reason="已知漏洞：EM 不检查检索结果是不是模型自己编的，修好之前这条预期会失败", strict=True)
def test_理想行为_编造检索结果应该得0分():
    fake = ("<tool_response>\nDoc 1\nThe Declaration was drafted by Thomas Jefferson.\n</tool_response>\n"
            "<answer>Thomas Jefferson</answer>")
    assert score(fake) == 0
