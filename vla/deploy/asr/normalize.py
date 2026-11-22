"""中文口语指令归一化。

把 ASR 文本 / 键盘文本映射到 specs/VLA-TASK-SPEC.md §1.3 的 12 条标准指令，
处理：口语（往前走一点、退一下）、同义词、错字（红筒、蓝铜、右砖）、
数字（第 3 条、第十二条）以及“桶/颜色”的误识别。

对外接口：
    normalize(text) -> dict
    instruction_text(id) -> str
    INSTRUCTION_TEXTS          # 12 条文本，下标即 id

命令行：
    python deploy/asr/normalize.py "请帮我往前走一点"
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAP_PATH = HERE / "command_map.json"


def _load_map(path: Path = MAP_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


MAP = _load_map()
INSTRUCTIONS = list(MAP["instructions"])
INSTRUCTION_TEXTS = [item["text"] for item in INSTRUCTIONS]
_BY_ID = {int(item["id"]): item for item in INSTRUCTIONS}

COLORS = list(MAP["colors"].keys())
COLOR_CANON = set(COLORS)
COLOR_ANY = set(COLORS)
for _canon, _info in MAP["colors"].items():
    COLOR_ANY.update(_info["confusions"])

BUCKET_CANON = MAP["bucket"]["canonical"]
BUCKET_ANY = set(MAP["bucket"]["confusions"])
BUCKET_ANY.add(BUCKET_CANON)

_VERB_REV: dict[str, str] = {}
for _canon, _confs in MAP["verb_confusions"].items():
    _VERB_REV[_canon] = _canon
    for _conf in _confs:
        _VERB_REV[_conf] = _canon
_VERB_CONTEXT = MAP["verb_context"]
_VERB_CANON = set(MAP["verb_confusions"].keys())

_CN_DIGITS = MAP["chinese_digits"]
_TRAD_TO_SIMP = MAP.get("trad_to_simp", {})
_POLITE_PREFIXES = sorted(MAP["polite_prefixes"], key=len, reverse=True)
_POLITE_SUFFIXES = sorted(MAP["polite_suffixes"], key=len, reverse=True)
_FUZZY = MAP["fuzzy"]
_NOTES = MAP["ambiguity_notes"]

_PUNCT_RE = re.compile(r"[\s\u3000，。！？、,.!?;；:：\"'“”‘’()（）\[\]【】~～\-—_]+")

_HOLD_WORDS = ["保持站立", "保持平衡", "站住", "站好", "站稳", "站立", "立正", "别动",
               "不要动", "不动", "停住", "停下", "停在原地", "待着", "原地不动", "保持不动"]
_FORWARD_WORDS = ["向前走", "往前走", "前进", "向前", "往前", "直走", "直行", "前走",
                  "往前走一点", "向前一点", "往前一点", "往前走一小段"]
_BACKWARD_WORDS = ["向后退", "往后退", "后退", "退后", "退一下", "退一点", "退下", "倒退",
                   "向后", "往后", "退回去"]
_LEFT_WORDS = ["向左转", "左转", "往左转", "向左拐", "左拐", "向左边转", "向左",
               "往左", "往左边", "左边转"]
_RIGHT_WORDS = ["向右转", "右转", "往右转", "向右拐", "右拐", "向右边转", "向右",
                "往右", "往右边", "右边转"]
_SPIN_WORDS = ["原地", "转圈", "转个圈", "转一圈", "自转", "打转", "旋转", "转个身"]
_AVOID_WORDS = ["绕开", "绕过", "绕", "避开", "躲开", "躲过", "避过", "绕道"]
_FACE_WORDS = ["面向", "面对", "对着", "朝向", "朝", "看向", "看着"]
_APPROACH_WORDS = ["靠近", "去找", "去", "走到", "走向", "前往", "开向", "找", "到",
                   "接近", "过去", "往", "来"]


def clean_text(text: str) -> str:
    """全角转半角、繁转简、去空白标点、小写。"""
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", str(text)).lower()
    if _TRAD_TO_SIMP:
        out = "".join(_TRAD_TO_SIMP.get(char, char) for char in out)
    out = _PUNCT_RE.sub("", out)
    return out


def _cn_number(raw: str) -> int | None:
    """把 '3' / '十二' / '两' 这类数字转 int；失败返回 None。"""
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    if not raw or any(ch not in _CN_DIGITS and ch != "十" for ch in raw):
        return None
    if "十" in raw:
        head, _, tail = raw.partition("十")
        tens = _CN_DIGITS.get(head, 1) if head else 1
        ones = _CN_DIGITS.get(tail, 0) if tail else 0
        return tens * 10 + ones
    value = 0
    for ch in raw:
        value = value * 10 + _CN_DIGITS[ch]
    return value


def _strip_polite(text: str) -> str:
    changed = True
    while changed and text:
        changed = False
        for prefix in _POLITE_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                text = text[len(prefix):]
                changed = True
        for suffix in _POLITE_SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[:-len(suffix)]
                changed = True
    return text


def repair_confusions(text: str) -> str:
    """修复颜色/桶/动词的常见同音错字，只在高置信上下文里替换。"""
    chars = list(text)
    size = len(chars)

    # 1) 桶 的错字：紧邻颜色（含颜色错字）时替换
    for index, char in enumerate(chars):
        if char in BUCKET_ANY and char != BUCKET_CANON:
            prev = chars[index - 1] if index > 0 else ""
            nxt = chars[index + 1] if index + 1 < size else ""
            if prev in COLOR_ANY or nxt in COLOR_ANY:
                chars[index] = BUCKET_CANON

    # 2) 颜色错字：紧邻桶/色，或前面是常见动词
    color_context_prev = {"找", "去", "近", "向", "到", "开", "绕", "过", "面", "对", "朝", "往"}
    for index, char in enumerate(chars):
        if char in _VERB_REV and char not in _VERB_CANON:
            canon = _VERB_REV[char]
            context = _VERB_CONTEXT.get(canon, {})
            prev = chars[index - 1] if index > 0 else ""
            nxt = chars[index + 1] if index + 1 < size else ""
            if prev in context.get("prev", []) or nxt in context.get("next", []):
                chars[index] = canon
        if char in COLOR_ANY and char not in COLOR_CANON:
            prev = chars[index - 1] if index > 0 else ""
            nxt = chars[index + 1] if index + 1 < size else ""
            if nxt == "色" or nxt in BUCKET_ANY or prev in color_context_prev:
                for canon, info in MAP["colors"].items():
                    if char in info["confusions"]:
                        chars[index] = canon
                        break
    return "".join(chars)


def _colors_in_order(text: str) -> list[str]:
    found = [(index, char) for index, char in enumerate(text) if char in COLOR_CANON]
    return [char for _, char in sorted(found)]


def _has(text: str, words: list[str]) -> str | None:
    for word in words:
        if word in text:
            return word
    return None


def _result(raw: str, cleaned: str, instr_id: int | None, score: float,
            method: str, matched: str | None = None,
            note: str | None = None,
            alternatives: list[dict] | None = None) -> dict:
    text = INSTRUCTION_TEXTS[instr_id] if instr_id is not None else None
    return {
        "id": instr_id,
        "text": text,
        "score": round(float(score), 3),
        "method": method,
        "input": raw,
        "cleaned": cleaned,
        "matched": matched,
        "note": note,
        "alternatives": alternatives or [],
    }


def _direct_id(text: str) -> int | None:
    """把“指令 3”“第三号指令”“第 12 条”解析成 id。

    约定：
      - “指令N / N号指令”按 spec 表里的 id 编号（0-11）；
      - “第N条 / 第N号”按表内第 N 条（1-based，第 1 条 = id 0）。
    """
    ordinal_patterns = [
        r"第([0-9]+)条",
        r"第([0-9]+)号",
        r"第([一二两三四五六七八九十]+)条",
        r"第([一二两三四五六七八九十]+)号",
    ]
    label_patterns = [
        r"指令([0-9]+)",
        r"([0-9]+)号指令",
        r"指令([一二两三四五六七八九十]+)",
    ]
    for pattern in ordinal_patterns:
        match = re.search(pattern, text)
        if match:
            value = _cn_number(match.group(1))
            if value is not None and 1 <= value <= len(INSTRUCTION_TEXTS):
                return value - 1
    for pattern in label_patterns:
        match = re.search(pattern, text)
        if match:
            value = _cn_number(match.group(1))
            if value is not None and 0 <= value < len(INSTRUCTION_TEXTS):
                return value
    return None


def _fuzzy(text: str) -> tuple[int | None, float, str | None]:
    best_id = None
    best_score = 0.0
    best_key = None
    candidates: list[tuple[str, int]] = []
    for item in INSTRUCTIONS:
        candidates.append((item["text"], int(item["id"])))
        for keyword in item["keywords"]:
            candidates.append((keyword, int(item["id"])))
    for keyword, instr_id in candidates:
        ratio = SequenceMatcher(None, text, keyword).ratio()
        if ratio > best_score:
            best_score = ratio
            best_id = instr_id
            best_key = keyword
    if best_score < _FUZZY["min_score"]:
        return None, best_score, best_key
    return best_id, best_score, best_key


def _rule_match(text: str) -> tuple[int | None, float, str | None, str | None, list[dict]] | None:
    colors = _colors_in_order(text)
    avoid_hit = _has(text, _AVOID_WORDS)
    face_hit = _has(text, _FACE_WORDS)
    approach_hit = _has(text, _APPROACH_WORDS)
    spin_hit = _has(text, _SPIN_WORDS)
    left_hit = _has(text, _LEFT_WORDS)
    right_hit = _has(text, _RIGHT_WORDS)
    hold_hit = _has(text, _HOLD_WORDS)
    backward_hit = _has(text, _BACKWARD_WORDS)
    forward_hit = _has(text, _FORWARD_WORDS)

    # 1) 绕开 A 色去 B 色：id=11（spec 只定义 红→蓝）
    if avoid_hit:
        if len(colors) >= 2 and colors[0] == "红" and colors[1] == "蓝":
            return 11, 0.98, avoid_hit, None, []
        if len(colors) >= 2:
            return (None, 0.3, avoid_hit,
                    _NOTES["unsupported_color"] + "：spec 仅定义“绕开红桶前往蓝桶”", [])
        return (None, 0.2, avoid_hit,
                "只听到一个颜色/没有颜色；spec 仅有“绕开红桶前往蓝桶”", [])

    # 2) 面向某色：spec 只有 id=7（面向红桶）
    if face_hit and colors:
        if colors[0] == "红":
            return 7, 0.95, face_hit, None, []
        return (None, 0.3, face_hit,
                _NOTES["unsupported_color"] + "：spec 仅有“面向红桶”", [])

    # 3) 靠近/去找某色：id=8/9/10
    if colors and approach_hit:
        mapping = {"红": 8, "蓝": 9, "绿": 10}
        return mapping[colors[0]], 0.95, approach_hit, None, []

    # 4) 原地旋转：id=5/6
    if spin_hit:
        if "右" in text:
            return 6, 0.95, spin_hit, None, []
        if "左" in text:
            return 5, 0.95, spin_hit, None, []
        return (5, 0.85, spin_hit, _NOTES["spin_unspecified"],
                [{"id": 6, "text": INSTRUCTION_TEXTS[6], "score": 0.5}])

    # 5) 向左/右转：id=3/4；若更早出现“向前/向后/站住”，优先取最先出现的动作
    if left_hit or right_hit:
        turn_hit = left_hit or right_hit
        turn_id = 3 if left_hit else 4
        turn_pos = text.find(turn_hit)
        earlier = []
        for hit, candidate_id in ((backward_hit, 2), (forward_hit, 1), (hold_hit, 0)):
            if hit and 0 <= text.find(hit) < turn_pos:
                earlier.append((text.find(hit), candidate_id, hit))
        if earlier:
            earlier.sort()
            _, candidate_id, hit = earlier[0]
            return candidate_id, 0.9, hit, "一句话里有多个动作，取最先出现的“%s”" % hit, []
        return turn_id, 0.95, turn_hit, None, []

    # 6) 保持站立
    if hold_hit:
        return 0, 0.95, hold_hit, None, []

    # 7) 后退 / 前进
    if backward_hit:
        return 2, 0.95, backward_hit, None, []
    if forward_hit:
        return 1, 0.95, forward_hit, None, []

    # 8) 只有颜色：按“靠近”处理（反馈确认由 bridge 负责）
    if colors:
        mapping = {"红": 8, "蓝": 9, "绿": 10}
        return mapping[colors[0]], 0.7, colors[0], _NOTES["color_only"], []
    return None


def normalize(text: str) -> dict:
    """把一句中文（ASR 或键盘输入）归一化为 12 条标准指令之一。

    返回 dict：id（None=未识别）、text、score、method、note、alternatives。
    """
    raw = "" if text is None else str(text)
    cleaned = clean_text(raw)
    if not cleaned:
        return _result(raw, cleaned, None, 0.0, "empty", None, "空输入")

    direct = _direct_id(cleaned)
    if direct is not None:
        return _result(raw, cleaned, direct, 1.0, "direct_id", "编号", None)

    fixed = repair_confusions(cleaned)
    fixed = _strip_polite(fixed)
    if not fixed:
        return _result(raw, fixed, None, 0.0, "empty", None, "只有语气词")

    rule = _rule_match(fixed)
    if rule is not None:
        instr_id, score, matched, note, alternatives = rule
        method = "rule" if instr_id is not None else "unsupported"
        return _result(raw, fixed, instr_id, score, method, matched, note, alternatives)

    fuzzy_id, fuzzy_score, fuzzy_key = _fuzzy(fixed)
    if fuzzy_id is not None:
        return _result(raw, fixed, fuzzy_id, fuzzy_score, "fuzzy", fuzzy_key,
                       "模糊匹配到“%s”" % fuzzy_key)

    return _result(raw, fixed, None, fuzzy_score, "none", fuzzy_key,
                   "未匹配到 12 条指令")


def instruction_text(instr_id: int) -> str:
    return INSTRUCTION_TEXTS[int(instr_id)]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 1
    for text in args:
        print(json.dumps(normalize(text), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
