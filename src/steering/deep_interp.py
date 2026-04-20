"""
Deep interpretability analysis of Human vs AI writing patterns.

Goes beyond surface-level word counts to reveal:
  1. **Contrastive N-grams** — discriminative multi-word sequences (LLR scored)
  2. **Syntactic Templates** — recurring dependency-structure motifs
  3. **Construction Grammar** — specific syntactic constructions
     (correlatives, clefts, fronted adverbials, hedged claims, etc.)
  4. **Lexical Bundles** — formulaic 3–5 word sequences with PMI scoring
  5. **Information Density** — surprisal curves, burstiness, lexical diversity
"""

import re
import math
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import spacy
from spacy.tokens import Doc, Span


# ── Language-aware tokenization ──────────────────────────
def _tokenize(text: str, language: str = "en") -> List[str]:
    """Split text into tokens, using jieba for Chinese."""
    if language == "zh":
        import jieba
        return list(jieba.cut(text))
    return text.split()


def _split_sentences(text: str, language: str = "en") -> List[str]:
    """Split text into sentences."""
    if language == "zh":
        sents = re.split(r'[。！？；\n]+', text)
    else:
        sents = re.split(r'[.!?]+', text)
    return [s.strip() for s in sents if s.strip()]


# ═══════════════════════════════════════════════════════════
#  1. Contrastive N-gram Analysis
# ═══════════════════════════════════════════════════════════

def _extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """Extract character-cleaned n-grams from token list."""
    cleaned = [t.lower().strip(".,;:!?\"'()[]{}") for t in tokens]
    cleaned = [t for t in cleaned if t]
    return [" ".join(cleaned[i:i+n]) for i in range(len(cleaned) - n + 1)]


def _log_likelihood_ratio(a: int, b: int, c: int, d: int) -> float:
    """
    Log-likelihood ratio (G²) for a 2×2 contingency table.
    a = count in corpus1, b = count in corpus2,
    c = total ngrams corpus1, d = total ngrams corpus2.
    """
    def _safe_log(x):
        return math.log(x) if x > 0 else 0

    n = a + b
    e1 = c * n / (c + d) if (c + d) > 0 else 0
    e2 = d * n / (c + d) if (c + d) > 0 else 0
    if e1 == 0 or e2 == 0:
        return 0.0
    g2 = 2 * (a * _safe_log(a / e1) + b * _safe_log(b / e2))
    return g2


def contrastive_ngrams(
    human_texts: List[str],
    ai_texts: List[str],
    ns: Tuple[int, ...] = (2, 3, 4, 5),
    min_freq: int = 3,
    top_k: int = 40,
    language: str = "en",
) -> Dict:
    """
    Find the most discriminative n-grams between human and AI text
    using log-likelihood ratio (G²) scoring.

    Returns per-n results with AI-skewed and Human-skewed n-grams.
    """
    results = {}

    for n in ns:
        human_counter = Counter()
        ai_counter = Counter()

        for text in human_texts:
            tokens = _tokenize(text, language)
            for ng in _extract_ngrams(tokens, n):
                human_counter[ng] += 1

        for text in ai_texts:
            tokens = _tokenize(text, language)
            for ng in _extract_ngrams(tokens, n):
                ai_counter[ng] += 1

        total_h = sum(human_counter.values())
        total_a = sum(ai_counter.values())

        all_ngrams = set(human_counter.keys()) | set(ai_counter.keys())

        scored = []
        for ng in all_ngrams:
            h_count = human_counter.get(ng, 0)
            a_count = ai_counter.get(ng, 0)
            if h_count + a_count < min_freq:
                continue
            g2 = _log_likelihood_ratio(a_count, h_count, total_a, total_h)
            direction = "ai" if a_count / max(total_a, 1) > h_count / max(total_h, 1) else "human"
            if direction == "human":
                g2 = -g2
            scored.append({
                "ngram": ng,
                "g2": round(g2, 2),
                "direction": direction,
                "ai_freq": a_count,
                "human_freq": h_count,
                "ai_per_1k": round(a_count / max(total_a, 1) * 1000, 3),
                "human_per_1k": round(h_count / max(total_h, 1) * 1000, 3),
            })

        scored.sort(key=lambda x: x["g2"], reverse=True)
        ai_skewed = [s for s in scored if s["direction"] == "ai"][:top_k]

        scored.sort(key=lambda x: x["g2"])
        human_skewed = [s for s in scored if s["direction"] == "human"][:top_k]

        results[f"{n}-gram"] = {
            "ai_distinctive": ai_skewed,
            "human_distinctive": human_skewed,
            "total_unique_ai": len(ai_counter),
            "total_unique_human": len(human_counter),
        }

    return results


# ═══════════════════════════════════════════════════════════
#  2. Syntactic Template Extraction
# ═══════════════════════════════════════════════════════════

def _extract_sent_template(sent: Span, max_depth: int = 3) -> str:
    """
    Extract a structural template from a spaCy sentence.
    Encodes the dependency skeleton up to `max_depth` levels from root.
    Example: "nsubj:PRON + ROOT:VERB + dobj:NOUN + prep:ADP>pobj:NOUN"
    """
    root = None
    for tok in sent:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if root is None:
        return ""

    def _subtree(tok, depth):
        if depth >= max_depth:
            return f"{tok.dep_}:{tok.pos_}"
        children = sorted(tok.children, key=lambda c: c.i)
        child_strs = [_subtree(c, depth + 1) for c in children
                      if c.dep_ not in ("punct", "sp", "")]
        base = f"{tok.dep_}:{tok.pos_}"
        if child_strs:
            return base + "(" + ", ".join(child_strs) + ")"
        return base

    return _subtree(root, 0)


def _classify_sentence_opening(sent: Span) -> str:
    """Classify the opening structure of a sentence."""
    tokens = list(sent)
    if len(tokens) < 2:
        return "minimal"

    first = tokens[0]
    second = tokens[1] if len(tokens) > 1 else None

    # Discourse marker opening
    if first.lower_ in {
        "however", "moreover", "furthermore", "nevertheless", "additionally",
        "meanwhile", "consequently", "therefore", "thus", "hence",
        "indeed", "notably", "similarly", "conversely", "subsequently",
        "overall", "essentially", "ultimately", "accordingly",
    }:
        return "discourse_marker"

    # Adverbial clause opening (SCONJ ...)
    if first.pos_ == "SCONJ" or first.dep_ == "mark":
        return "adverbial_clause"

    # Prepositional phrase opening
    if first.pos_ == "ADP":
        return "prepositional"

    # Gerund/participial opening (-ing/-ed at start)
    if first.pos_ == "VERB" and first.tag_ in ("VBG", "VBN"):
        return "participial"

    # Existential "There is/are"
    if first.lower_ == "there" and second and second.lower_ in ("is", "are", "was", "were"):
        return "existential"

    # Pronoun subject
    if first.pos_ == "PRON" and first.dep_ in ("nsubj", "nsubjpass"):
        return "pronoun_subject"

    # Definite NP ("The X ...")
    if first.pos_ == "DET" and first.lower_ in ("the", "this", "these", "those"):
        return "definite_np"

    # Indefinite NP ("A X ...")
    if first.pos_ == "DET" and first.lower_ in ("a", "an", "some", "many", "several"):
        return "indefinite_np"

    # Proper noun subject
    if first.pos_ == "PROPN":
        return "proper_noun"

    # Numeric opening
    if first.pos_ == "NUM" or first.like_num:
        return "numeric"

    return "other"


def syntactic_templates(
    human_texts: List[str],
    ai_texts: List[str],
    nlp,
    top_k: int = 30,
) -> Dict:
    """
    Extract and compare syntactic templates and sentence openings.
    """
    human_templates = Counter()
    ai_templates = Counter()
    human_openings = Counter()
    ai_openings = Counter()

    # depth-2 skeleton for more generalized patterns
    human_skeletons = Counter()
    ai_skeletons = Counter()

    for text, t_counter, o_counter, s_counter in [
        (human_texts, human_templates, human_openings, human_skeletons),
        (ai_texts, ai_templates, ai_openings, ai_skeletons),
    ]:
        for t in text:
            doc = nlp(t)
            for sent in doc.sents:
                template = _extract_sent_template(sent, max_depth=3)
                if template:
                    t_counter[template] += 1

                skeleton = _extract_sent_template(sent, max_depth=2)
                if skeleton:
                    s_counter[skeleton] += 1

                opening = _classify_sentence_opening(sent)
                o_counter[opening] += 1

    # Compare openings
    all_openings = set(human_openings.keys()) | set(ai_openings.keys())
    total_h = sum(human_openings.values())
    total_a = sum(ai_openings.values())
    opening_comparison = {}
    for op in all_openings:
        h_pct = human_openings.get(op, 0) / max(total_h, 1) * 100
        a_pct = ai_openings.get(op, 0) / max(total_a, 1) * 100
        opening_comparison[op] = {
            "human_pct": round(h_pct, 2),
            "ai_pct": round(a_pct, 2),
            "delta_pct": round(a_pct - h_pct, 2),
            "human_count": human_openings.get(op, 0),
            "ai_count": ai_openings.get(op, 0),
        }

    # Find discriminative skeletons
    total_h_skel = sum(human_skeletons.values())
    total_a_skel = sum(ai_skeletons.values())
    all_skel = set(k for k, v in human_skeletons.items() if v >= 2) | \
               set(k for k, v in ai_skeletons.items() if v >= 2)

    skeleton_scored = []
    for skel in all_skel:
        h = human_skeletons.get(skel, 0)
        a = ai_skeletons.get(skel, 0)
        if h + a < 3:
            continue
        g2 = _log_likelihood_ratio(a, h, total_a_skel, total_h_skel)
        direction = "ai" if a / max(total_a_skel, 1) > h / max(total_h_skel, 1) else "human"
        if direction == "human":
            g2 = -g2
        skeleton_scored.append({
            "skeleton": skel,
            "g2": round(g2, 2),
            "direction": direction,
            "ai_count": a,
            "human_count": h,
        })
    skeleton_scored.sort(key=lambda x: abs(x["g2"]), reverse=True)

    return {
        "sentence_openings": dict(sorted(
            opening_comparison.items(),
            key=lambda x: abs(x[1]["delta_pct"]), reverse=True
        )),
        "top_skeletons": skeleton_scored[:top_k * 2],
        "human_template_diversity": len(human_templates),
        "ai_template_diversity": len(ai_templates),
        "human_skeleton_diversity": len(human_skeletons),
        "ai_skeleton_diversity": len(ai_skeletons),
    }


# ═══════════════════════════════════════════════════════════
#  3. Construction Grammar Patterns
# ═══════════════════════════════════════════════════════════

# Each pattern: (name, regex_or_func, description)
_CONSTRUCTION_PATTERNS = [
    # Correlatives
    ("not_only_but_also",
     r"\bnot\s+only\b.{1,80}\bbut\s+(also|additionally)\b",
     "Correlative: not only X but also Y"),
    ("neither_nor",
     r"\bneither\b.{1,60}\bnor\b",
     "Correlative: neither X nor Y"),
    ("both_and",
     r"\bboth\b.{1,60}\band\b",
     "Correlative: both X and Y"),
    ("either_or",
     r"\beither\b.{1,60}\bor\b",
     "Correlative: either X or Y"),
    ("whether_or",
     r"\bwhether\b.{1,80}\bor\b",
     "Alternative: whether X or Y"),

    # Cleft / pseudo-cleft
    ("it_cleft",
     r"\bit\s+(is|was|has\s+been)\b.{1,60}\bthat\b",
     "Cleft sentence: It is X that Y"),
    ("what_cleft",
     r"\bwhat\b.{1,40}\b(is|was)\b.{1,60}",
     "Pseudo-cleft: What X is Y"),

    # Existential
    ("existential_there",
     r"\bthere\s+(is|are|was|were|has\s+been|have\s+been|remains?)\b",
     "Existential: There is/are X"),

    # Hedged claims
    ("hedged_claim",
     r"\b(it\s+)?(may|might|could|can)\s+(be\s+)?(argued|said|noted|suggested|seen)\b",
     "Hedged claim: it may be argued/noted/said"),
    ("appears_seems",
     r"\b(it\s+)?(appears?|seems?)\s+(that|to|as\s+if)\b",
     "Hedged: appears/seems that/to"),
    ("is_worth_noting",
     r"\b(it\s+is\s+)?(worth|important|notable|noteworthy)\s+(to\s+)?(note|noting|mention|consider|recognize)\b",
     "Metadiscursive: worth noting/important to note"),

    # Contrast/concession
    ("while_although_clause",
     r"\b(while|although|though|even\s+though|whereas)\b.{5,120},",
     "Concessive fronted clause: While/Although X, Y"),
    ("despite_in_spite",
     r"\b(despite|in\s+spite\s+of)\b",
     "Concessive: despite/in spite of"),

    # Listing / enumeration
    ("first_second_third",
     r"\b(first(ly)?|second(ly)?|third(ly)?|finally|lastly)\b",
     "Enumeration marker: firstly/secondly/..."),
    ("on_one_hand",
     r"\bon\s+(the\s+)?(one|other)\s+hand\b",
     "Contrast: on the one/other hand"),

    # Causal chains
    ("as_a_result",
     r"\bas\s+a\s+result\b",
     "Causal: as a result"),
    ("this_means_that",
     r"\bthis\s+(means|implies|suggests|indicates)\s+that\b",
     "Causal chain: this means/implies that"),
    ("due_to",
     r"\bdue\s+to\s+(the|a|this|its)\b",
     "Causal: due to"),
    ("in_order_to",
     r"\bin\s+order\s+to\b",
     "Purpose: in order to"),

    # Metadiscursive
    ("it_is_important",
     r"\bit\s+is\s+(important|crucial|essential|critical|vital|worth)\s+(to|that)\b",
     "Metadiscursive: it is important to/that"),
    ("in_terms_of",
     r"\bin\s+terms\s+of\b",
     "Metadiscursive: in terms of"),
    ("in_the_context_of",
     r"\bin\s+the\s+context\s+of\b",
     "Metadiscursive: in the context of"),

    # Rhetorical questions
    ("rhetorical_question",
     r"[A-Z][^.!?]{10,80}\?",
     "Potential rhetorical question"),

    # Parenthetical
    ("parenthetical_dash",
     r"\s—\s.{3,60}\s—\s",
     "Parenthetical: — X —"),
    ("parenthetical_paren",
     r"\([^)]{5,80}\)",
     "Parenthetical: (X)"),

    # Appositive
    ("appositive_comma",
     r",\s+[a-z].{3,50},\s+",
     "Potential appositive/non-restrictive clause"),

    # Passive patterns
    ("by_passive",
     r"\b(was|were|been|being|is|are)\s+\w+ed\s+by\b",
     "Passive with agent: X was Yed by Z"),
    ("agentless_passive",
     r"\b(was|were|been|being|is|are)\s+\w+ed\b(?!\s+by\b)",
     "Agentless passive: X was Yed"),
]


# ── Chinese construction grammar patterns ─────────────────
_ZH_CONSTRUCTION_PATTERNS = [
    # Correlatives
    ("not_X_but_Y",
     r"不是.{1,20}而是",
     "关联句式：不是X而是Y"),
    ("although_but",
     r"虽然.{1,40}但(是)?",
     "让步转折：虽然X但(是)Y"),
    ("not_only_but_also",
     r"不(仅|但|光).{1,40}(而且|还|也|更)",
     "递进关联：不仅X而且Y"),
    ("neither_nor",
     r"既不.{1,30}也不",
     "否定并列：既不X也不Y"),
    ("both_and",
     r"既.{1,30}又",
     "并列关联：既X又Y"),
    ("either_or",
     r"(要么|或者).{1,40}(要么|或者)",
     "选择关联：要么X要么Y"),
    ("if_then",
     r"如果.{1,40}(那么|就|则)",
     "条件关联：如果X那么Y"),
    ("only_if",
     r"只有.{1,40}才",
     "必要条件：只有X才Y"),
    ("because_so",
     r"因为.{1,40}所以",
     "因果关联：因为X所以Y"),
    ("the_more_the_more",
     r"越.{1,20}越",
     "递进句式：越X越Y"),

    # Hedging / epistemic
    ("may_possibly",
     r"(可能|也许|或许|大概|似乎)",
     "模糊语气：可能/也许/似乎"),
    ("it_is_said",
     r"(据[说称报]|有人[认说]为|一般认为|普遍认为)",
     "引述来源：据说/据称"),
    ("relatively",
     r"(相对而言|总体而言|一般来说|总的来说|严格来说)",
     "元话语限定：相对而言/总体而言"),

    # Metadiscursive
    ("worth_noting",
     r"(值得注意的是|需要指出的是|需要强调的是|不可忽视的是)",
     "元话语：值得注意的是"),
    ("in_other_words",
     r"(换[句言]话说|也就是说|简而言之|换而言之)",
     "元话语改述：换句话说/也就是说"),
    ("as_mentioned",
     r"(如[前上]所述|正如.{1,10}所[说指述]|综上所述)",
     "回指元话语：如前所述/综上所述"),

    # Enumerative
    ("first_second",
     r"(首先|其次|再次|最后|第[一二三四五])",
     "列举标记：首先/其次/最后"),
    ("on_one_hand",
     r"一方面.{1,60}另一方面",
     "对比列举：一方面X另一方面Y"),

    # Causal chains
    ("because_of",
     r"(由于|鉴于|基于).{1,40}",
     "原因导入：由于/鉴于"),
    ("therefore",
     r"(因此|从而|进而|以至于)",
     "因果推进：因此/从而"),
    ("in_order_to",
     r"(为了|旨在).{1,30}",
     "目的句式：为了/旨在"),

    # Contrast
    ("however",
     r"(然而|但是|不过|可是|尽管如此)",
     "转折连接：然而/但是/不过"),
    ("on_contrary",
     r"(相反|与此相反|恰恰相反)",
     "对立转折：相反/恰恰相反"),

    # Rhetorical
    ("rhetorical_question",
     r"[^。！？]{8,60}[？?]",
     "修辞疑问句"),

    # Passive / impersonal
    ("bei_passive",
     r"被.{1,20}[了过着]",
     "被字句（被动语态）"),
    ("ba_construction",
     r"把.{1,20}[了过着]",
     "把字句（处置式）"),

    # Parallel structure
    ("parallel_comma",
     r"[\u4e00-\u9fff]{2,6}[，,][\u4e00-\u9fff]{2,6}[，,][\u4e00-\u9fff]{2,6}",
     "三项并列短语"),

    # Emphasis
    ("indeed",
     r"(事实上|实际上|确实|的确|毫无疑问)",
     "强调语气：事实上/确实"),
    ("undeniably",
     r"(不可否认|毋庸置疑|显而易见|众所周知)",
     "共识强化：不可否认/显而易见"),
]


def construction_patterns(
    human_texts: List[str],
    ai_texts: List[str],
    language: str = "en",
) -> Dict:
    """
    Detect specific syntactic constructions in H vs AI, with examples.
    """
    patterns = _ZH_CONSTRUCTION_PATTERNS if language == "zh" else _CONSTRUCTION_PATTERNS
    results = {}

    for name, pattern, description in patterns:
        regex = re.compile(pattern, re.IGNORECASE)

        h_count = 0
        a_count = 0
        h_examples = []
        a_examples = []

        for text in human_texts:
            matches = regex.findall(text)
            h_count += len(matches)
            if matches and len(h_examples) < 3:
                # Get full match with context
                for m in regex.finditer(text):
                    start = max(0, m.start() - 20)
                    end = min(len(text), m.end() + 20)
                    h_examples.append("..." + text[start:end] + "...")
                    if len(h_examples) >= 3:
                        break

        for text in ai_texts:
            matches = regex.findall(text)
            a_count += len(matches)
            if matches and len(a_examples) < 3:
                for m in regex.finditer(text):
                    start = max(0, m.start() - 20)
                    end = min(len(text), m.end() + 20)
                    a_examples.append("..." + text[start:end] + "...")
                    if len(a_examples) >= 3:
                        break

        h_per_text = h_count / max(len(human_texts), 1)
        a_per_text = a_count / max(len(ai_texts), 1)

        results[name] = {
            "description": description,
            "human_count": h_count,
            "ai_count": a_count,
            "human_per_text": round(h_per_text, 3),
            "ai_per_text": round(a_per_text, 3),
            "ratio": round(a_per_text / max(h_per_text, 0.001), 2),
            "human_examples": h_examples[:3],
            "ai_examples": a_examples[:3],
        }

    return results


# ═══════════════════════════════════════════════════════════
#  4. Lexical Bundles / Formulaic Sequences
# ═══════════════════════════════════════════════════════════

def _pointwise_mutual_info(
    joint_count: int, word_a_count: int, word_b_count: int, total: int,
) -> float:
    """PMI for a bigram component."""
    if joint_count == 0 or word_a_count == 0 or word_b_count == 0 or total == 0:
        return 0.0
    p_joint = joint_count / total
    p_a = word_a_count / total
    p_b = word_b_count / total
    return math.log2(p_joint / (p_a * p_b + 1e-15))


def lexical_bundles(
    human_texts: List[str],
    ai_texts: List[str],
    ns: Tuple[int, ...] = (3, 4, 5),
    min_freq: int = 3,
    top_k: int = 30,
    language: str = "en",
) -> Dict:
    """
    Extract formulaic multi-word sequences and compare repertoires.
    Uses frequency + PMI-based scoring.
    """
    results = {}

    for n in ns:
        human_bundles = Counter()
        ai_bundles = Counter()
        human_unigrams = Counter()
        ai_unigrams = Counter()

        for text in human_texts:
            tokens = _tokenize(text, language)
            tokens_clean = [t.lower().strip(".,;:!?\"'()[]，。；：！？""''【】") for t in tokens]
            tokens_clean = [t for t in tokens_clean if t]
            for t in tokens_clean:
                human_unigrams[t] += 1
            for ng in _extract_ngrams(tokens, n):
                human_bundles[ng] += 1

        for text in ai_texts:
            tokens = _tokenize(text, language)
            tokens_clean = [t.lower().strip(".,;:!?\"'()[]，。；：！？""''【】") for t in tokens]
            tokens_clean = [t for t in tokens_clean if t]
            for t in tokens_clean:
                ai_unigrams[t] += 1
            for ng in _extract_ngrams(tokens, n):
                ai_bundles[ng] += 1

        # AI-exclusive bundles (freq >= min_freq in AI, 0 in human)
        ai_exclusive = []
        for bundle, count in ai_bundles.most_common():
            if count >= min_freq and human_bundles.get(bundle, 0) == 0:
                ai_exclusive.append({"bundle": bundle, "freq": count})
            if len(ai_exclusive) >= top_k:
                break

        # Human-exclusive bundles
        human_exclusive = []
        for bundle, count in human_bundles.most_common():
            if count >= min_freq and ai_bundles.get(bundle, 0) == 0:
                human_exclusive.append({"bundle": bundle, "freq": count})
            if len(human_exclusive) >= top_k:
                break

        # Shared bundles with ratio
        shared = []
        total_h = sum(human_bundles.values())
        total_a = sum(ai_bundles.values())
        all_keys = set(human_bundles.keys()) & set(ai_bundles.keys())
        for bundle in all_keys:
            h = human_bundles[bundle]
            a = ai_bundles[bundle]
            if h + a < min_freq:
                continue
            h_rate = h / max(total_h, 1) * 1000
            a_rate = a / max(total_a, 1) * 1000
            shared.append({
                "bundle": bundle,
                "human_freq": h, "ai_freq": a,
                "human_per_1k": round(h_rate, 3),
                "ai_per_1k": round(a_rate, 3),
                "ratio": round(a_rate / max(h_rate, 0.001), 2),
            })
        shared.sort(key=lambda x: abs(math.log2(max(x["ratio"], 0.01))), reverse=True)

        results[f"{n}-word"] = {
            "ai_exclusive": ai_exclusive[:top_k],
            "human_exclusive": human_exclusive[:top_k],
            "shared_most_skewed": shared[:top_k],
            "ai_bundle_types": len([b for b, c in ai_bundles.items() if c >= min_freq]),
            "human_bundle_types": len([b for b, c in human_bundles.items() if c >= min_freq]),
        }

    return results


# ═══════════════════════════════════════════════════════════
#  5. Information Density Profile
# ═══════════════════════════════════════════════════════════

def _word_surprisal(word: str, unigram_probs: Dict[str, float]) -> float:
    """Estimate surprisal as -log2(P(word))."""
    p = unigram_probs.get(word.lower(), 1e-6)
    return -math.log2(p)


def _type_token_ratio_window(tokens: List[str], window: int = 50) -> List[float]:
    """Moving-window TTR (lexical diversity over text)."""
    if len(tokens) < window:
        return [len(set(tokens)) / max(len(tokens), 1)]
    ratios = []
    for i in range(len(tokens) - window + 1):
        window_tokens = tokens[i:i + window]
        ratios.append(len(set(window_tokens)) / window)
    return ratios


def information_density(
    human_texts: List[str],
    ai_texts: List[str],
    language: str = "en",
) -> Dict:
    """
    Analyze information distribution patterns:
    - Lexical diversity curves (moving TTR)
    - Word frequency distribution (hapax ratio, vocabulary richness)
    - Burstiness of rare/content words
    - Sentence length variation (CV)
    """
    # Build combined unigram model for surprisal estimation
    punct_strip = ".,;:!?\"'()[]，。；：！？""''【】"
    all_words = Counter()
    for t in human_texts + ai_texts:
        for w in _tokenize(t, language):
            w = w.lower().strip(punct_strip)
            if w:
                all_words[w] += 1
    total_words = sum(all_words.values())
    unigram_probs = {w: c / total_words for w, c in all_words.items()}

    def _analyze_corpus(texts, label):
        all_tokens = []
        sent_lengths = []
        surprisals = []
        vocab = Counter()

        for text in texts:
            words = [w.strip(punct_strip).lower() for w in _tokenize(text, language)]
            words = [w for w in words if w]
            all_tokens.extend(words)
            for w in words:
                vocab[w] += 1

            # Sentence-level
            sents = _split_sentences(text, language)
            for s in sents:
                s_words = _tokenize(s, language)
                s_words = [w for w in s_words if w.strip(punct_strip)]
                if len(s_words) >= 3:
                    sent_lengths.append(len(s_words))

            # Per-word surprisal
            for w in words:
                surprisals.append(_word_surprisal(w, unigram_probs))

        n_tokens = len(all_tokens)
        n_types = len(vocab)

        # Hapax legomena (words appearing exactly once)
        hapax = sum(1 for w, c in vocab.items() if c == 1)
        dis_legomena = sum(1 for w, c in vocab.items() if c == 2)

        # TTR curves
        ttr_values = _type_token_ratio_window(all_tokens, window=50)

        # Sentence length variation
        sl_mean = float(np.mean(sent_lengths)) if sent_lengths else 0
        sl_std = float(np.std(sent_lengths)) if sent_lengths else 0
        sl_cv = sl_std / max(sl_mean, 1)

        # Surprisal statistics
        surprisal_arr = np.array(surprisals) if surprisals else np.array([0])

        # Content word density: rough proxy via surprisal > median
        median_surp = float(np.median(surprisal_arr))
        high_info_ratio = float(np.mean(surprisal_arr > median_surp))

        # Burstiness: coefficient of variation of inter-arrival times
        # for words in top 5% surprisal
        surp_threshold = float(np.percentile(surprisal_arr, 95))
        high_surp_positions = [i for i, s in enumerate(surprisals) if s >= surp_threshold]
        if len(high_surp_positions) > 2:
            gaps = np.diff(high_surp_positions)
            burstiness_cv = float(np.std(gaps) / max(np.mean(gaps), 1))
        else:
            burstiness_cv = 0.0

        return {
            "n_tokens": n_tokens,
            "n_types": n_types,
            "ttr": round(n_types / max(n_tokens, 1), 4),
            "hapax_ratio": round(hapax / max(n_types, 1), 4),
            "dis_legomena_ratio": round(dis_legomena / max(n_types, 1), 4),
            "mean_ttr_window50": round(float(np.mean(ttr_values)), 4),
            "std_ttr_window50": round(float(np.std(ttr_values)), 4),
            "sent_length_mean": round(sl_mean, 2),
            "sent_length_std": round(sl_std, 2),
            "sent_length_cv": round(sl_cv, 4),
            "mean_surprisal": round(float(np.mean(surprisal_arr)), 4),
            "std_surprisal": round(float(np.std(surprisal_arr)), 4),
            "surprisal_skew": round(float(
                np.mean(((surprisal_arr - np.mean(surprisal_arr)) / max(np.std(surprisal_arr), 1e-8)) ** 3)
            ), 4),
            "high_info_word_ratio": round(high_info_ratio, 4),
            "rare_word_burstiness": round(burstiness_cv, 4),
        }

    human_stats = _analyze_corpus(human_texts, "human")
    ai_stats = _analyze_corpus(ai_texts, "ai")

    comparison = {}
    for key in human_stats:
        comparison[key] = {
            "human": human_stats[key],
            "ai": ai_stats[key],
            "delta": round(ai_stats[key] - human_stats[key], 4)
            if isinstance(ai_stats[key], (int, float)) else None,
        }

    return {
        "human": human_stats,
        "ai": ai_stats,
        "comparison": comparison,
    }


# ═══════════════════════════════════════════════════════════
#  Full deep analysis pipeline
# ═══════════════════════════════════════════════════════════

def run_deep_interpretability(
    samples: List[dict],
    max_samples: int = 200,
    nlp=None,
    top_k: int = 40,
    language: str = "en",
) -> Dict:
    """
    Run all 5 deep interpretability analyses on paired samples.

    Args:
        samples: list of paired samples (each has 'human_continuation',
                 'ai_continuation_chat')
        max_samples: cap on number of samples
        nlp: spaCy nlp pipeline (loaded if None; skipped for zh)
        top_k: items to report per analysis
        language: "en" or "zh"

    Returns:
        dict with keys: contrastive_ngrams, syntactic_templates,
        construction_patterns, lexical_bundles, information_density
    """
    if language != "zh" and nlp is None:
        nlp = spacy.load("en_core_web_sm")

    subset = samples[:max_samples]
    human_texts = [s["human_continuation"] for s in subset if s.get("human_continuation")]
    ai_texts = [s["ai_continuation_chat"] for s in subset if s.get("ai_continuation_chat")]
    print(f"  Deep interpretability ({language}): {len(human_texts)} human, {len(ai_texts)} AI texts")

    results = {}

    # 1. Contrastive n-grams
    print("    [1/5] Contrastive n-gram analysis...")
    results["contrastive_ngrams"] = contrastive_ngrams(
        human_texts, ai_texts, ns=(2, 3, 4, 5), min_freq=3, top_k=top_k,
        language=language,
    )
    for n_label, ng_data in results["contrastive_ngrams"].items():
        ai_top3 = ng_data["ai_distinctive"][:3]
        human_top3 = ng_data["human_distinctive"][:3]
        print(f"      {n_label} AI-ward:    {' | '.join(d['ngram'] for d in ai_top3)}")
        print(f"      {n_label} Human-ward: {' | '.join(d['ngram'] for d in human_top3)}")

    # 2. Syntactic templates (skip for Chinese — spaCy zh model is unreliable)
    if language == "zh":
        print("    [2/5] Syntactic templates — skipped for Chinese")
        results["syntactic_templates"] = {"skipped": True, "reason": "no reliable Chinese dependency parser"}
    else:
        print("    [2/5] Syntactic template extraction...")
        results["syntactic_templates"] = syntactic_templates(
            human_texts, ai_texts, nlp, top_k=top_k,
        )
        openings = results["syntactic_templates"]["sentence_openings"]
        print(f"      Sentence opening types: {len(openings)}")
        for op, data in list(openings.items())[:5]:
            print(f"        {op}: H={data['human_pct']:.1f}% AI={data['ai_pct']:.1f}% Δ={data['delta_pct']:+.1f}%")

    # 3. Construction grammar
    print("    [3/5] Construction grammar patterns...")
    results["construction_patterns"] = construction_patterns(
        human_texts, ai_texts, language=language,
    )
    sorted_constructions = sorted(
        results["construction_patterns"].items(),
        key=lambda x: abs(x[1]["ratio"] - 1) * (x[1]["human_count"] + x[1]["ai_count"]),
        reverse=True,
    )
    for name, data in sorted_constructions[:8]:
        print(f"        {name}: H={data['human_per_text']:.2f}/text AI={data['ai_per_text']:.2f}/text (×{data['ratio']:.1f})")

    # 4. Lexical bundles
    print("    [4/5] Lexical bundle detection...")
    results["lexical_bundles"] = lexical_bundles(
        human_texts, ai_texts, ns=(3, 4, 5), min_freq=3, top_k=top_k,
        language=language,
    )
    for n_label, b_data in results["lexical_bundles"].items():
        print(f"      {n_label}: AI-exclusive={len(b_data['ai_exclusive'])}, "
              f"H-exclusive={len(b_data['human_exclusive'])}, "
              f"AI-types={b_data['ai_bundle_types']}, H-types={b_data['human_bundle_types']}")

    # 5. Information density
    print("    [5/5] Information density profile...")
    results["information_density"] = information_density(
        human_texts, ai_texts, language=language,
    )
    comp = results["information_density"]["comparison"]
    for key in ["ttr", "hapax_ratio", "sent_length_cv", "mean_surprisal", "rare_word_burstiness"]:
        c = comp[key]
        print(f"        {key}: H={c['human']}, AI={c['ai']}, Δ={c['delta']}")

    return results
