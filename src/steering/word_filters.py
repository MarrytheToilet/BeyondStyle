"""
Centralized word/n-gram filters and semantic categorization.

Used by the interpretability plotting layer (interp_plots.py) to clean
noisy outputs (stopwords, profanity, corpus-specific news-quote artifacts)
and to tag tokens with semantic categories so that figures highlight
linguistically meaningful patterns instead of high-frequency noise.

Design goals
------------
* Pure data + small helpers. No model / torch dependencies.
* Both English and Chinese (zh) lexicons.
* Conservative: when in doubt, keep the word (return False from is_*).
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple

# ════════════════════════════════════════════════════════════════════
#  English stopwords / profanity / corpus artifacts
# ════════════════════════════════════════════════════════════════════

# Function words and ultra-generic content words that dominate any corpus
# and obscure real stylistic differences.
STOPWORDS_EN = {
    # determiners / pronouns / aux
    "the", "a", "an", "this", "that", "these", "those", "such",
    "they", "them", "their", "theirs", "we", "us", "our", "ours",
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "you", "your", "yours", "i", "me", "my", "mine", "who", "whom",
    "which", "what", "whose", "whichever", "whatever",
    # prepositions / conjunctions
    "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "about", "into", "onto", "over", "under", "after", "before",
    "between", "among", "across", "through", "during", "without",
    "within", "around", "against", "above", "below", "near",
    "and", "or", "but", "nor", "so", "yet", "if", "while", "when",
    "where", "as", "than", "though", "until", "because", "since",
    # aux/copula
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "may", "might",
    "can", "could", "must", "ought",
    # generic adverbs/quantifiers
    "very", "much", "many", "few", "some", "any", "all", "both",
    "each", "every", "either", "neither", "no", "not", "only",
    "also", "even", "still", "just", "ever", "never", "yes",
    "more", "most", "less", "least", "another", "other", "others",
    "same", "different", "next", "last", "first", "second",
    "then", "now", "here", "there", "today", "tomorrow", "tonight",
    "yesterday", "soon", "later", "always", "often", "sometimes",
    # corpus / OCR junk-ish
    "source", "via", "etc", "ie", "eg",
}

# Words to drop entirely (offensive / not appropriate to spotlight in figures).
PROFANITY_EN = {
    "fuck", "fucking", "fucked", "shit", "shitty", "bitch", "bitches",
    "ass", "asshole", "damn", "damned", "dick", "cock", "pussy",
    "bastard", "crap", "piss", "prick", "slut", "whore", "cunt",
}

# News-corpus attribution verbs that contaminate "Human-distinctive" n-grams
# because the human side is CNN/DailyMail (heavy on quoted speech).
NEWS_QUOTE_TOKENS_EN = {
    "said", "says", "told", "telling", "stating", "stated", "reports",
    "reported", "reporter", "reporters", "according",
    "spokesman", "spokeswoman", "spokesperson",
    "interview", "interviewed", "press", "statement", "officials", "official",
}

# ════════════════════════════════════════════════════════════════════
#  Chinese
# ════════════════════════════════════════════════════════════════════

STOPWORDS_ZH = {
    "的", "了", "是", "在", "和", "也", "都", "就", "还", "又",
    "或", "或者", "及", "与", "并", "并且", "但", "但是",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
    "这", "那", "这个", "那个", "这些", "那些",
    "有", "没", "没有", "不", "不是", "会", "可以", "应该",
    "把", "被", "让", "使", "给", "对", "向", "为", "于",
    "上", "下", "里", "外", "中", "之", "之后", "之前", "之中",
    "一", "一个", "一些", "一种", "一样",
    "什么", "怎么", "如何", "哪", "哪个", "哪些",
}

PROFANITY_ZH = {
    "操", "妈的", "他妈", "傻逼", "牛逼", "卧槽", "我擦",
}

NEWS_QUOTE_TOKENS_ZH = {
    "说", "称", "表示", "据报道", "据悉", "记者", "声明",
    "新闻", "报道", "采访", "接受采访",
}

# ════════════════════════════════════════════════════════════════════
#  Semantic categories — extended (paper-grade)
# ════════════════════════════════════════════════════════════════════

CATEGORIES_EN = {
    "discourse": {
        "however", "moreover", "furthermore", "nevertheless", "nonetheless",
        "therefore", "consequently", "thus", "hence", "accordingly",
        "meanwhile", "subsequently", "ultimately", "indeed", "conversely",
        "instead", "specifically", "notably", "importantly", "interestingly",
        "additionally", "essentially", "overall", "similarly", "likewise",
        "whereas", "whilst", "albeit", "whereby",
    },
    "hedging": {
        "may", "might", "could", "perhaps", "possibly", "potentially",
        "likely", "unlikely", "somewhat", "arguably", "seemingly",
        "approximately", "roughly", "tends", "appears", "presumably",
        "reportedly", "allegedly", "supposedly",
        "relatively", "generally", "typically", "often", "usually", "broadly",
    },
    "intensifier": {
        "very", "extremely", "highly", "incredibly", "remarkably",
        "significantly", "substantially", "absolutely", "utterly",
        "deeply", "profoundly", "critically", "fundamentally",
        "particularly", "especially", "notably", "exceedingly",
        "vastly", "immensely", "enormously",
    },
    "evaluative": {
        "important", "significant", "essential", "critical",
        "effective", "valuable", "remarkable", "notable", "crucial",
        "impressive", "excellent", "compelling", "promising", "pivotal",
        "vital", "key", "noteworthy", "outstanding",
    },
    "abstract_filler": {
        "landscape", "narrative", "paradigm", "framework", "dynamics",
        "implications", "multifaceted", "comprehensive", "nuanced",
        "robust", "underscore", "underscores", "navigate", "navigates",
        "foster", "fosters", "realm", "spectrum", "trajectory",
        "delve", "intricate", "tapestry", "facilitate", "leverage",
        "innovative", "evolving", "diverse", "impactful",
        "intersection", "interplay", "dimension",
    },
    "informal": {
        "gonna", "wanna", "gotta", "yeah", "okay", "ok", "stuff",
        "pretty", "really", "actually", "basically", "literally",
        "honestly", "anyway", "whatever", "totally", "super",
        "kinda", "sorta", "guys", "lol",
    },
    "causal": {
        "because", "since", "therefore", "thus", "hence",
        "causing", "resulting", "leading", "contributing",
        "driven", "influenced", "shaped", "sparked", "triggered",
        "prompted", "induced", "enabled",
    },
    "temporal": {
        "recently", "currently", "previously", "initially",
        "eventually", "gradually", "immediately", "ongoing",
        "emerging", "upcoming", "earlier", "later", "shortly",
        "henceforth", "thereafter", "hitherto",
    },
    "concrete_news": {
        "police", "officer", "officers", "victim", "victims", "court",
        "judge", "trial", "verdict", "killed", "found", "arrested",
        "charged", "sentenced", "investigation", "witness", "witnesses",
        "suspect", "evidence", "crime", "scene",
    },
}

CATEGORIES_ZH = {
    "discourse": {
        "然而", "但是", "不过", "虽然", "尽管", "因此", "所以",
        "此外", "另外", "同时", "总之", "事实上", "实际上",
        "值得注意", "换言之", "进而", "从而", "与此同时",
        "综上所述", "由此可见", "首先", "其次", "最后",
    },
    "hedging": {
        "可能", "也许", "或许", "大概", "似乎", "恐怕", "未必",
        "不一定", "往往", "通常", "一般", "相对", "据称", "据传",
    },
    "intensifier": {
        "非常", "极其", "十分", "特别", "尤其", "格外", "极为",
        "异常", "相当", "颇为", "高度", "深刻", "极大",
    },
    "evaluative": {
        "重要", "关键", "核心", "显著", "突出", "卓越", "优秀",
        "可观", "深远", "巨大", "积极", "宝贵",
    },
    "abstract_filler": {
        "层面", "维度", "框架", "范式", "格局", "态势",
        "赋能", "助力", "深耕", "聚焦", "打造", "践行",
        "多元", "创新", "高质量", "新时代", "举措", "落地",
        "生态", "纽带", "蓝图",
    },
    "informal": {
        "其实", "反正", "随便", "干嘛", "搞", "弄", "挺", "蛮",
        "啊", "呢", "嘛", "吧", "咋", "哎",
    },
    "causal": {
        "因为", "由于", "导致", "引发", "促使", "推动", "带来",
        "造成", "引起", "产生",
    },
    "temporal": {
        "目前", "当前", "近年来", "此前", "随后", "最终", "逐步",
        "近日", "日前", "未来", "即将",
    },
}


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _stop(language: str) -> set:
    return STOPWORDS_ZH if language == "zh" else STOPWORDS_EN


def _profanity(language: str) -> set:
    return PROFANITY_ZH if language == "zh" else PROFANITY_EN


def _news_quote(language: str) -> set:
    return NEWS_QUOTE_TOKENS_ZH if language == "zh" else NEWS_QUOTE_TOKENS_EN


def _categories(language: str) -> dict:
    return CATEGORIES_ZH if language == "zh" else CATEGORIES_EN


def is_clean_word(word: str, language: str = "en",
                  drop_stopwords: bool = True,
                  drop_profanity: bool = True,
                  min_len: int = 0,
                  min_freq: float = 0.0) -> bool:
    """Return True if the word should be kept for display.

    If ``min_freq`` > 0 and ``language == "en"``, also requires
    ``wordfreq.word_frequency(w, "en") >= min_freq`` to filter out
    proper nouns / training-data fragments (wordpress, mozilla, …).
    """
    if not word:
        return False
    w = word.strip()
    if not w:
        return False
    if min_len and len(w) < min_len:
        return False
    wl = w.lower() if language == "en" else w
    if drop_profanity and wl in _profanity(language):
        return False
    if drop_stopwords and wl in _stop(language):
        return False
    if min_freq > 0 and language == "en":
        try:
            from wordfreq import word_frequency
        except ImportError:
            return True
        if word_frequency(wl, "en") < min_freq:
            return False
    return True


def categorize(word: str, language: str = "en") -> str:
    """Return the semantic-category tag for a word, or '' if uncategorized."""
    if not word:
        return ""
    wl = word.lower() if language == "en" else word
    for cat, vocab in _categories(language).items():
        if wl in vocab:
            return cat
    return ""


def is_news_quote_ngram(ngram: str, language: str = "en") -> bool:
    """True if the n-gram contains a news-attribution token (drop these)."""
    if not ngram:
        return False
    quote = _news_quote(language)
    if language == "zh":
        return any(q in ngram for q in quote)
    tokens = [t.strip(".,;:!?\"'()[]") for t in ngram.split()]
    return any(t.lower() in quote for t in tokens)


def filter_word_list(items: List[dict], language: str = "en",
                     key: str = "word",
                     drop_stopwords: bool = True,
                     drop_profanity: bool = True,
                     also_drop_news_quote_words: bool = False,
                     min_len: int = 0,
                     min_freq: float = 0.0) -> List[dict]:
    """Filter a list of {word, score, category?} dicts and re-tag categories."""
    out = []
    nq = _news_quote(language)
    for item in items:
        w = item.get(key, "")
        if not is_clean_word(w, language=language,
                             drop_stopwords=drop_stopwords,
                             drop_profanity=drop_profanity,
                             min_len=min_len,
                             min_freq=min_freq):
            continue
        if also_drop_news_quote_words and (w.lower() if language == "en" else w) in nq:
            continue
        new_item = dict(item)
        if not new_item.get("category"):
            new_item["category"] = categorize(w, language=language)
        out.append(new_item)
    return out


def filter_ngram_list(items: List[dict], language: str = "en",
                      ngram_key: str = "ngram",
                      drop_news_quote: bool = True) -> List[dict]:
    """Drop n-grams contaminated by news-attribution tokens."""
    if not drop_news_quote:
        return items
    return [it for it in items
            if not is_news_quote_ngram(it.get(ngram_key, ""), language=language)]


def category_palette() -> dict:
    """Stable color per category for figure consistency.

    Keys mirror CATEGORIES_EN / CATEGORIES_ZH; "other" is the fallback.
    """
    return {
        "discourse":       "#5B8DBF",   # blue
        "hedging":         "#7DA9CF",   # lighter blue
        "intensifier":     "#D9776E",   # red-orange
        "evaluative":      "#E8A87C",   # peach
        "abstract_filler": "#C76F94",   # magenta
        "informal":        "#7DBE8A",   # green
        "causal":          "#9C7BC5",   # purple
        "temporal":        "#C9A14B",   # gold
        "concrete_news":   "#7A8B99",   # slate
        "other":           "#BBBBBB",   # grey
    }
