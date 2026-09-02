from src.data.natural_window import NaturalLanguageWindowSplitter


def test_english_window_uses_words_and_sentence_boundaries():
    text = " ".join(
        f"Sentence {i} contains exactly five simple content words."
        for i in range(30)
    )
    splitter = NaturalLanguageWindowSplitter(
        language="en",
        target_prefix_units=30,
        target_continuation_units=50,
        tolerance=0.20,
    )
    window = splitter.split_text(text)

    assert window is not None
    assert window.length_unit == "word"
    assert 24 <= window.prefix_units <= 36
    assert 40 <= window.continuation_units <= 60
    assert window.prefix.endswith(".")
    assert window.continuation.endswith(".")


def test_chinese_window_uses_text_units_and_sentence_boundaries():
    text = "".join(
        f"这是第{i}个自然语言句子，其中包含连续的中文内容。"
        for i in range(30)
    )
    splitter = NaturalLanguageWindowSplitter(
        language="zh",
        target_prefix_units=50,
        target_continuation_units=80,
        tolerance=0.25,
    )
    window = splitter.split_text(text)

    assert window is not None
    assert window.length_unit == "zh_text_unit"
    assert 37 <= window.prefix_units <= 62
    assert 60 <= window.continuation_units <= 100
    assert window.prefix.endswith("。")
    assert window.continuation.endswith("。")


def test_long_sentence_outside_bounds_is_rejected():
    text = ("word " * 100).strip() + ". Short ending."
    splitter = NaturalLanguageWindowSplitter(
        language="en",
        target_prefix_units=20,
        target_continuation_units=20,
        tolerance=0.20,
    )
    assert splitter.split_text(text) is None


def test_unit_counts_do_not_require_a_model_tokenizer():
    en = NaturalLanguageWindowSplitter("en", 10, 20)
    zh = NaturalLanguageWindowSplitter("zh", 10, 20)

    assert en.count_units("Model-independent natural language.") == 3
    assert zh.count_units("模型无关的natural language 2026。") == 8


def test_prose_gate_accepts_ordinary_prose():
    en = NaturalLanguageWindowSplitter("en", 10, 20)
    zh = NaturalLanguageWindowSplitter("zh", 10, 20)

    assert en.prose_quality_issues(
        "The committee reviewed the evidence and published a careful response."
    ) == []
    assert zh.prose_quality_issues(
        "研究团队检查了全部证据，并对实验结果作出了谨慎解释。"
    ) == []


def test_prose_gate_rejects_structured_non_prose():
    en = NaturalLanguageWindowSplitter("en", 10, 20)
    zh = NaturalLanguageWindowSplitter("zh", 10, 20)

    assert "table_like" in en.prose_quality_issues(
        "| item | value | score | result |"
    )
    assert "json_like" in zh.prose_quality_issues(
        '{"smallUrl":"https://example.com/a","bigUrl":"x"}'
    )
    assert "reference_list" in zh.prose_quality_issues(
        "参考文献：第一条资料。第二条资料。"
    )
    assert "dot_leader_list" in en.prose_quality_issues(
        "Shell . . . . . . . . Un obus. Grenade . . . . Une grenade."
    )
    assert "list_heading" in zh.prose_quality_issues(
        "演员阵容 甲饰演角色一 乙饰演角色二 丙饰演角色三。"
    )
    assert "short_line_list" in zh.prose_quality_issues(
        "歌曲一\n歌曲二\n歌曲三\n歌曲四\n歌曲五\n歌曲六"
    )
    assert "equation_block" in en.prose_quality_issues(
        "$ x = 1 $ and $ y = 2 $; therefore z = x = y."
    )
