from cocktail_server.services.prompt_builder import (
    NEGATIVE_DEFAULT,
    build_system_prompt,
    build_user_message,
)


def test_negative_default_matches_official() -> None:
    assert NEGATIVE_DEFAULT == (
        "worst quality, low quality, score_1, score_2, score_3, artist name"
    )


def test_system_prompt_declares_anima_is_not_pony() -> None:
    assert "NOT Pony-SDXL" in build_system_prompt()


def test_system_prompt_mentions_artist_at_prefix() -> None:
    p = build_system_prompt()
    assert "@" in p
    assert "artist" in p.lower()


def test_system_prompt_prefers_gelbooru() -> None:
    assert "Gelbooru" in build_system_prompt()


def test_system_prompt_restricts_underscores_to_score_tags() -> None:
    p = build_system_prompt()
    assert "underscore" in p.lower()
    assert "score_7" in p


def test_system_prompt_forbids_realism() -> None:
    assert "realism" in build_system_prompt().lower()


def test_system_prompt_enumerates_all_safety_tags() -> None:
    p = build_system_prompt()
    for tag in ("safe", "sensitive", "nsfw", "explicit"):
        assert f'"{tag}"' in p


def test_system_prompt_requests_turnspec_json() -> None:
    p = build_system_prompt()
    assert "LlmTurnSpec" in p
    assert "reasoning" in p
    assert "tool_calls" in p
    assert "JSON" in p


def test_system_prompt_describes_aspect_ratio_presets() -> None:
    p = build_system_prompt()
    for label in ("portrait", "landscape", "square"):
        assert label in p
    assert "896" in p
    assert "1152" in p
    assert "1024" in p


def test_system_prompt_describes_cfg_presets() -> None:
    p = build_system_prompt()
    for label in ("soft", "standard", "crisp"):
        assert label in p
    # cfg の代表値
    assert "3.5" in p
    assert "4.0" in p
    assert "4.5" in p


def test_system_prompt_describes_seed_semantics() -> None:
    p = build_system_prompt()
    assert "seed" in p.lower()


def test_system_prompt_instructs_reasoning_in_japanese() -> None:
    p = build_system_prompt()
    assert "Japanese" in p
    assert "reasoning" in p


def test_user_message_embeds_instruction_verbatim() -> None:
    msg = build_user_message("ピンクの髪の猫耳少女が星空の下で微笑んでいる絵")
    assert "ピンクの髪の猫耳少女が星空の下で微笑んでいる絵" in msg
    assert "JSON" in msg


def test_system_prompt_respects_user_situation() -> None:
    p = build_system_prompt()
    # ユーザーの要求したシチュエーションを忠実に守るメタ指示が残っていること
    assert "RESPECT THE USER'S SITUATION" in p
    assert "silently downgrade" in p.lower()
    # ヌード要求は黙って服を着せるなと明記されている（語彙羅列は避け、ヌード程度に留める）
    assert "ヌード" in p
    assert "nude" in p.lower()


def test_system_prompt_allows_additive_negative() -> None:
    p = build_system_prompt()
    # ベース固定 + 追加可能方式
    assert "Always start with this base" in p
    assert "append" in p.lower()
    assert "censored" in p  # NSFW 用の追加例が残っている
    # ベース文字列自体は維持されている
    assert '"worst quality, low quality, score_1, score_2, score_3, artist name"' in p


def test_system_prompt_avoids_draw_verb_for_assistant() -> None:
    p = build_system_prompt()
    # 公開時の炎上回避: Gemma 自身の行為は「生成」と言う。「描く」は不可
    assert "生成する" in p
    assert "描く / 描きます" in p  # 禁止例として列挙されている
    # Example の reasoning も「生成しますね」に差し替わっている
    assert "生成しますね" in p
    assert "描きますね" not in p
