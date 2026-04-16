from cocktail_server.services.prompt_builder import (
    NEGATIVE_DEFAULT,
    POSITIVE_PREFIX,
    build_system_prompt,
    build_user_message,
)


def test_positive_prefix_is_the_official_one() -> None:
    assert POSITIVE_PREFIX == "score_7, masterpiece, best quality, safe,"


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
    # スコア以外は underscore 禁止の表現がある
    assert "underscore" in p.lower()
    assert "score_7" in p


def test_system_prompt_forbids_realism() -> None:
    assert "realism" in build_system_prompt().lower()


def test_system_prompt_requests_json_only() -> None:
    p = build_system_prompt()
    assert "JSON" in p
    assert "positive" in p
    assert "negative" in p
    assert "rationale" in p


def test_user_message_embeds_instruction_verbatim() -> None:
    msg = build_user_message("ピンクの髪の猫耳少女が星空の下で微笑んでいる絵")
    assert "ピンクの髪の猫耳少女が星空の下で微笑んでいる絵" in msg
    assert "JSON" in msg
