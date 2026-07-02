from cocktail_server.services.prompt_builder import (
    GENERATE_IMAGE_TOOL,
    NEGATIVE_DEFAULT,
    SEARCH_TAGS_TOOL,
    build_system_prompt,
    build_user_message,
    compose_negative,
)


def test_negative_default_matches_official() -> None:
    assert NEGATIVE_DEFAULT == (
        "worst quality, low quality, score_1, score_2, score_3, artist name"
    )


# --- compose_negative: 固定ベース + 追加分 ---------------------------------------


def test_compose_negative_prepends_fixed_base_when_extra_present() -> None:
    assert compose_negative("censored, bar censor") == f"{NEGATIVE_DEFAULT}, censored, bar censor"


def test_compose_negative_is_base_only_when_extra_empty() -> None:
    assert compose_negative("") == NEGATIVE_DEFAULT
    assert compose_negative("   ") == NEGATIVE_DEFAULT


# --- システムプロンプト: 会話ペルソナが主、JSON straitjacket は撤去 -----------------


def test_system_prompt_dropped_json_straitjacket() -> None:
    p = build_system_prompt()
    assert "LlmTurnSpec" not in p
    assert "Return ONLY a single JSON" not in p


def test_system_prompt_has_conversational_persona_not_corporate_ai() -> None:
    p = build_system_prompt()
    assert "対話相手" in p
    assert "優等生" in p  # "優等生AIを演じる必要はない"


def test_system_prompt_avoids_draw_verb_for_the_ai() -> None:
    p = build_system_prompt()
    # 公開時の炎上回避: AI 自身の行為は「生成」。「描く」は不可（ユーザーの「描いて」は受ける）。
    assert "生成する" in p
    assert "描く / 描きます" in p


# --- システムプロンプト: 生き残った Anima タグ規約 --------------------------------


def test_system_prompt_declares_anima_is_not_pony() -> None:
    assert "Pony-SDXL" in build_system_prompt()


def test_system_prompt_keeps_quality_prefix_and_score_tag() -> None:
    p = build_system_prompt()
    assert "score_7, masterpiece, best quality, highres," in p


def test_system_prompt_enumerates_all_safety_tags() -> None:
    p = build_system_prompt()
    for tag in ("safe", "sensitive", "nsfw", "explicit"):
        assert f'"{tag}"' in p


def test_system_prompt_mentions_artist_at_prefix_and_gelbooru() -> None:
    p = build_system_prompt()
    assert "@" in p
    assert "artist" in p.lower()
    assert "Gelbooru" in p


def test_system_prompt_restricts_underscores_and_forbids_realism() -> None:
    p = build_system_prompt()
    assert "underscore" in p.lower()
    assert "realism" in p.lower()


def test_system_prompt_respects_user_situation() -> None:
    p = build_system_prompt()
    assert "RESPECT THE USER'S SITUATION" in p
    assert "downgrade" in p.lower()
    assert "ヌード" in p
    assert "nude" in p.lower()


def test_system_prompt_describes_back_reference_rules() -> None:
    p = build_system_prompt()
    assert "[Turn N]" in p
    assert "1個前" in p
    assert "2個前" in p
    assert "current - 1" in p
    assert "current - 2" in p


def test_system_prompt_explains_negative_base_is_server_prepended() -> None:
    p = build_system_prompt()
    assert "negative_extra" in p
    # モデルには固定ベースを繰り返させない（サーバが前置する）
    assert NEGATIVE_DEFAULT in p
    assert "prepended by the server" in p


def test_system_prompt_guides_search_tags_when_unsure() -> None:
    # 綴りに確信が無いときだけ search_tags を使う指針が入っている（自信のあるタグは直接書く）。
    p = build_system_prompt()
    assert "search_tags" in p
    assert "確信" in p
    # 実機スモークで判明した過剰修飾（「ホシノ (Blue Archive)」）を避けるため短クエリを指示。
    assert "括弧" in p
    # 調べただけで止めず generate_image まで進む収束指示（空ターン回避）。
    assert "generate_image に進んで" in p


# --- generate_image ツールスキーマ（tools= で渡す） ------------------------------


def test_generate_image_tool_schema_shape() -> None:
    fn = GENERATE_IMAGE_TOOL["function"]
    assert fn["name"] == "generate_image"
    props = fn["parameters"]["properties"]
    assert set(props) == {"positive", "aspect_ratio", "seed_action", "negative_extra"}
    assert props["aspect_ratio"]["enum"] == ["portrait", "landscape", "square"]
    assert props["seed_action"]["enum"] == ["new", "keep"]
    assert fn["parameters"]["required"] == ["positive", "aspect_ratio"]


def test_neither_prompt_nor_tool_exposes_cfg_or_steps_knobs() -> None:
    # 退行ガード: CFG/steps は base/Turbo モードでサーバが決め、Gemma には振らせない。
    p = build_system_prompt()
    assert "cfg_preset" not in p
    assert "CFG PRESET" not in p
    params = GENERATE_IMAGE_TOOL["function"]["parameters"]["properties"]
    assert "cfg" not in params
    assert "steps" not in params


def test_generate_image_tool_describes_aspect_sizes() -> None:
    desc = GENERATE_IMAGE_TOOL["function"]["parameters"]["properties"]["aspect_ratio"][
        "description"
    ]
    assert "896x1152" in desc
    assert "1152x896" in desc
    assert "1024x1024" in desc


# --- search_tags ツールスキーマ（tools= で渡す） --------------------------------


def test_search_tags_tool_schema_shape() -> None:
    fn = SEARCH_TAGS_TOOL["function"]
    assert fn["name"] == "search_tags"
    props = fn["parameters"]["properties"]
    # query は必須、category は任意（返す件数はサーバが決めるので limit は持たせない）。
    assert set(props) == {"query", "category"}
    assert fn["parameters"]["required"] == ["query"]
    assert "limit" not in props


# --- build_user_message ----------------------------------------------------------


def test_user_message_embeds_instruction_without_json_directive() -> None:
    msg = build_user_message("ピンクの髪の猫耳少女", turn_index=1, is_current=True)
    assert "ピンクの髪の猫耳少女" in msg
    assert "JSON" not in msg


def test_user_message_marks_current_turn() -> None:
    assert "[Turn 3 / current]" in build_user_message("いまのお願い", turn_index=3, is_current=True)


def test_user_message_past_turn_has_no_current_marker() -> None:
    msg = build_user_message("昔のお願い", turn_index=1, is_current=False)
    assert "[Turn 1]" in msg
    assert "/ current" not in msg
