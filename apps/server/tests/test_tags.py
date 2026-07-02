"""語彙タグ検索(services/tags.py)の振る舞いテスト。

決定論のため実 6MB CSV ではなく、日本語エイリアス入りの小さな固定フィクスチャを使う。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from cocktail_server.config import Settings
from cocktail_server.services import tags as tags_module
from cocktail_server.services.tags import TagIndex, TagService, parse_csv

# tag,category,post_count,"aliases"。post_count は「同 tier では人気順」を非タウトロジーに
# 検証できるよう、フィクスチャの並び順とわざとずらして置いてある(smile 5M を solo 6.6M の前に)。
_FIXTURE_CSV = """\
1girl,0,7899553,"1girls,女の子,女性,少女,girl,おんなのこ"
smile,0,5000000,"笑顔,スマイル"
solo,0,6614995,"ソロ,ひとり"
looking_at_viewer,0,4000000,"カメラ目線,視聴者を見る"
long_hair,0,3000000,"長髪,ロングヘア"
hatsune_miku,4,2000000,"初音ミク,miku,ミク"
vocaloid,3,1500000,"ボーカロイド,ボカロ"
red_hair,0,1000000,"赤髪,レッドヘア"
1boy,0,900000,"男の子,少年"
masterpiece,5,800000,"傑作"
"""


def _write_csv(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "danbooru.csv"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def index(tmp_path: Path) -> TagIndex:
    return TagIndex(parse_csv(_write_csv(tmp_path, _FIXTURE_CSV)))


# --- TagIndex.search: 段構成とランキング ------------------------------------------


def test_prefix_exact_tag_ranks_first(index: TagIndex) -> None:
    # tag 名の完全一致(最上位 tier)が、同語をエイリアス prefix で持つ他候補より上に来る。
    results = index.search("1girl")
    assert results[0].tag == "1girl"


def test_same_tier_orders_by_post_count(index: TagIndex) -> None:
    # 「s」に prefix 一致するのは solo と smile（どちらも同 tier=TAG_PREFIX）。
    # フィクスチャは smile を先に並べているが、人気(post_count)で solo が上に来る。
    results = index.search("s")
    assert [r.tag for r in results] == ["solo", "smile"]


def test_japanese_alias_resolves_to_english_tag(index: TagIndex) -> None:
    # 日本語エイリアス完全一致で英語正規タグに解決し、matched に当たった元エイリアスが入る。
    results = index.search("女の子")
    assert results[0].tag == "1girl"
    assert results[0].matched == "女の子"


def test_category_filter_excludes_other_categories(index: TagIndex) -> None:
    # 1girl は category 0。character(4) で絞れば除外され、general(0) では残る。
    assert [r.tag for r in index.search("女の子", category=0)] == ["1girl"]
    assert index.search("女の子", category=4) == []


def test_tier2_substring_catches_mid_string_japanese(index: TagIndex) -> None:
    # prefix では当たらない中間一致の日本語(「目線」→「カメラ目線」)を Tier2 で拾う。
    results = index.search("目線")
    assert results[0].tag == "looking_at_viewer"
    assert results[0].matched == "カメラ目線"


def test_fuzzy_tolerates_typo(index: TagIndex) -> None:
    # prefix でも substring でもない 1 文字違い(masterpeece)を Tier3 fuzzy が拾う。
    results = index.search("masterpeece")
    assert results[0].tag == "masterpiece"


def test_fuzzy_respects_cutoff(index: TagIndex, monkeypatch: pytest.MonkeyPatch) -> None:
    # cutoff を 100 に上げる(=完全一致のみ)と、同じタイポは fuzzy 段で落ちて何も返らない。
    # 既定 cutoff では拾えていた(直前のテスト)ので、閾値が実際にゲートしていることを示す。
    monkeypatch.setattr(tags_module, "_FUZZY_CUTOFF", 100.0)
    assert index.search("masterpeece") == []


def test_index_size_reflects_row_count(index: TagIndex) -> None:
    assert index.size == 10


# --- TagService: ライフサイクル ---------------------------------------------------


def test_search_is_empty_before_index_loads() -> None:
    # 索引未構築(load 前)は空を返す — graceful degradation。
    service = TagService(Settings())
    assert service.ready is False
    assert service.search("1girl") == []


def test_service_search_after_load(tmp_path: Path) -> None:
    # load 後は索引が立ち、検索が効く。CSV を tags_dir に置いて load する。
    (tmp_path / "danbooru.csv").write_text(_FIXTURE_CSV, encoding="utf-8")
    service = TagService(Settings(tags_dir=tmp_path))
    service.load()
    assert service.ready is True
    assert service.size == 10
    assert service.search("1girl")[0].tag == "1girl"


def test_service_missing_csv_stays_empty_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CSV 不在 + 自動 DL 無効なら、索引は空のまま & 起動 warning を出す(黙って壊れない)。
    # アプリ生成時に cocktail_server ロガーは propagate=False にされる(main._configure_logging)ため、
    # root 経由で捕捉する caplog に届くよう伝播を一時復元する(製品は warning を出せている)。
    monkeypatch.setattr(logging.getLogger("cocktail_server"), "propagate", True)
    service = TagService(Settings(tags_dir=tmp_path, tags_auto_download=False))
    with caplog.at_level(logging.WARNING):
        service.load()
    assert service.ready is False
    assert "tag CSV not found" in caplog.text


# --- parse_csv: fail-loud なパース ------------------------------------------------


def test_parse_csv_skips_malformed_row_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # category が非整数の行は 0 埋めせず warning + スキップ、他の行は生かす。
    # caplog 捕捉のため伝播を一時復元(上のテストと同理由: アプリが propagate=False にする)。
    monkeypatch.setattr(logging.getLogger("cocktail_server"), "propagate", True)
    content = '1girl,0,7899553,"女の子"\nbadrow,notint,123,"x"\nsolo,0,6614995,"ソロ"\n'
    with caplog.at_level(logging.WARNING):
        rows = parse_csv(_write_csv(tmp_path, content))
    assert [r.tag for r in rows] == ["1girl", "solo"]
    assert "malformed" in caplog.text


def test_parse_csv_skips_empty_and_short_rows(tmp_path: Path) -> None:
    # 空行/列不足の行は静かに飛ばし、正常行だけ残す。
    content = '1girl,0,7899553,"女の子"\n\n,0,5,"empty tag"\nsolo,0\nred_hair,0,1000000,"赤髪"\n'
    rows = parse_csv(_write_csv(tmp_path, content))
    assert [r.tag for r in rows] == ["1girl", "red_hair"]
