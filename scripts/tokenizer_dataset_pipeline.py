"""Deterministic HPLT 3.0 tokenizer-corpus acquisition pipeline."""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import hashlib
import heapq
import html
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import yaml
import zstandard

from artifact_io import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


LANGUAGES = ("eng_Latn", "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang")
REQUIRED_SOURCE_FIELDS = {
    "source_id",
    "source_type",
    "license",
    "homepage",
    "download_uri",
    "version_or_snapshot",
    "languages",
    "expected_files",
    "checksum_or_size",
    "enabled",
    "notes",
}
LOCK_SCHEMA_VERSION = 1
USER_AGENT = "Diesel-MT-tokenizer-dataset-fetch/1.0"
RAM_CHECKPOINT_SCHEMA_VERSION = 2
MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_BATCH_CHARACTERS = 2_000_000
MINHASH_SENTINEL = (1 << 128) - 1
HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0x20000, 0x2FA1F))
HANS_FEATURES = frozenset("这为国发后里时会个们来对过还开关门车书见东风云广台万与业严丝丧两并历丽举么义乌乐乔习乡买乱争于亏亚产亩亲亿仅从仓仪价众优伙伞伟传伤伦伪体余佣侠侣侥侧侦俩债倾偿党兰兴养兽冈册写军农冲决况冻净减凑击划刘则刚创删别制剂剑动务势劳励劲区医华协单卖卢卫却厂厅历压厌厕县叁参双变叙叶号叹吗吨听启吴呐员呛呜咏咙哑响哟唤啧喷嘱团园围图圆圣场坏块坚坛坝坞坟坠垄垒垦执扩扫扬扰抚抛抢护报担拟拥拦拧拨择挂挚损换据掳掺携摄摆摇撑数斋斗断无旧显晋晓暂术机杀杂权条来杨杰松板极构枣枪柜柠树样桥档梦检椭楼欢欧歼殁毁毂毕气汉汤沟没泽洁浅浆浇浊测济浓涛涝涡涣润涧涨涩淀渊渔渗温湾湿溃溅滚满滤滥滨滩潜潴灯灵灾灿炉炼烁烂烛烟烦烧烫热爱爷牵犹狭狮独狱猎猪猫献环现琐电画畅疗疟疡疮疯痈痉痒痨痪瘾皑皱盘着矾矿码砖砚砾础确碍礼祷祸离种积称税稳穷窃窍笔笋笼筑签简粮纠纤约级纪纬纯纱纲纳纵纷纸纹纺纽线练组细织终绍经绑绒结绕绘给络绝统继绩绪续绰绳维绵绷绸综绿缀缉缎缓编缘缚缝缠缨缩缴网罗罚职联聪肃肠肤肿胀胁胆胜胶脉脏脑脚脱脸腻腾舆舰艺节芜苇苍苏苹范茎荐药莱莲获莹营萧萨葱蓝虑虚虫虽虾蚀蚁蚂蛊蛮补衬袜袭装裤见观规觅视览觉触誉计订认讥讨让训议讯记讲讳讴许论讼设访诀证评诅识诈诉诊词译试诗诚话诞询该详语误诱说请诸诺读课谁调谈谊谋谍谎谓谜谢谨谱贝贞负贡财责贤败账货质贩贫贬购贯贱贴贵贷贸费贺贼贾资赋赌赎赏赐赔赖赞赵赶趋跃践踊踪车轨轩转轮软轰辆输辙辞辩边辽达迁过迈运还进远违连迟适选递逻遗邮邻郑酝酱采释里鉴针钉钓钙钝钞钟钢钥钦钧钩钱钳钻铁铃铅铜铝铲银铺链销锁锅锋锐错锡锦锨锭键锯锻镇镜长门闭问闯闲间闷闸闹闻阁阅阔队阳阴阵阶际陆陈陕险随隐隶难雏鸡鸣鸦鸭鸽鹅鹏鹰麦黄点齐齿龄龙龟")
HANT_FEATURES = frozenset("這為國發後裡時會個們來對過還開關門車書見東風雲廣臺萬與業嚴絲喪兩並歷麗舉麼義烏樂喬習鄉買亂爭於虧亞產畝親億僅從倉儀價眾優夥傘偉傳傷倫偽體餘傭俠侶僥側偵倆債傾償黨蘭興養獸岡冊寫軍農衝決況凍淨減湊擊劃劉則剛創刪別製劑劍動務勢勞勵勁區醫華協單賣盧衛卻廠廳壓厭廁縣參雙變敘葉號嘆嗎噸聽啟吳吶員嗆嗚詠嚨啞響喲喚嘖噴囑團園圍圖圓聖場壞塊堅壇壩塢墳墜壟壘墾執擴掃揚擾撫拋搶護報擔擬擁攔擰撥擇掛摯損換據擄摻攜攝擺搖撐數齋鬥斷無舊顯晉曉暫術機殺雜權條來楊傑鬆板極構棗槍櫃檸樹樣橋檔夢檢橢樓歡歐殲歿毀轂畢氣漢湯溝沒澤潔淺漿澆濁測濟濃濤澇渦渙潤澗漲澀澱淵漁滲溫灣濕潰濺滾滿濾濫濱灘潛瀦燈靈災燦爐煉爍爛燭煙煩燒燙熱愛爺牽猶狹獅獨獄獵豬貓獻環現瑣電畫暢療瘧瘍瘡瘋癰痙癢癆瘓癮皚皺盤著礬礦碼磚硯礫礎確礙禮禱禍離種積稱稅穩窮竊竅筆筍籠築簽簡糧糾纖約級紀緯純紗綱納縱紛紙紋紡紐線練組細織終紹經綁絨結繞繪給絡絕統繼績緒續綽繩維綿繃綢綜綠綴緝緞緩編緣縛縫纏纓縮繳網羅罰職聯聰肅腸膚腫脹脅膽勝膠脈臟腦腳脫臉膩騰輿艦藝節蕪葦蒼蘇蘋範莖薦藥萊蓮獲瑩營蕭薩蔥藍慮虛蟲雖蝦蝕蟻螞蠱蠻補襯襪襲裝褲見觀規覓視覽覺觸譽計訂認譏討讓訓議訊記講諱謳許論訟設訪訣證評詛識詐訴診詞譯試詩誠話誕詢該詳語誤誘說請諸諾讀課誰調談誼謀諜謊謂謎謝謹譜貝貞負貢財責賢敗賬貨質販貧貶購貫賤貼貴貸貿費賀賊賈資賦賭贖賞賜賠賴贊趙趕趨躍踐踴蹤車軌軒轉輪軟轟輛輸轍辭辯邊遼達遷過邁運還進遠違連遲適選遞邏遺郵鄰鄭醞醬採釋裡鑒針釘釣鈣鈍鈔鐘鋼鑰欽鈞鉤錢鉗鑽鐵鈴鉛銅鋁鏟銀鋪鏈銷鎖鍋鋒銳錯錫錦鍁錠鍵鋸鍛鎮鏡長門閉問闖閒間悶閘鬧聞閣閱闊隊陽陰陣階際陸陳陝險隨隱隸難雛雞鳴鴉鴨鴿鵝鵬鷹麥黃點齊齒齡龍龜")

# Worker configuration is installed once by ProcessPoolExecutor.initializer.
# The sequential path uses the same initializer and worker function so both
# execution modes exercise identical content logic.
_WORKER_LANGUAGE: str | None = None
_WORKER_CLEANING: dict[str, Any] | None = None
_WORKER_DEDUP: dict[str, Any] | None = None
_WORKER_SEED = 0
_WORKER_SCRIPT_THRESHOLD = 0.0


class PipelineError(RuntimeError):
    exit_code = 5


class ConfigError(PipelineError):
    exit_code = 2


class LockError(PipelineError):
    exit_code = 3


class FetchError(PipelineError):
    exit_code = 4


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot load config {path}: {exc}") from exc
    validate_config(value)
    return value


def validate_config(config: Any) -> None:
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ConfigError("config.schema_version must be 1")
    for key in ("dataset", "reproducibility", "profiles", "cleaning", "deduplication", "holdout", "quality", "sources"):
        if key not in config:
            raise ConfigError(f"config missing required field: {key}")
    dataset = config["dataset"]
    for key in ("name", "version_or_snapshot", "homepage", "license", "terms_url", "base_uri", "quality_wds", "max_shards_per_language", "shard_selection"):
        if key not in dataset or dataset[key] in (None, ""):
            raise ConfigError(f"dataset missing required field: {key}")
    if not (str(dataset["homepage"]).startswith("https://") and str(dataset["base_uri"]).startswith("https://")):
        raise ConfigError("dataset homepage and base_uri must use HTTPS")
    minimum_wds = int(dataset["quality_wds"].get("minimum", -1))
    maximum_wds = int(dataset["quality_wds"].get("maximum", -1))
    if minimum_wds < 0 or maximum_wds < minimum_wds or int(dataset["max_shards_per_language"]) <= 0:
        raise ConfigError("dataset WDS range and max_shards_per_language must be positive and ordered")
    if dataset["shard_selection"] != "one-per-wds-descending":
        raise ConfigError("dataset.shard_selection must be one-per-wds-descending")
    for key in ("contract", "canonical_json", "content_hash", "output_encoding", "output_newline", "final_trailing_newline"):
        if key not in config["reproducibility"]:
            raise ConfigError(f"reproducibility missing required field: {key}")
    outputs: set[str] = set()
    source_ids: set[str] = set()
    for index, source in enumerate(config["sources"]):
        missing = REQUIRED_SOURCE_FIELDS - set(source)
        if missing:
            raise ConfigError(f"sources[{index}] missing fields: {', '.join(sorted(missing))}")
        if not str(source["source_id"]).strip() or source["source_id"] in source_ids:
            raise ConfigError(f"sources[{index}] has empty or duplicate source_id")
        source_ids.add(source["source_id"])
        if source["enabled"]:
            for key in ("license", "homepage", "download_uri", "version_or_snapshot"):
                if not str(source[key]).strip():
                    raise ConfigError(f"enabled source {source['source_id']} has empty {key}")
            mapping = source["languages"]
            if not isinstance(mapping, dict) or not mapping.get("source") or not mapping.get("output"):
                raise ConfigError(f"enabled source {source['source_id']} has invalid language mapping")
            if mapping["output"] in outputs:
                raise ConfigError(f"duplicate enabled output language: {mapping['output']}")
            if not str(source["download_uri"]).startswith("https://"):
                raise ConfigError(f"enabled source {source['source_id']} download_uri must use HTTPS")
            outputs.add(mapping["output"])
    if outputs != set(LANGUAGES):
        raise ConfigError(f"enabled sources must map exactly to {', '.join(LANGUAGES)}")
    for name in ("smoke", "mvp"):
        profile = config["profiles"].get(name)
        if not isinstance(profile, dict):
            raise ConfigError(f"missing profile: {name}")
        for key in ("enabled_languages", "character_budget_per_language", "locked_prefix_bytes_per_shard", "random_seed", "concurrency", "corpus_subdir", "holdout_subdir"):
            if key not in profile:
                raise ConfigError(f"profile {name} missing {key}")
        if set(profile["enabled_languages"]) != set(LANGUAGES):
            raise ConfigError(f"profile {name} must enable exactly five project languages")
        if (
            int(profile["character_budget_per_language"]) <= 0
            or int(profile["locked_prefix_bytes_per_shard"]) <= 0
            or int(profile["concurrency"]) <= 0
        ):
            raise ConfigError(f"profile {name} budgets must be positive")
        for path_key in ("corpus_subdir", "holdout_subdir"):
            output_path = Path(str(profile[path_key]))
            if output_path.is_absolute() or ".." in output_path.parts:
                raise ConfigError(f"profile {name} {path_key} must be a relative path without parent traversal")
        if profile["corpus_subdir"] == profile["holdout_subdir"]:
            raise ConfigError(f"profile {name} train and holdout directories must differ")
        per_language = profile.get("locked_prefix_bytes_by_language", {})
        if set(per_language) - set(LANGUAGES) or any(int(value) <= 0 for value in per_language.values()):
            raise ConfigError(f"profile {name} has invalid per-language locked prefix bytes")
    cleaning = config["cleaning"]
    for key in (
        "algorithm_version",
        "min_characters",
        "max_characters",
        "max_utf8_bytes",
        "max_replacement_character_ratio",
        "max_control_character_ratio",
        "min_visible_character_ratio",
        "max_numeric_symbol_ratio",
        "max_url_email_ratio",
        "max_long_token_characters",
        "max_repeated_character_run",
        "reject_mojibake_pattern",
        "reject_mechanical_repetition_pattern",
        "reject_encoded_blob_pattern",
        "reject_html_pattern",
        "reject_template_pattern",
        "reject_keyword_stuffing_pattern",
        "max_keyword_stuffing_matches",
        "reject_repeated_spam_pattern",
        "max_repeated_spam_matches",
        "split_on_lines",
        "collapse_horizontal_whitespace",
    ):
        if key not in cleaning:
            raise ConfigError(f"cleaning missing required field: {key}")
    if (
        int(cleaning["min_characters"]) <= 0
        or int(cleaning["max_characters"]) < int(cleaning["min_characters"])
        or int(cleaning["max_utf8_bytes"]) < int(cleaning["min_characters"])
        or int(cleaning["max_long_token_characters"]) <= 0
        or int(cleaning["max_repeated_character_run"]) <= 1
    ):
        raise ConfigError("cleaning character limits must be positive and ordered")
    for ratio_key in (
        "max_replacement_character_ratio",
        "max_control_character_ratio",
        "min_visible_character_ratio",
        "max_numeric_symbol_ratio",
        "max_url_email_ratio",
    ):
        if not 0.0 <= float(cleaning[ratio_key]) <= 1.0:
            raise ConfigError(f"cleaning.{ratio_key} must be in [0, 1]")
    if int(cleaning["max_keyword_stuffing_matches"]) < 0:
        raise ConfigError("cleaning.max_keyword_stuffing_matches cannot be negative")
    if int(cleaning["max_repeated_spam_matches"]) < 0:
        raise ConfigError("cleaning.max_repeated_spam_matches cannot be negative")
    for pattern_name in (
        "reject_html_pattern",
        "reject_template_pattern",
        "reject_keyword_stuffing_pattern",
        "reject_repeated_spam_pattern",
        "reject_mojibake_pattern",
        "reject_mechanical_repetition_pattern",
        "reject_encoded_blob_pattern",
    ):
        try:
            re.compile(str(cleaning[pattern_name]))
        except re.error as exc:
            raise ConfigError(f"cleaning.{pattern_name} is not a valid regular expression: {exc}") from exc
    dedup = config["deduplication"]
    if set(dedup.get("approximate_languages", [])) != set(LANGUAGES):
        raise ConfigError("deduplication.approximate_languages must cover all five project languages")
    minhash = dedup.get("minhash", {})
    for key in ("token_unit", "ngram_size", "hash", "permutations", "bands", "threshold", "seed", "max_ngrams_per_text", "tie_break"):
        if key not in minhash:
            raise ConfigError(f"deduplication.minhash missing required field: {key}")
    if (
        int(minhash["ngram_size"]) <= 0
        or int(minhash["permutations"]) <= 0
        or int(minhash["bands"]) <= 0
        or int(minhash["permutations"]) % int(minhash["bands"]) != 0
        or not 0.0 <= float(minhash["threshold"]) <= 1.0
        or int(minhash["max_ngrams_per_text"]) <= 0
    ):
        raise ConfigError("deduplication.minhash parameters are invalid")
    holdout = config["holdout"]
    if (
        not 0.02 <= float(holdout.get("fraction", 0.0)) <= 0.05
        or holdout.get("split_unit") != "document"
        or holdout.get("split_key") != "source-url-or-document-content"
        or holdout.get("cross_split_deduplication") != "exact-and-minhash"
        or not isinstance(holdout.get("seed"), int)
    ):
        raise ConfigError("holdout must configure a deterministic 2%-5% document split with exact-and-minhash isolation")
    quality = config["quality"]
    thresholds = quality.get("language_min_script_ratio", {})
    if set(thresholds) != set(LANGUAGES) or any(not 0.0 <= float(value) <= 1.0 for value in thresholds.values()):
        raise ConfigError("quality language script-ratio thresholds must cover exactly five languages")
    if int(quality.get("review_sample_count", 0)) < 200:
        raise ConfigError("quality.review_sample_count must be at least 200")
    if int(quality.get("rejected_review_sample_count", 0)) <= 0:
        raise ConfigError("quality.rejected_review_sample_count must be positive")
    if int(quality.get("excerpt_characters", 0)) < 40:
        raise ConfigError("quality.excerpt_characters must be at least 40")
    if not 0.0 < float(quality.get("max_domain_fraction", 0.0)) <= 0.1:
        raise ConfigError("quality.max_domain_fraction must be in (0, 0.1]")
    domain_overrides = quality.get("max_domain_fraction_by_language", {})
    if (
        not isinstance(domain_overrides, dict)
        or set(domain_overrides) - set(LANGUAGES)
        or any(not 0.0 < float(value) <= 0.1 for value in domain_overrides.values())
    ):
        raise ConfigError("quality.max_domain_fraction_by_language has invalid languages or values")
    if not 1.05 <= float(quality.get("candidate_oversampling_factor", 0.0)) <= 10.0:
        raise ConfigError("quality.candidate_oversampling_factor must be in [1.05, 10]")
    if int(config["quality"].get("review_sample_count", 0)) <= 0:
        raise ConfigError("quality.review_sample_count must be positive")


def enabled_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted((s for s in config["sources"] if s["enabled"]), key=lambda s: s["languages"]["output"])


def request_bytes(url: str, *, timeout: int, retries: int = 3, headers: dict[str, str] | None = None) -> bytes:
    merged = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    merged.update(headers or {})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=merged), timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise FetchError(f"network request failed for {url}: {last}")


def remote_size(url: str, *, timeout: int = 60) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            size = response.headers.get("Content-Length")
    except (OSError, urllib.error.URLError) as exc:
        raise FetchError(f"HEAD failed for {url}: {exc}") from exc
    if not size or not size.isdigit():
        raise FetchError(f"remote Content-Length missing for {url}")
    return int(size)


def parse_map(data: bytes, minimum_wds: int, maximum_wds: int) -> list[tuple[int, int, str]]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LockError(f"map is not UTF-8: {exc}") from exc
    parsed: list[tuple[int, int, str]] = []
    pattern = re.compile(r"/(\d+)_(\d+)\.jsonl\.zst$")
    for line in lines:
        line = line.strip()
        match = pattern.search(line)
        if not match:
            continue
        wds, shard = map(int, match.groups())
        if minimum_wds <= wds <= maximum_wds:
            parsed.append((wds, shard, line))
    parsed.sort(key=lambda item: (-item[0], item[1], item[2]))
    if not parsed:
        raise LockError("map contains no shard in configured WDS range")
    return parsed


def select_stratified_shards(
    parsed: list[tuple[int, int, str]],
    count: int,
) -> list[tuple[int, int, str]]:
    """Select across WDS tiers before taking another shard from any tier."""
    if count <= 0:
        raise ValueError("shard count must be positive")
    selected: list[tuple[int, int, str]] = []
    selected_urls: set[str] = set()
    by_wds: dict[int, list[tuple[int, int, str]]] = {}
    for item in parsed:
        by_wds.setdefault(item[0], []).append(item)
    for wds in sorted(by_wds, reverse=True):
        item = by_wds[wds][0]
        selected.append(item)
        selected_urls.add(item[2])
        if len(selected) == count:
            return selected
    for item in parsed:
        if item[2] not in selected_urls:
            selected.append(item)
            selected_urls.add(item[2])
            if len(selected) == count:
                break
    return selected


def parse_md5_list(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in data.decode("ascii").splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]{32}", parts[0]):
            result[parts[1].replace("\\", "/")] = parts[0].lower()
    return result


def cache_path_for(cache_root: Path, source_id: str, url: str) -> Path:
    name = url.rsplit("/", 1)[-1]
    return cache_root / "hplt3" / source_id / f"{name}.prefix"


def download_locked_prefix(url: str, path: Path, target_bytes: int, *, timeout: int = 120, retries: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if path.exists():
        current = path.stat().st_size
        if current == target_bytes:
            return
        if current < target_bytes:
            with contextlib.suppress(FileNotFoundError):
                partial.unlink()
            os.replace(path, partial)
        else:
            path.unlink()
    last: Exception | None = None
    for attempt in range(retries):
        start = partial.stat().st_size if partial.exists() else 0
        if start > target_bytes:
            partial.unlink()
            start = 0
        if start == target_bytes:
            os.replace(partial, path)
            return
        headers = {
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{target_bytes - 1}",
        }
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                if start and response.status != 206:
                    raise FetchError(f"server ignored resume range for {url}")
                mode = "ab" if start else "wb"
                with partial.open(mode) as handle:
                    remaining = target_bytes - start
                    while remaining:
                        chunk = response.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if partial.stat().st_size == target_bytes:
                os.replace(partial, path)
                return
            raise FetchError(f"short download for {url}: {partial.stat().st_size}/{target_bytes} bytes")
        except (OSError, urllib.error.URLError, FetchError) as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise FetchError(f"download failed for {url}: {last}")


def resolve_lock(config: dict[str, Any], config_path: Path, lock_path: Path, cache_root: Path, profile_name: str) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    dataset = config["dataset"]
    selected_count = int(dataset["max_shards_per_language"])
    source_locks: list[dict[str, Any]] = []
    for source in enabled_sources(config):
        map_uri = source["download_uri"]
        map_data = request_bytes(map_uri, timeout=60)
        metadata_cache = cache_root / "hplt3" / source["source_id"] / "metadata"
        atomic_write_bytes(metadata_cache / f"{source['languages']['source']}.map", map_data)
        parsed = parse_map(map_data, int(dataset["quality_wds"]["minimum"]), int(dataset["quality_wds"]["maximum"]))
        source_lang = source["languages"]["source"]
        md5_uri = f"{dataset['base_uri']}/{source_lang}.md5"
        md5_data = request_bytes(md5_uri, timeout=60)
        atomic_write_bytes(metadata_cache / f"{source_lang}.md5", md5_data)
        md5s = parse_md5_list(md5_data)
        shards: list[dict[str, Any]] = []
        selected = select_stratified_shards(parsed, selected_count)
        for order, (wds, shard_number, url) in enumerate(selected):
            full_size = remote_size(url)
            per_language = profile.get("locked_prefix_bytes_by_language", {})
            requested_bytes = int(per_language.get(source["languages"]["output"], profile["locked_prefix_bytes_per_shard"]))
            locked_bytes = min(full_size, requested_bytes)
            cache_path = cache_path_for(cache_root, source["source_id"], url)
            download_locked_prefix(url, cache_path, locked_bytes)
            relative_name = f"{source_lang}/{url.rsplit('/', 1)[-1]}"
            upstream_md5 = md5s.get(relative_name)
            if not upstream_md5:
                raise LockError(f"official MD5 missing for {relative_name}")
            shards.append(
                {
                    "logical_order": order,
                    "wds": wds,
                    "shard_number": shard_number,
                    "url": url,
                    "remote_size": full_size,
                    "upstream_md5": upstream_md5,
                    "locked_bytes": locked_bytes,
                    "sha256": sha256_file(cache_path),
                    "sha256_scope": "bytes=0-(locked_bytes-1)",
                }
            )
        source_locks.append(
            {
                "source_id": source["source_id"],
                "source_language": source_lang,
                "output_language": source["languages"]["output"],
                "map_uri": map_uri,
                "map_sha256": sha256_bytes(map_data),
                "md5_uri": md5_uri,
                "shards": shards,
            }
        )
    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "dataset_name": dataset["name"],
        "version_or_snapshot": dataset["version_or_snapshot"],
        "resolved_profile": profile_name,
        "quality_wds": dataset["quality_wds"],
        "config_sha256": sha256_file(config_path),
        "sources": sorted(source_locks, key=lambda item: item["output_language"]),
    }
    atomic_write_bytes(lock_path, canonical_json_bytes(lock))
    return lock


def load_lock(path: Path, config: dict[str, Any], profile_name: str, config_path: Path | None = None) -> dict[str, Any]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockError(f"cannot load source lock {path}: {exc}") from exc
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise LockError("source lock schema_version must be 1")
    if lock.get("dataset_name") != config["dataset"]["name"] or lock.get("version_or_snapshot") != config["dataset"]["version_or_snapshot"]:
        raise LockError("source lock dataset identity differs from config")
    if lock.get("quality_wds") != config["dataset"]["quality_wds"]:
        raise LockError("source lock WDS range differs from config")
    if config_path is not None and lock.get("config_sha256") != sha256_file(config_path):
        raise LockError("source lock config SHA-256 differs from the selected config file")
    if len(lock.get("sources", [])) != len(LANGUAGES):
        raise LockError("source lock must contain exactly five enabled sources")
    if lock["sources"] != sorted(lock["sources"], key=lambda item: item.get("output_language", "")):
        raise LockError("source lock sources must be sorted by output_language")
    profile = config["profiles"][profile_name]
    registry = {source["source_id"]: source for source in enabled_sources(config)}
    outputs: set[str] = set()
    for source in lock["sources"]:
        for key in ("source_id", "source_language", "output_language", "map_uri", "map_sha256", "shards"):
            if not source.get(key):
                raise LockError(f"source lock entry missing {key}")
        configured = registry.get(source["source_id"])
        if configured is None:
            raise LockError(f"source lock contains unknown or disabled source: {source['source_id']}")
        if source["source_language"] != configured["languages"]["source"] or source["output_language"] != configured["languages"]["output"]:
            raise LockError(f"source lock language mapping differs from config: {source['source_id']}")
        if source["output_language"] in outputs:
            raise LockError(f"source lock has duplicate output language: {source['output_language']}")
        outputs.add(source["output_language"])
        if not re.fullmatch(r"[0-9a-f]{64}", source["map_sha256"]):
            raise LockError(f"invalid map SHA-256: {source['source_id']}")
        if not source["shards"]:
            raise LockError(f"source lock has no shards: {source['source_id']}")
        if len(source["shards"]) > int(config["dataset"]["max_shards_per_language"]):
            raise LockError(f"source lock has too many shards: {source['source_id']}")
        logical_orders = [int(shard.get("logical_order", -1)) for shard in source["shards"]]
        if logical_orders != list(range(len(source["shards"]))):
            raise LockError(f"source lock shard order is not contiguous: {source['source_id']}")
        for shard in source["shards"]:
            for key in ("logical_order", "wds", "shard_number", "url", "remote_size", "upstream_md5", "locked_bytes", "sha256", "sha256_scope"):
                if key not in shard or shard[key] in (None, ""):
                    raise LockError(f"locked shard missing {key}: {source['source_id']}")
            if not (int(config["dataset"]["quality_wds"]["minimum"]) <= int(shard["wds"]) <= int(config["dataset"]["quality_wds"]["maximum"])):
                raise LockError(f"locked shard WDS is outside config range: {source['source_id']}")
            if int(shard["remote_size"]) <= 0 or int(shard["locked_bytes"]) <= 0 or int(shard["locked_bytes"]) > int(shard["remote_size"]):
                raise LockError(f"locked shard has invalid sizes: {source['source_id']}")
            if not re.fullmatch(r"[0-9a-f]{32}", str(shard["upstream_md5"])):
                raise LockError(f"invalid upstream MD5: {source['source_id']}")
            required_bytes = int(profile.get("locked_prefix_bytes_by_language", {}).get(source["output_language"], profile["locked_prefix_bytes_per_shard"]))
            if int(shard["locked_bytes"]) < min(int(shard["remote_size"]), required_bytes):
                raise LockError(f"lock resolved for too small a prefix for profile {profile_name}: {source['source_id']}")
            if not re.fullmatch(r"[0-9a-f]{64}", shard["sha256"]):
                raise LockError(f"invalid shard SHA-256: {source['source_id']}")
            if shard["sha256_scope"] != "bytes=0-(locked_bytes-1)":
                raise LockError(f"invalid SHA-256 scope: {source['source_id']}")
    if outputs != set(LANGUAGES):
        raise LockError("source lock must map exactly the five project languages")
    return lock


def ensure_cache(lock: dict[str, Any], cache_root: Path, *, offline: bool, use_cache: bool) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for source in lock["sources"]:
        for shard in sorted(source["shards"], key=lambda item: item["logical_order"]):
            path = cache_path_for(cache_root, source["source_id"], shard["url"])
            valid = path.exists() and path.stat().st_size == int(shard["locked_bytes"]) and sha256_file(path) == shard["sha256"]
            if not valid:
                if path.exists():
                    path.unlink()
                if offline or use_cache:
                    raise FetchError(f"validated cache missing or corrupt for {source['source_id']} shard {shard['url']}")
                download_locked_prefix(shard["url"], path, int(shard["locked_bytes"]))
                if sha256_file(path) != shard["sha256"]:
                    path.unlink(missing_ok=True)
                    raise FetchError(f"downloaded prefix SHA-256 differs from source lock: {shard['url']}")
            result[f"{source['source_id']}:{shard['logical_order']}"] = path
    return result


def iter_jsonl_zst_prefix(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="strict", newline="")
        try:
            for line in text:
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        yield value
        except (zstandard.ZstdError, UnicodeDecodeError):
            # A lock may intentionally end before the remote zstd frame. Complete
            # JSONL records yielded before the boundary are the locked input.
            return
        finally:
            with contextlib.suppress(Exception):
                text.detach()


def _bounded_units(text: str, rules: dict[str, Any]) -> Iterator[str]:
    """Split on line/sentence/space boundaries under Unicode and UTF-8 caps."""
    initial = text.splitlines() if rules["split_on_lines"] else [text]
    if not initial:
        initial = [""]
    maximum_characters = int(rules["max_characters"])
    maximum_bytes = int(rules["max_utf8_bytes"])
    for raw in initial:
        value = html.unescape(raw.replace("\u00a0", " "))
        if rules["collapse_horizontal_whitespace"]:
            value = re.sub(r"[\t\v\f \u2000-\u200b\u202f\u205f\u3000]+", " ", value)
        value = value.strip()
        if not value:
            yield ""
            continue
        while value:
            upper = min(len(value), maximum_characters)
            while upper > 1 and len(value[:upper].encode("utf-8")) > maximum_bytes:
                upper = max(1, upper * maximum_bytes // len(value[:upper].encode("utf-8")))
            while upper < len(value) and len(value[: upper + 1].encode("utf-8")) <= maximum_bytes and upper < maximum_characters:
                upper += 1
            if upper < len(value):
                floor = max(int(upper * 0.55), int(rules["min_characters"]))
                boundary = 0
                for match in re.finditer(r"[.!?。！？；;](?:\s+|$)|\s+", value[:upper]):
                    if match.end() >= floor:
                        boundary = match.end()
                if boundary:
                    upper = boundary
            unit = value[:upper].strip()
            if unit:
                yield unit
            value = value[upper:].strip()


def _matched_character_ratio(pattern: re.Pattern[str], text: str) -> float:
    matched = sum(match.end() - match.start() for match in pattern.finditer(text))
    return matched / len(text) if text else 0.0


def _clean_units(text: Any, rules: dict[str, Any]) -> Iterator[tuple[str | None, str | None, str]]:
    if not isinstance(text, str):
        yield None, "missing_text", ""
        return
    html_re = re.compile(rules["reject_html_pattern"])
    template_re = re.compile(rules["reject_template_pattern"])
    keyword_stuffing_re = re.compile(rules["reject_keyword_stuffing_pattern"])
    repeated_spam_re = re.compile(rules["reject_repeated_spam_pattern"])
    mojibake_re = re.compile(rules["reject_mojibake_pattern"])
    repetition_re = re.compile(rules["reject_mechanical_repetition_pattern"])
    encoded_blob_re = re.compile(rules["reject_encoded_blob_pattern"])
    url_email_re = re.compile(r"(?i)(?:https?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
    repeated_character_re = re.compile(r"(.)\1{" + str(int(rules["max_repeated_character_run"]) - 1) + r",}")
    for value in _bounded_units(text, rules):
        excerpt = value
        if not value:
            yield None, "empty", excerpt
            continue
        if len(value) < int(rules["min_characters"]):
            yield None, "too_short", excerpt
            continue
        if len(value) > int(rules["max_characters"]) or len(value.encode("utf-8")) > int(rules["max_utf8_bytes"]):
            yield None, "segment_bound_exceeded", excerpt
            continue
        replacement_ratio = value.count("\ufffd") / len(value)
        if replacement_ratio > float(rules["max_replacement_character_ratio"]):
            yield None, "replacement_characters", excerpt
            continue
        control_count = sum(unicodedata.category(char) in {"Cc", "Cs"} for char in value)
        if control_count / len(value) > float(rules["max_control_character_ratio"]):
            yield None, "control_characters", excerpt
            continue
        visible = sum(not unicodedata.category(char).startswith("C") for char in value)
        if visible / len(value) < float(rules["min_visible_character_ratio"]):
            yield None, "low_visible_ratio", excerpt
            continue
        if html_re.search(value):
            yield None, "html_residue", excerpt
            continue
        if template_re.search(value):
            yield None, "template_residue", excerpt
            continue
        if mojibake_re.search(value):
            yield None, "mojibake", excerpt
            continue
        if encoded_blob_re.search(value):
            yield None, "encoded_blob", excerpt
            continue
        if repeated_character_re.search(value) or repetition_re.search(value):
            yield None, "mechanical_repetition", excerpt
            continue
        non_space = [char for char in value if not char.isspace()]
        numeric_symbol = sum(unicodedata.category(char)[0] in {"N", "P", "S"} for char in non_space)
        if non_space and numeric_symbol / len(non_space) > float(rules["max_numeric_symbol_ratio"]):
            yield None, "numeric_symbol_dominance", excerpt
            continue
        if _matched_character_ratio(url_email_re, value) > float(rules["max_url_email_ratio"]):
            yield None, "url_email_dominance", excerpt
            continue
        for token in value.split():
            if len(token) > int(rules["max_long_token_characters"]):
                ascii_ratio = sum(ord(char) < 128 for char in token) / len(token)
                if ascii_ratio >= 0.70:
                    yield None, "long_mechanical_token", excerpt
                    break
        else:
            if sum(1 for _ in keyword_stuffing_re.finditer(value)) > int(rules["max_keyword_stuffing_matches"]):
                yield None, "keyword_stuffing", excerpt
                continue
            if sum(1 for _ in repeated_spam_re.finditer(value)) > int(rules["max_repeated_spam_matches"]):
                yield None, "keyword_stuffing", excerpt
                continue
            yield value, None, excerpt
            continue


def split_and_clean(text: Any, rules: dict[str, Any]) -> Iterator[tuple[str | None, str | None]]:
    """Public compatibility wrapper returning cleaned text or a reason code."""
    for cleaned, reason, _excerpt in _clean_units(text, rules):
        yield cleaned, reason


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def one_permutation_minhash(text: str, parameters: dict[str, Any]) -> tuple[int, ...]:
    n = int(parameters["ngram_size"])
    permutations = int(parameters["permutations"])
    seed = int(parameters["seed"])
    max_ngrams = int(parameters["max_ngrams_per_text"])
    compact = re.sub(r"\s+", " ", text)
    count = max(0, len(compact) - n + 1)
    if count <= max_ngrams:
        positions = range(count)
    else:
        # Bound CPU for very long web segments while retaining deterministic
        # coverage across the entire text instead of truncating to a prefix.
        positions = (index * count // max_ngrams for index in range(max_ngrams))
    tokens = {compact[index : index + n] for index in positions}
    key = seed.to_bytes(16, "little", signed=False)
    bins = [-1] * permutations
    limit = (1 << 128) // permutations
    for token in tokens:
        value = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=16, key=key).digest(), "big")
        bucket = min(permutations - 1, value // limit)
        remainder = value % limit
        if bins[bucket] < 0 or remainder < bins[bucket]:
            bins[bucket] = remainder
    return tuple(bins)


def minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    comparable = [(a, b) for a, b in zip(left, right) if a >= 0 or b >= 0]
    if not comparable:
        return 1.0
    return sum(a == b and a >= 0 for a, b in comparable) / len(comparable)


def signature_buckets(signature: tuple[int, ...], bands: int) -> list[str]:
    rows = len(signature) // bands
    result: list[str] = []
    for band in range(bands):
        section = signature[band * rows : (band + 1) * rows]
        payload = ",".join(map(str, section)).encode("ascii")
        result.append(hashlib.sha256(payload).hexdigest()[:24])
    return result


def script_ratio(language: str, text: str) -> float:
    meaningful = 0
    matches = 0
    if language == "eng_Latn":
        for char in text:
            if char.isalpha():
                meaningful += 1
                matches += "A" <= char <= "Z" or "a" <= char <= "z"
    elif language in {"zho_Hans", "zho_Hant"}:
        for char in text:
            if char.isalpha():
                meaningful += 1
                codepoint = ord(char)
                matches += any(start <= codepoint <= end for start, end in HAN_RANGES)
    elif language == "jpn_Jpan":
        for char in text:
            if char.isalpha():
                meaningful += 1
                matches += ("\u3040" <= char <= "\u30ff") or ("\u3400" <= char <= "\u9fff")
    else:
        for char in text:
            if char.isalpha():
                meaningful += 1
                matches += "\uac00" <= char <= "\ud7af"
    return matches / meaningful if meaningful else 0.0


def chinese_variant_score(language: str, text: str) -> tuple[int, float | None]:
    """Return distinctive Han count and expected-variant share.

    Shared Han characters intentionally do not contribute.  A score is only
    actionable when several distinctive characters provide strong evidence.
    """
    if language not in {"zho_Hans", "zho_Hant"}:
        return 0, None
    simplified = sum(char in HANS_FEATURES for char in text)
    traditional = sum(char in HANT_FEATURES for char in text)
    total = simplified + traditional
    if not total:
        return 0, None
    expected = simplified if language == "zho_Hans" else traditional
    return total, expected / total


def signature_bucket_keys(signature: tuple[int, ...], bands: int) -> tuple[int, ...]:
    """Return compact keys equivalent to the configured 24-hex-digit buckets."""
    rows = len(signature) // bands
    result: list[int] = []
    for band in range(bands):
        section = signature[band * rows : (band + 1) * rows]
        payload = ",".join(map(str, section)).encode("ascii")
        result.append(int.from_bytes(hashlib.sha256(payload).digest()[:12], "big"))
    return tuple(result)


def pack_minhash(signature: tuple[int, ...]) -> bytes:
    return b"".join((value if value >= 0 else MINHASH_SENTINEL).to_bytes(16, "big") for value in signature)


def packed_minhash_similarity(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or len(left) % 16:
        raise ValueError("packed MinHash signatures have incompatible lengths")
    comparable = 0
    equal = 0
    sentinel = MINHASH_SENTINEL.to_bytes(16, "big")
    for offset in range(0, len(left), 16):
        a = left[offset : offset + 16]
        b = right[offset : offset + 16]
        if a != sentinel or b != sentinel:
            comparable += 1
            equal += a == b and a != sentinel
    return equal / comparable if comparable else 1.0


def _initialize_worker(language: str, cleaning: dict[str, Any], dedup: dict[str, Any], seed: int, script_threshold: float) -> None:
    global _WORKER_LANGUAGE, _WORKER_CLEANING, _WORKER_DEDUP, _WORKER_SEED, _WORKER_SCRIPT_THRESHOLD
    _WORKER_LANGUAGE = language
    _WORKER_CLEANING = cleaning
    _WORKER_DEDUP = dedup
    _WORKER_SEED = seed
    _WORKER_SCRIPT_THRESHOLD = script_threshold


def _fingerprint_cleaned(text: str) -> tuple[bytes, bytes, bytes, int, bool, bytes | None, tuple[int, ...]]:
    assert _WORKER_LANGUAGE is not None and _WORKER_DEDUP is not None
    encoded = text.encode("utf-8")
    cid = hashlib.sha256(encoded).digest()
    priority = hashlib.sha256(str(_WORKER_SEED).encode("ascii") + b":" + cid.hex().encode("ascii")).digest()
    signature_bytes: bytes | None = None
    bucket_keys: tuple[int, ...] = ()
    if _WORKER_LANGUAGE in _WORKER_DEDUP["approximate_languages"]:
        parameters = _WORKER_DEDUP["minhash"]
        signature = one_permutation_minhash(text, parameters)
        signature_bytes = pack_minhash(signature)
        bucket_keys = signature_bucket_keys(signature, int(parameters["bands"]))
    passes_script_check = script_ratio(_WORKER_LANGUAGE, text) >= _WORKER_SCRIPT_THRESHOLD
    return priority, cid, encoded, len(text), passes_script_check, signature_bytes, bucket_keys


def _process_document_batch(
    batch: list[tuple[Any, str, str]],
) -> list[tuple[str, str, int, int, int, dict[str, int], list[Any], list[tuple[str, bytes, str]]]]:
    """Clean and fingerprint a batch; document boundaries remain explicit."""
    assert _WORKER_CLEANING is not None
    results = []
    for raw_text, source_url, split in batch:
        input_units = 0
        cleaned_units = 0
        cleaned_characters = 0
        filters: Counter[str] = Counter()
        fingerprints = []
        rejected_examples: list[tuple[str, bytes, str]] = []
        for cleaned, reason, excerpt in _clean_units(raw_text, _WORKER_CLEANING):
            input_units += 1
            if reason:
                filters[reason] += 1
                encoded_excerpt = excerpt.encode("utf-8", errors="replace")
                rejected_examples.append((reason, hashlib.sha256(encoded_excerpt).digest(), excerpt))
                continue
            assert cleaned is not None
            ratio = script_ratio(_WORKER_LANGUAGE or "", cleaned)
            if ratio < _WORKER_SCRIPT_THRESHOLD:
                filters["script_mismatch"] += 1
                encoded_excerpt = cleaned.encode("utf-8")
                rejected_examples.append(("script_mismatch", hashlib.sha256(encoded_excerpt).digest(), cleaned))
                continue
            distinctive, variant_share = chinese_variant_score(_WORKER_LANGUAGE or "", cleaned)
            if distinctive >= 4 and variant_share is not None and variant_share < 0.20:
                filters["chinese_variant_mismatch"] += 1
                encoded_excerpt = cleaned.encode("utf-8")
                rejected_examples.append(("chinese_variant_mismatch", hashlib.sha256(encoded_excerpt).digest(), cleaned))
                continue
            cleaned_units += 1
            cleaned_characters += len(cleaned)
            fingerprints.append(_fingerprint_cleaned(cleaned))
        results.append(
            (source_url, split, input_units, cleaned_units, cleaned_characters, dict(filters), fingerprints, rejected_examples)
        )
    return results


def _iter_locked_documents(source: dict[str, Any], cached: dict[str, Path]) -> Iterator[dict[str, Any]]:
    for shard in sorted(source["shards"], key=lambda item: item["logical_order"]):
        path = cached[f"{source['source_id']}:{shard['logical_order']}"]
        yield from iter_jsonl_zst_prefix(path)


def _document_split(raw_text: Any, source_url: str, holdout: dict[str, Any]) -> str:
    if source_url:
        key = source_url
    elif isinstance(raw_text, str):
        key = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    else:
        key = "missing-text"
    payload = f"{holdout['seed']}\0{key}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    return "holdout" if value < float(holdout["fraction"]) else "train"


def _iter_document_batches(
    documents: Iterator[dict[str, Any]],
    target_characters: int,
    holdout: dict[str, Any],
) -> Iterator[list[tuple[Any, str, str]]]:
    batch: list[tuple[Any, str, str]] = []
    characters = 0
    for document in documents:
        raw_text = document.get("text")
        source_url = str(document.get("u") or "")
        batch.append((raw_text, source_url, _document_split(raw_text, source_url, holdout)))
        characters += len(raw_text) if isinstance(raw_text, str) else 1
        if characters >= target_characters or len(batch) >= 512:
            yield batch
            batch = []
            characters = 0
    if batch:
        yield batch


def _ordered_document_results(
    documents: Iterator[dict[str, Any]],
    language: str,
    cleaning: dict[str, Any],
    dedup: dict[str, Any],
    seed: int,
    script_threshold: float,
    workers: int,
    batch_characters: int,
    holdout: dict[str, Any],
) -> Iterator[tuple[str, str, int, int, int, dict[str, int], list[Any], list[tuple[str, bytes, str]]]]:
    batches = _iter_document_batches(documents, batch_characters, holdout)
    if workers == 1:
        _initialize_worker(language, cleaning, dedup, seed, script_threshold)
        for batch in batches:
            yield from _process_document_batch(batch)
        return

    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(language, cleaning, dedup, seed, script_threshold),
    )
    pending: deque[Future[Any]] = deque()
    exhausted = False

    def submit_one() -> None:
        nonlocal exhausted
        if exhausted:
            return
        try:
            batch = next(batches)
        except StopIteration:
            exhausted = True
            return
        pending.append(executor.submit(_process_document_batch, batch))

    try:
        for _ in range(workers):
            submit_one()
        while pending:
            result = pending.popleft().result()
            submit_one()
            yield from result
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memory_snapshot() -> tuple[int, int, int]:
    """Return total physical, available physical, and current-process RSS."""
    total = available = rss = 0
    if os.name == "nt":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        process = get_current_process()
        get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
        if get_process_memory(process, ctypes.byref(counters), counters.cb):
            rss = int(counters.WorkingSetSize)
    else:
        with contextlib.suppress(OSError, ValueError):
            pages = os.sysconf("SC_PHYS_PAGES")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total = int(pages * page_size)
            available = int(available_pages * page_size)
        with contextlib.suppress(OSError, ValueError):
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            rss = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    return total, available, rss


def _memory_limits(max_memory_gib: float | None, min_available_memory_gib: float | None) -> tuple[int, int]:
    total, _, _ = memory_snapshot()
    if total <= 0:
        total = 8 * GIB
    if max_memory_gib is not None and max_memory_gib <= 0:
        raise PipelineError("max_memory_gib must be positive")
    if min_available_memory_gib is not None and min_available_memory_gib < 0:
        raise PipelineError("min_available_memory_gib cannot be negative")
    max_rss = int(max_memory_gib * GIB) if max_memory_gib is not None else min(48 * GIB, max(2 * GIB, int(total * 0.40)))
    min_available = (
        int(min_available_memory_gib * GIB)
        if min_available_memory_gib is not None
        else min(32 * GIB, max(512 * MIB, int(total * 0.25)))
    )
    return max_rss, min_available


def _sample_resources(tracker: dict[str, int], language: str, max_rss: int, min_available: int) -> None:
    _, available, rss = memory_snapshot()
    tracker["peak_main_rss_bytes"] = max(tracker.get("peak_main_rss_bytes", 0), rss)
    if available:
        previous = tracker.get("minimum_available_memory_bytes")
        tracker["minimum_available_memory_bytes"] = available if previous is None else min(previous, available)
    if rss > max_rss:
        raise PipelineError(
            f"RAM-first safety stop for {language}: main RSS {rss / GIB:.2f} GiB exceeds {max_rss / GIB:.2f} GiB"
        )
    if available and available < min_available:
        raise PipelineError(
            f"RAM-first safety stop for {language}: system available memory {available / GIB:.2f} GiB is below {min_available / GIB:.2f} GiB"
        )


def percentile_from_counts(counts: dict[int, int] | Counter[int], fraction: float) -> int:
    total = sum(counts.values())
    if total <= 0:
        return 0
    wanted = min(total - 1, int((total - 1) * fraction))
    seen = 0
    for value, count in sorted(counts.items()):
        seen += count
        if seen > wanted:
            return int(value)
    raise AssertionError("unreachable percentile state")


def git_identity(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, text=True, capture_output=True, check=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True


def relevant_hashes(repo_root: Path) -> dict[str, str]:
    paths = [
        repo_root / "scripts" / "fetch_tokenizer_datasets.py",
        repo_root / "scripts" / "tokenizer_dataset_pipeline.py",
        repo_root / "requirements.lock",
    ]
    return {path.relative_to(repo_root).as_posix(): sha256_file(path) for path in paths if path.exists()}


def _build_provenance(
    config: dict[str, Any], config_path: Path, lock_path: Path, profile_name: str, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    commit, dirty = git_identity(repo_root)
    dependency_hash = sha256_file(repo_root / "requirements.lock")
    code_hashes = relevant_hashes(repo_root)
    provenance = {
        "algorithm_version": {
            "cleaning": config["cleaning"]["algorithm_version"],
            "deduplication": config["deduplication"]["algorithm_version"],
            "pipeline": "ordered-ram-first-five-language-holdout-v2",
        },
        "config_sha256": sha256_file(config_path),
        "source_lock_sha256": sha256_file(lock_path),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "dependency_lock_sha256": dependency_hash,
        "source_code_sha256": code_hashes,
        "profile": profile_name,
        "seed": seed,
    }
    checkpoint_identity = {
        "algorithm_version": provenance["algorithm_version"],
        "config_sha256": provenance["config_sha256"],
        "source_lock_sha256": provenance["source_lock_sha256"],
        "dependency_lock_sha256": dependency_hash,
        "source_code_sha256": code_hashes,
        "profile": profile_name,
        "seed": seed,
    }
    return provenance, checkpoint_identity


def _checkpoint_path(checkpoint_dir: Path, language: str) -> Path:
    return checkpoint_dir / f"{language}.json"


def _load_language_checkpoint(
    checkpoint_dir: Path,
    staging_dir: Path,
    corpus_dir: Path,
    holdout_dir: Path,
    language: str,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    path = _checkpoint_path(checkpoint_dir, language)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        state.get("schema_version") != RAM_CHECKPOINT_SCHEMA_VERSION
        or state.get("identity") != identity
        or state.get("language") != language
    ):
        return None
    output = state.get("output")
    holdout_output = state.get("holdout_output")
    if (
        not isinstance(output, dict)
        or not isinstance(holdout_output, dict)
        or not re.fullmatch(r"[0-9a-f]{64}", str(output.get("sha256", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(holdout_output.get("sha256", "")))
    ):
        return None
    train_candidates = (corpus_dir / f"{language}.txt", staging_dir / "train" / f"{language}.txt")
    holdout_candidates = (holdout_dir / f"{language}.txt", staging_dir / "holdout" / f"{language}.txt")
    train_artifact = next(
        (
            artifact for artifact in train_candidates
            if artifact.is_file()
            and artifact.stat().st_size == int(output.get("bytes", -1))
            and sha256_file(artifact) == output["sha256"]
        ),
        None,
    )
    holdout_artifact = next(
        (
            artifact for artifact in holdout_candidates
            if artifact.is_file()
            and artifact.stat().st_size == int(holdout_output.get("bytes", -1))
            and sha256_file(artifact) == holdout_output["sha256"]
        ),
        None,
    )
    if train_artifact is None or holdout_artifact is None:
        return None
    state["_artifact_path"] = train_artifact
    state["_holdout_artifact_path"] = holdout_artifact
    return state


def _write_language_checkpoint(checkpoint_dir: Path, state: dict[str, Any]) -> None:
    serializable = {key: value for key, value in state.items() if not key.startswith("_")}
    atomic_write_bytes(_checkpoint_path(checkpoint_dir, state["language"]), canonical_json_bytes(serializable))


def _transfer_staged_file(staged_path: Path, final_path: Path, expected_sha256: str, expected_bytes: int) -> str:
    """Copy one staged corpus with one sequential stream, then publish atomically."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if staged_path.resolve() == final_path.resolve():
        return str(final_path)
    if staged_path.drive.lower() == final_path.drive.lower():
        os.replace(staged_path, final_path)
    else:
        fd, temporary = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".transfer.tmp", dir=final_path.parent)
        digest = hashlib.sha256()
        copied = 0
        try:
            with staged_path.open("rb") as source, os.fdopen(fd, "wb") as destination:
                while chunk := source.read(8 * MIB):
                    destination.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if copied != expected_bytes or digest.hexdigest() != expected_sha256:
                raise PipelineError(f"staged transfer verification failed for {final_path.name}")
            os.replace(temporary, final_path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
        staged_path.unlink()
    if final_path.stat().st_size != expected_bytes:
        raise PipelineError(f"published corpus verification failed for {final_path.name}")
    return str(final_path)


def _build_language_ram_first(
    source: dict[str, Any],
    cached: dict[str, Path],
    config: dict[str, Any],
    budget: int,
    seed: int,
    workers: int,
    staging_dir: Path,
    checkpoint_dir: Path,
    checkpoint_identity: dict[str, Any],
    resource_tracker: dict[str, int],
    max_rss: int,
    min_available: int,
) -> dict[str, Any]:
    language = source["output_language"]
    source_id = source["source_id"]
    parameters = config["deduplication"]["minhash"]
    band_count = int(parameters["bands"])
    threshold = float(parameters["threshold"])
    exact_seen: dict[bytes, str] = {}
    lsh_buckets: list[dict[int, tuple[bytes, str]]] = [dict() for _ in range(band_count)]
    candidates: dict[str, list[tuple[bytes, bytes, bytes, int, str]]] = {"train": [], "holdout": []}
    candidate_characters = {"train": 0, "holdout": 0}
    holdout_budget = max(1, round(budget * float(config["holdout"]["fraction"])))
    targets = {"train": budget, "holdout": holdout_budget}
    oversampling = float(config["quality"]["candidate_oversampling_factor"])
    candidate_targets = {split: max(target + 1, int(target * oversampling)) for split, target in targets.items()}
    stats: dict[str, Any] = {
        "documents": 0,
        "documents_with_source_url": 0,
        "documents_by_split": Counter(),
        "input_units": 0,
        "cleaned_units": 0,
        "cleaned_characters": 0,
        "filters": Counter(),
        "filters_by_split": {"train": Counter(), "holdout": Counter()},
        "exact_duplicates": 0,
        "approximate_duplicates": 0,
        "cross_split_duplicates": 0,
    }
    batch_characters = min(DEFAULT_BATCH_CHARACTERS, max(1_000, budget // 4))
    next_progress = 100_000
    rejection_heap: list[tuple[int, bytes, str, str, str]] = []
    rejection_count = int(config["quality"]["rejected_review_sample_count"])
    excerpt_characters = int(config["quality"]["excerpt_characters"])
    review_seed = int(config["quality"]["review_seed"])
    documents = _iter_locked_documents(source, cached)
    results = _ordered_document_results(
        documents,
        language,
        config["cleaning"],
        config["deduplication"],
        seed,
        float(config["quality"]["language_min_script_ratio"][language]),
        workers,
        batch_characters,
        config["holdout"],
    )
    try:
        for source_url, split, input_units, cleaned_units, cleaned_characters, filters, fingerprints, rejected in results:
            stats["documents"] += 1
            stats["documents_with_source_url"] += bool(source_url)
            stats["documents_by_split"][split] += 1
            stats["input_units"] += input_units
            stats["cleaned_units"] += cleaned_units
            stats["cleaned_characters"] += cleaned_characters
            stats["filters"].update(filters)
            stats["filters_by_split"][split].update(filters)
            for reason, rejected_id, excerpt in rejected:
                rank = int.from_bytes(
                    hashlib.sha256(str(review_seed).encode("ascii") + b":reject:" + rejected_id).digest(), "big"
                )
                item = (-rank, rejected_id, reason, split, excerpt[:excerpt_characters].replace("\n", " "))
                if len(rejection_heap) < rejection_count:
                    heapq.heappush(rejection_heap, item)
                elif rank < -rejection_heap[0][0]:
                    heapq.heapreplace(rejection_heap, item)
            for priority, cid, text_bytes, characters, script_pass, signature, bucket_keys in fingerprints:
                previous_split = exact_seen.get(cid)
                if previous_split is not None:
                    stats["exact_duplicates"] += 1
                    stats["cross_split_duplicates"] += previous_split != split
                    continue
                assert signature is not None
                possible: list[tuple[bytes, str]] = []
                for band, bucket in enumerate(bucket_keys):
                    other = lsh_buckets[band].get(bucket)
                    if other is not None and other not in possible:
                        possible.append(other)
                duplicate = next(
                    ((other_signature, other_split) for other_signature, other_split in possible
                     if packed_minhash_similarity(signature, other_signature) >= threshold),
                    None,
                )
                if duplicate is not None:
                    stats["approximate_duplicates"] += 1
                    stats["cross_split_duplicates"] += duplicate[1] != split
                    continue
                exact_seen[cid] = split
                for band, bucket in enumerate(bucket_keys):
                    lsh_buckets[band].setdefault(bucket, (signature, split))
                candidates[split].append((priority, cid, text_bytes, characters, source_url))
                candidate_characters[split] += characters
                total_candidates = len(candidates["train"]) + len(candidates["holdout"])
                if total_candidates >= next_progress:
                    _sample_resources(resource_tracker, language, max_rss, min_available)
                    print(
                        f"[{language}] train={candidate_characters['train']:,} holdout={candidate_characters['holdout']:,} "
                        f"main_rss={resource_tracker['peak_main_rss_bytes'] / GIB:.2f}GiB",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_progress += 100_000
            if all(candidate_characters[split_name] >= target for split_name, target in candidate_targets.items()):
                break
    finally:
        results.close()
        documents.close()

    stats["candidate_samples_by_split"] = {split: len(items) for split, items in candidates.items()}
    stats["candidate_characters_by_split"] = dict(candidate_characters)
    stats["candidate_samples"] = sum(stats["candidate_samples_by_split"].values())
    stats["candidate_characters"] = sum(candidate_characters.values())
    for split, target in targets.items():
        if target >= 1_000_000 and candidate_characters[split] < target:
            raise PipelineError(
                f"locked input exhausted before {language} {split} reached its {target:,}-character budget "
                f"({candidate_characters[split]:,} candidates)"
            )
    _sample_resources(resource_tracker, language, max_rss, min_available)
    for items in candidates.values():
        items.sort()
    staging_dir.mkdir(parents=True, exist_ok=True)
    review_count = int(config["quality"]["review_sample_count"])
    domain_fraction = float(
        config["quality"].get("max_domain_fraction_by_language", {}).get(
            language,
            config["quality"]["max_domain_fraction"],
        )
    )
    maximum_domain_characters = {
        split: max(int(config["cleaning"]["max_characters"]), int(target * domain_fraction))
        for split, target in targets.items()
    }

    def write_split(split: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]], Counter[int], int]:
        split_dir = staging_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        staged_path = split_dir / f"{language}.txt"
        for stale in split_dir.glob(f".{language}.*.tmp"):
            stale.unlink(missing_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{language}.", suffix=".tmp", dir=split_dir)
        selected_characters = 0
        selected_lines = 0
        domain_capped = 0
        domain_characters: Counter[str] = Counter()
        length_counts: Counter[int] = Counter()
        review_heap: list[tuple[int, bytes, bytes, bytes]] = []
        output_digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as handle:
                for _priority, cid, text_bytes, characters, source_url in candidates[split]:
                    if selected_characters + characters > targets[split] and not (
                        targets[split] < 1_000_000 and selected_lines == 0
                    ):
                        continue
                    hostname = (urllib.parse.urlsplit(source_url).hostname or "").lower()
                    if hostname and domain_characters[hostname] + characters > maximum_domain_characters[split]:
                        domain_capped += 1
                        continue
                    line = text_bytes + b"\n"
                    handle.write(line)
                    output_digest.update(line)
                    selected_characters += characters
                    selected_lines += 1
                    if hostname:
                        domain_characters[hostname] += characters
                    length_counts[characters] += 1
                    rank = int.from_bytes(
                        hashlib.sha256(str(review_seed).encode("ascii") + b":" + cid.hex().encode("ascii")).digest(), "big"
                    )
                    source_url_digest = hashlib.sha256(source_url.encode("utf-8")).digest()
                    item = (-rank, cid, source_url_digest, text_bytes[: excerpt_characters * 4])
                    if len(review_heap) < review_count:
                        heapq.heappush(review_heap, item)
                    elif rank < -review_heap[0][0]:
                        heapq.heapreplace(review_heap, item)
                    if selected_characters >= targets[split]:
                        break
                handle.flush()
                os.fsync(handle.fileno())
            minimum_characters = 1 if targets[split] < 1_000_000 else targets[split] - int(config["cleaning"]["max_characters"])
            if selected_lines == 0 or selected_characters < minimum_characters:
                raise PipelineError(
                    f"deterministic selection for {language} {split} retained only "
                    f"{selected_characters:,}/{targets[split]:,} characters; domain cap may be too strict"
                )
            os.replace(temporary, staged_path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
        review = []
        for _negative_rank, cid, source_url_digest, text_bytes in sorted(review_heap, key=lambda item: (-item[0], item[1])):
            text = text_bytes.decode("utf-8", errors="ignore")[:excerpt_characters].replace("\n", " ")
            distinctive, variant_share = chinese_variant_score(language, text)
            review.append(
                {
                    "content_id": cid.hex(),
                    "source_url_sha256": source_url_digest.hex(),
                    "status": "pass",
                    "script_ratio": round(script_ratio(language, text), 6),
                    "distinctive_chinese_characters": distinctive,
                    "expected_chinese_variant_share": round(variant_share, 6) if variant_share is not None else None,
                    "text_excerpt": text,
                }
            )
        output = {
            "sha256": output_digest.hexdigest(),
            "bytes": staged_path.stat().st_size,
            "lines": selected_lines,
            "characters": selected_characters,
        }
        return staged_path, output, review, length_counts, domain_capped

    train_path, output, review, train_lengths, train_domain_capped = write_split("train")
    holdout_path, holdout_output, holdout_review, holdout_lengths, holdout_domain_capped = write_split("holdout")
    rejected_review = [
        {
            "content_id": cid.hex(),
            "reason": reason,
            "split": split,
            "text_excerpt": excerpt,
        }
        for _negative_rank, cid, reason, split, excerpt in sorted(rejection_heap, key=lambda item: (-item[0], item[1]))
    ]
    stats.update(
        {
            "sampled_out": len(candidates["train"]) - output["lines"],
            "final_lines": output["lines"],
            "final_characters": output["characters"],
            "length_counts": {str(length): count for length, count in sorted(train_lengths.items())},
            "holdout_sampled_out": len(candidates["holdout"]) - holdout_output["lines"],
            "holdout_final_lines": holdout_output["lines"],
            "holdout_final_characters": holdout_output["characters"],
            "holdout_length_counts": {str(length): count for length, count in sorted(holdout_lengths.items())},
            "domain_capped": train_domain_capped,
            "holdout_domain_capped": holdout_domain_capped,
            "max_domain_fraction": domain_fraction,
            "documents_by_split": dict(sorted(stats["documents_by_split"].items())),
            "filters_by_split": {
                split: dict(sorted(values.items())) for split, values in stats["filters_by_split"].items()
            },
            "filters": dict(sorted(stats["filters"].items())),
        }
    )
    state = {
        "schema_version": RAM_CHECKPOINT_SCHEMA_VERSION,
        "identity": checkpoint_identity,
        "language": language,
        "output": output,
        "holdout_output": holdout_output,
        "stats": stats,
        "source_sample_counts": {source_id: output["lines"]},
        "holdout_source_sample_counts": {source_id: holdout_output["lines"]},
        "review": review,
        "holdout_review": holdout_review,
        "rejected_review": rejected_review,
        "_artifact_path": train_path,
        "_holdout_artifact_path": holdout_path,
    }
    _write_language_checkpoint(checkpoint_dir, state)
    _sample_resources(resource_tracker, language, max_rss, min_available)
    print(
        f"[{language}] staged train={output['lines']:,}/{output['characters']:,} chars, "
        f"holdout={holdout_output['lines']:,}/{holdout_output['characters']:,} chars",
        file=sys.stderr,
        flush=True,
    )
    return state


def build_corpus(
    config: dict[str, Any],
    config_path: Path,
    lock: dict[str, Any],
    lock_path: Path,
    out_root: Path,
    cache_root: Path,
    profile_name: str,
    seed: int,
    *,
    offline: bool,
    use_cache: bool,
    resume: bool = False,
    staging_root: Path | None = None,
    max_memory_gib: float | None = None,
    min_available_memory_gib: float | None = None,
) -> dict[str, Any]:
    started = time.time()
    profile = config["profiles"][profile_name]
    budget = int(profile["character_budget_per_language"])
    workers = int(profile["concurrency"])
    if workers <= 0:
        raise PipelineError("profile concurrency must be positive")
    max_rss, min_available = _memory_limits(max_memory_gib, min_available_memory_gib)
    resource_tracker: dict[str, int] = {"peak_main_rss_bytes": 0}
    _sample_resources(resource_tracker, "startup", max_rss, min_available)
    provenance, checkpoint_identity = _build_provenance(config, config_path, lock_path, profile_name, seed)

    # Validation is a sequential read of each locked prefix. No candidate or
    # index data is written during this phase.
    cached = ensure_cache(lock, cache_root, offline=offline, use_cache=use_cache)
    interim = out_root / "interim" / profile_name / "ram-first"
    checkpoint_dir = interim / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = out_root / profile["corpus_subdir"]
    holdout_dir = out_root / profile["holdout_subdir"]
    corpus_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
        for directory in (corpus_dir, holdout_dir):
            for stale in directory.glob(f".{language}.*.transfer.tmp"):
                stale.unlink(missing_ok=True)
    namespace = hashlib.sha256(str(out_root.resolve()).encode("utf-8")).hexdigest()[:16]
    staging_dir = (
        staging_root / namespace / profile_name
        if staging_root is not None
        else interim / "staging"
    )
    manifest_path = corpus_dir / "manifest.jsonl"
    holdout_manifest_path = holdout_dir / "manifest.jsonl"
    run_path = corpus_dir / "run.json"
    report_path = out_root / "reports" / f"tokenizer_corpus_{profile_name}.md"
    # A stale manifest must never make an interrupted rebuild look complete.
    manifest_path.unlink(missing_ok=True)
    holdout_manifest_path.unlink(missing_ok=True)
    run_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)

    states: dict[str, dict[str, Any]] = {}
    transfers: dict[tuple[str, str], Future[str]] = {}
    sources = sorted(lock["sources"], key=lambda item: item["output_language"])
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tokenizer-corpus-transfer") as transfer_pool:
        for source in sources:
            language = source["output_language"]
            state = (
                _load_language_checkpoint(checkpoint_dir, staging_dir, corpus_dir, holdout_dir, language, checkpoint_identity)
                if resume
                else None
            )
            if state is None:
                state = _build_language_ram_first(
                    source,
                    cached,
                    config,
                    budget,
                    seed,
                    workers,
                    staging_dir,
                    checkpoint_dir,
                    checkpoint_identity,
                    resource_tracker,
                    max_rss,
                    min_available,
                )
            else:
                print(f"[{language}] resume checkpoint verified", file=sys.stderr, flush=True)
            states[language] = state
            artifact = Path(state["_artifact_path"])
            final_path = corpus_dir / f"{language}.txt"
            if artifact.resolve() != final_path.resolve():
                output = state["output"]
                transfers[(language, "train")] = transfer_pool.submit(
                    _transfer_staged_file,
                    artifact,
                    final_path,
                    output["sha256"],
                    int(output["bytes"]),
                )
            else:
                state["_artifact_path"] = final_path
            holdout_artifact = Path(state["_holdout_artifact_path"])
            holdout_final_path = holdout_dir / f"{language}.txt"
            if holdout_artifact.resolve() != holdout_final_path.resolve():
                holdout_output = state["holdout_output"]
                transfers[(language, "holdout")] = transfer_pool.submit(
                    _transfer_staged_file,
                    holdout_artifact,
                    holdout_final_path,
                    holdout_output["sha256"],
                    int(holdout_output["bytes"]),
                )
            else:
                state["_holdout_artifact_path"] = holdout_final_path

        # The deterministic manifest is the completion marker and is not
        # published until every background transfer has verified and renamed.
        for language in LANGUAGES:
            if (language, "train") in transfers:
                final_path = Path(transfers[(language, "train")].result())
                states[language]["_artifact_path"] = final_path
                print(f"[{language}] background transfer verified at {final_path}", file=sys.stderr, flush=True)
            if (language, "holdout") in transfers:
                final_path = Path(transfers[(language, "holdout")].result())
                states[language]["_holdout_artifact_path"] = final_path
                print(f"[{language}] holdout transfer verified at {final_path}", file=sys.stderr, flush=True)

    output_metadata = {language: states[language]["output"] for language in LANGUAGES}
    holdout_output_metadata = {language: states[language]["holdout_output"] for language in LANGUAGES}
    stats = {language: states[language]["stats"] for language in LANGUAGES}
    source_proportions = {language: states[language]["source_sample_counts"] for language in LANGUAGES}
    holdout_source_proportions = {language: states[language]["holdout_source_sample_counts"] for language in LANGUAGES}
    review = {language: states[language]["review"] for language in LANGUAGES}
    holdout_review = {language: states[language]["holdout_review"] for language in LANGUAGES}
    rejected_review = {language: states[language]["rejected_review"] for language in LANGUAGES}

    source_config = {source["source_id"]: source for source in enabled_sources(config)}
    manifest_records: list[dict[str, Any]] = []
    holdout_manifest_records: list[dict[str, Any]] = []
    for language in LANGUAGES:
        language_sources = []
        for source in lock["sources"]:
            if source["output_language"] == language:
                registry = source_config[source["source_id"]]
                language_sources.append(
                    {
                        "source_id": source["source_id"],
                        "source_language": source["source_language"],
                        "output_language": language,
                        "version_or_snapshot": registry["version_or_snapshot"],
                        "license": registry["license"],
                        "homepage": registry["homepage"],
                        "map_sha256": source["map_sha256"],
                        "shard_prefix_sha256": [item["sha256"] for item in source["shards"]],
                    }
                )
        item = output_metadata[language]
        language_stats = stats[language]
        cleaned_units = max(1, int(language_stats["cleaned_units"]))
        manifest_records.append(
            {
                "language": language,
                "file": f"{language}.txt",
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "samples": item["lines"],
                "characters": item["characters"],
                "documents": language_stats["documents"],
                "documents_with_source_url": language_stats["documents_with_source_url"],
                "input_units": language_stats["input_units"],
                "cleaned_units": language_stats["cleaned_units"],
                "cleaned_characters": language_stats["cleaned_characters"],
                "candidate_samples": language_stats["candidate_samples"],
                "candidate_characters": language_stats["candidate_characters"],
                "filters": dict(sorted(language_stats["filters"].items())),
                "exact_duplicates": language_stats["exact_duplicates"],
                "exact_duplicate_rate": round(language_stats["exact_duplicates"] / cleaned_units, 9),
                "approximate_duplicates": language_stats["approximate_duplicates"],
                "approximate_duplicate_rate": round(language_stats["approximate_duplicates"] / cleaned_units, 9),
                "sampled_out": language_stats["sampled_out"],
                "split": "train",
                "holdout_policy": config["holdout"],
                "cross_split_duplicates_removed": language_stats["cross_split_duplicates"],
                "domain_capped": language_stats["domain_capped"],
                "source_sample_counts": source_proportions[language],
                "sources": language_sources,
                "provenance": provenance,
            }
        )
        holdout_item = holdout_output_metadata[language]
        holdout_manifest_records.append(
            {
                "language": language,
                "file": f"{language}.txt",
                "sha256": holdout_item["sha256"],
                "bytes": holdout_item["bytes"],
                "samples": holdout_item["lines"],
                "characters": holdout_item["characters"],
                "documents": language_stats["documents_by_split"].get("holdout", 0),
                "input_units": language_stats["input_units"],
                "cleaned_units": language_stats["cleaned_units"],
                "filters": language_stats["filters_by_split"].get("holdout", {}),
                "sampled_out": language_stats["holdout_sampled_out"],
                "split": "holdout",
                "holdout_policy": config["holdout"],
                "cross_split_duplicates_removed": language_stats["cross_split_duplicates"],
                "domain_capped": language_stats["holdout_domain_capped"],
                "source_sample_counts": holdout_source_proportions[language],
                "sources": language_sources,
                "provenance": provenance,
            }
        )
    atomic_write_bytes(manifest_path, b"".join(canonical_json_bytes(record) for record in manifest_records))
    atomic_write_bytes(
        holdout_manifest_path,
        b"".join(canonical_json_bytes(record) for record in holdout_manifest_records),
    )

    lines = [
        f"# Tokenizer corpus quality report: {profile_name}",
        "",
        "Deterministic rebuild command:",
        "",
        f"`python scripts/fetch_tokenizer_datasets.py --profile {profile_name} --use-cache --offline`",
        "",
        "## Provenance",
        "",
        f"- Config SHA-256: `{provenance['config_sha256']}`",
        f"- Source lock SHA-256: `{provenance['source_lock_sha256']}`",
        f"- Dependency lock SHA-256: `{provenance['dependency_lock_sha256']}`",
        f"- Pipeline: `{provenance['algorithm_version']['pipeline']}`; seed: `{seed}`",
        "",
        "## Summary",
        "",
        "| Language | Train chars | Holdout chars | p50 | p95 | Exact dedup | Approx dedup | Cross-split dedup | SHA-256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for language in LANGUAGES:
        language_stats = stats[language]
        length_counts = {int(key): int(value) for key, value in language_stats["length_counts"].items()}
        lines.append(
            f"| {language} | {language_stats['final_characters']} | {language_stats['holdout_final_characters']} | "
            f"{percentile_from_counts(length_counts, .5)} | {percentile_from_counts(length_counts, .95)} | "
            f"{language_stats['exact_duplicates']} | {language_stats['approximate_duplicates']} | "
            f"{language_stats['cross_split_duplicates']} | `{output_metadata[language]['sha256']}` |"
        )
    lines.extend(["", "## Filtering and sources", ""])
    for language in LANGUAGES:
        language_stats = stats[language]
        input_units = max(1, int(language_stats["input_units"]))
        cleaned_units = max(1, int(language_stats["cleaned_units"]))
        filter_text = ", ".join(
            f"`{key}`={value} ({value / input_units:.4%})" for key, value in sorted(language_stats["filters"].items())
        ) or "none"
        source_total = max(1, sum(source_proportions[language].values()))
        source_text = ", ".join(
            f"`{key}`={value} ({value / source_total:.4%})" for key, value in source_proportions[language].items()
        )
        lines.extend(
            [
                f"### {language}",
                "",
                f"Documents: {language_stats['documents']}; input units: {language_stats['input_units']}; "
                f"documents with source URL: {language_stats['documents_with_source_url']}; "
                f"cleaned units: {language_stats['cleaned_units']}; cleaned characters: {language_stats['cleaned_characters']}; "
                f"candidate units: {language_stats['candidate_samples']}; candidate characters: {language_stats['candidate_characters']}.",
                "",
                f"Exact dedup rate: {language_stats['exact_duplicates'] / cleaned_units:.6%}; "
                f"approximate dedup rate: {language_stats['approximate_duplicates'] / cleaned_units:.6%}; "
                f"cross-split duplicates removed: {language_stats['cross_split_duplicates']}.",
                "",
                f"Filter reason counts and input rates: {filter_text}.",
                "",
                f"Final source proportions: {source_text}.",
                f"Domain-cap removals: train={language_stats['domain_capped']}, holdout={language_stats['holdout_domain_capped']}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Fixed manual-review sample",
            "",
            "Stable, safely truncated excerpts are included for accepted and rejected samples so filtering can be audited.",
            "",
        ]
    )
    for language in LANGUAGES:
        review_items = review[language]
        lines.extend(
            [
                f"### {language}",
                "",
                f"Accepted train samples: {len(review_items)}; accepted holdout samples: {len(holdout_review[language])}; "
                f"rejected samples: {len(rejected_review[language])}.",
                "",
            ]
        )
        for item in review_items:
            excerpt = item["text_excerpt"].replace("`", "'")
            lines.append(
                f"- accepted `{item['content_id'][:16]}` script={item['script_ratio']:.3f}: `{excerpt}`"
            )
        for item in rejected_review[language]:
            excerpt = item["text_excerpt"].replace("`", "'")
            lines.append(
                f"- rejected `{item['content_id'][:16]}` ({item['reason']}, {item['split']}): `{excerpt}`"
            )
        lines.append("")
    atomic_write_bytes(report_path, ("\n".join(lines).rstrip() + "\n").encode("utf-8"))

    _sample_resources(resource_tracker, "finalize", max_rss, min_available)
    run_record = {
        "started_at_utc": dt.datetime.fromtimestamp(started, dt.timezone.utc).isoformat(),
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "out_root": str(out_root.resolve()),
        "cache_root": str(cache_root.resolve()),
        "staging_root": str(staging_root.resolve()) if staging_root is not None else str(staging_dir.resolve()),
        "profile": profile_name,
        "concurrency": workers,
        "offline": offline,
        "use_cache": use_cache,
        "resume": resume,
        "max_main_rss_bytes": max_rss,
        "minimum_available_memory_limit_bytes": min_available,
        "peak_main_rss_bytes": resource_tracker["peak_main_rss_bytes"],
        "minimum_available_memory_bytes": resource_tracker.get("minimum_available_memory_bytes"),
        "manifest_sha256": sha256_file(manifest_path),
        "holdout_manifest_sha256": sha256_file(holdout_manifest_path),
        "report_sha256": sha256_file(report_path),
    }
    atomic_write_bytes(run_path, json.dumps(run_record, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return {
        "corpus_dir": str(corpus_dir),
        "holdout_dir": str(holdout_dir),
        "report": str(report_path),
        "manifest_sha256": run_record["manifest_sha256"],
        "holdout_manifest_sha256": run_record["holdout_manifest_sha256"],
        "outputs": output_metadata,
        "holdout_outputs": holdout_output_metadata,
        "duration_seconds": run_record["duration_seconds"],
        "peak_main_rss_bytes": run_record["peak_main_rss_bytes"],
    }


def dry_run_plan(
    config: dict[str, Any],
    lock: dict[str, Any],
    out_root: Path,
    cache_root: Path,
    profile_name: str,
    seed: int,
    offline: bool,
    use_cache: bool,
    *,
    staging_root: Path | None = None,
    max_memory_gib: float | None = None,
    min_available_memory_gib: float | None = None,
) -> dict[str, Any]:
    profile = config["profiles"][profile_name]
    return {
        "action": "build",
        "profile": profile_name,
        "sources": [
            {
                "source_id": source["source_id"],
                "mapping": f"{source['source_language']} -> {source['output_language']}",
                "shards": [
                    {
                        "wds": shard["wds"],
                        "url": shard["url"],
                        "locked_bytes": shard["locked_bytes"],
                    }
                    for shard in source["shards"]
                ],
            }
            for source in lock["sources"]
        ],
        "quality_wds": config["dataset"]["quality_wds"],
        "character_budget_per_language": profile["character_budget_per_language"],
        "seed": seed,
        "concurrency": profile["concurrency"],
        "cache": str(cache_root),
        "staging": str(staging_root) if staging_root is not None else str(out_root / "interim" / profile_name / "ram-first" / "staging"),
        "output": str(out_root / profile["corpus_subdir"]),
        "ram_first": {
            "single_language_resident": True,
            "candidate_database": False,
            "max_memory_gib": max_memory_gib,
            "min_available_memory_gib": min_available_memory_gib,
            "background_transfers": 1,
        },
        "network_allowed": not (offline or use_cache),
        "operations": ["validate locked cache or download locked prefixes", "single-stream JSONL read", "ordered parallel cleaning and fingerprints", "RAM exact and configured MinHash deduplication", "RAM seeded balanced sampling", "stage completed language", "one-at-a-time verified background transfer", "atomic manifest, run record, and quality report output"],
    }
