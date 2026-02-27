import os
import math
import re
import time
import unicodedata
import functools
import difflib

import pandas as pd
import requests
from pykakasi import kakasi
from flask import Flask, render_template, request, session, redirect, url_for

app = Flask(__name__)
app.secret_key = "your-secret-key"

# =========================
# 楽天レシピ API
# =========================
RAKUTEN_APP_ID = os.environ.get("RAKUTEN_APP_ID", "1083313192545553191")
RAKUTEN_CATEGORY_LIST_URL = "https://app.rakuten.co.jp/services/api/Recipe/CategoryList/20170426"
RAKUTEN_CATEGORY_RANKING_URL = "https://app.rakuten.co.jp/services/api/Recipe/CategoryRanking/20170426"


def _rakuten_get(url: str, params: dict):
    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()
    return res.json()


# =========================
# パス / CSV
# =========================
base_dir = os.path.dirname(__file__)
CSV_PATH = os.path.join(base_dir, "data.csv")

df = pd.read_csv(CSV_PATH, encoding="cp932")
food_col = "食品名(100g当たり)"

numeric_cols = [
    "エネルギー", "たんぱく質", "脂質", "炭水化物",
    "食物繊維総量", "食塩相当量", "カ ル シ ウ ム",
    "鉄", "ビタミンA", "ビタミンC"
]
cols = numeric_cols + ["備　　考"]

units = {
    "エネルギー": "kcal",
    "たんぱく質": "g",
    "脂質": "g",
    "炭水化物": "g",
    "食物繊維総量": "g",
    "食塩相当量": "g",
    "カ ル シ ウ ム": "mg",
    "鉄": "mg",
    "ビタミンA": "µg",
    "ビタミンC": "mg",
    "備　　考": ""
}

# =========================
# 数値
# =========================
def safe_float(v):
    try:
        return float(v)
    except:
        return 0.0

def floor2(v):
    return math.floor(v * 100) / 100


# =========================
# 正規化
# =========================
def norm_basic(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text)
    return s.lower().strip()


_kks = kakasi()
_kks.setMode("J", "H")  # 漢字→ひらがな
_kks.setMode("K", "H")  # カタカナ→ひらがな
_kks.setMode("H", "H")  # ひらがな→ひらがな
_kks.setMode("a", "a")  # 英数字
_conv = _kks.getConverter()

def norm_reading(text: str) -> str:
    return _conv.do(norm_basic(text))


# =========================
# 同義語辞書（alias -> canon）
# ※文字列replace連発しない（トークン単位で1回だけ）
# =========================
SYNONYMS = {
    # 鶏
    "鶏もも肉": ["鶏もも", "もも肉", "もも", "とりもも", "とりもも肉"],
    "鶏むね肉": ["鶏むね", "胸肉", "むね肉", "むね", "とりむね", "とりむね肉"],
    "鶏ささみ": ["ささみ", "ササミ"],
    "鶏ひき肉": ["鶏ミンチ", "鶏挽肉", "鶏挽き肉", "ミンチ(鶏)"],
    "鶏肉": ["とり肉", "とりにく", "鶏"],

    # 豚
    "豚こま": ["豚こま肉", "豚小間", "豚コマ", "こま切れ(豚)", "こま肉(豚)"],
    "豚バラ": ["豚ばら", "豚ばら肉", "豚バラスライス", "豚ばらスライス", "ばら肉(豚)"],
    "豚ロース": ["豚ろーす", "豚ロース肉", "豚ローススライス", "ロース(豚)"],
    "豚ひき肉": ["豚ミンチ", "豚挽肉", "豚挽き肉", "ミンチ(豚)"],
    "豚肉": ["ぶた肉", "ぶたにく", "豚"],

    # 牛
    "牛こま": ["牛こま肉", "牛小間", "牛コマ", "こま切れ(牛)", "こま肉(牛)"],
    "牛バラ": ["牛ばら", "牛ばら肉", "牛バラスライス", "ばら肉(牛)"],
    "牛ロース": ["牛ろーす", "牛ロース肉", "牛ローススライス", "ロース(牛)"],
    "牛ひき肉": ["牛ミンチ", "牛挽肉", "牛挽き肉", "ミンチ(牛)"],
    "牛肉": ["ぎゅう肉", "ぎゅうにく", "牛", "うし", "和牛肉", "和牛"],

    # 部位/形状
    "バラ": ["ばら", "バラ肉"],
    "ロース": ["ろーす", "ロース肉"],
    "ひき肉": ["ミンチ", "挽肉", "挽き肉"],

    # 野菜/調味料（例）
    "じゃがいも": ["ジャガイモ", "馬鈴薯", "ばれいしょ", "新じゃが", "男爵", "メークイン"],
    "たまねぎ": ["玉ねぎ", "タマネギ"],
    "にんじん": ["人参", "ニンジン"],
    "ねぎ": ["長ねぎ", "青ねぎ", "万能ねぎ", "ネギ"],
    "しょうゆ": ["醤油", "しょう油", "正油"],
    "みそ": ["味噌", "ミソ"],
    "さとう": ["砂糖", "シュガー"],
    "こめ": ["米", "ごはん", "ご飯", "白米"],
    "たまご": ["卵", "玉子", "たまご"],
}

ALIAS_BASIC = {}
ALIAS_READING = {}

def _add_alias(canon: str, alias: str):
    ALIAS_BASIC[norm_basic(alias)] = canon
    ALIAS_READING[norm_reading(alias)] = canon

for canon, aliases in SYNONYMS.items():
    _add_alias(canon, canon)
    for a in aliases:
        _add_alias(canon, a)

def canon_token_basic(tok: str) -> str:
    t = norm_basic(tok)
    return norm_basic(ALIAS_BASIC.get(t, tok))

def canon_token_reading(tok: str) -> str:
    t = norm_reading(tok)
    return norm_reading(ALIAS_READING.get(t, tok))


# =========================
# 食品検索用（basic）
# =========================
df[food_col] = df[food_col].astype(str)
df["__norm_name__"] = df[food_col].apply(norm_basic)

# =========================
# トークン化（スペース区切り優先）
# =========================
_token_splitter = re.compile(r"[^0-9a-zA-Zぁ-んァ-ヶ一-龠]+")
_space_splitter = re.compile(r"[ \u3000]+")
_noise_to_space = re.compile(r"[\[\]<>（）()\{\}「」『』【】・/\\:;.,|]+")

# 材料に出にくい分類語は除外（必要なら調整）
STOP_TOKENS_BASIC = {
    "畜肉類", "食肉類", "可食部", "加工", "冷凍", "冷蔵",
    "赤肉", "白肉", "生",
    "うし", "ぶた", "とり",
    # ※ここは好み：分類語を当てたいなら消してOK
    "いも類", "塊茎", "皮つき",
}

def split_by_space_like(text: str) -> list[str]:
    s = norm_basic(text)
    s = _noise_to_space.sub(" ", s)
    s = s.replace("　", " ")
    return [p.strip() for p in _space_splitter.split(s) if p.strip()]

def extract_ingredient_tokens(names: list[str]) -> dict:
    basic_set = set()
    reading_set = set()

    for name in names:
        chunks = split_by_space_like(name) or [name]

        for chunk in chunks:
            # basic
            for t in _token_splitter.split(norm_basic(chunk)):
                t = t.strip()
                if not t:
                    continue
                tb = canon_token_basic(t)
                if tb in STOP_TOKENS_BASIC:
                    continue
                if len(tb) == 1 and tb not in {"牛", "豚", "鶏", "米", "卵"}:
                    continue
                basic_set.add(tb)

            # reading
            for t in _token_splitter.split(norm_reading(chunk)):
                t = t.strip()
                if not t:
                    continue
                tr = canon_token_reading(t)
                if len(tr) == 1 and tr not in {"うし", "ぶた", "とり", "こめ", "たまご"}:
                    continue
                reading_set.add(tr)

    basic_tokens = sorted(basic_set, key=len, reverse=True)
    reading_tokens = sorted(reading_set, key=len, reverse=True)
    return {"basic_tokens": basic_tokens, "reading_tokens": reading_tokens}


# =========================
# 楽天カテゴリ / ランキング取得
# =========================
@functools.lru_cache(maxsize=1)
def fetch_all_recipe_categories() -> list[dict]:
    result = []
    for ct in ("large", "medium", "small"):
        data = _rakuten_get(RAKUTEN_CATEGORY_LIST_URL, {
            "applicationId": RAKUTEN_APP_ID,
            "format": "json",
            "categoryType": ct,
        })
        result.extend(data["result"][ct])
        time.sleep(0.25)
    return result

@functools.lru_cache(maxsize=2048)
def fetch_recipes_by_category(category_id: str) -> list[dict]:
    data = _rakuten_get(RAKUTEN_CATEGORY_RANKING_URL, {
        "applicationId": RAKUTEN_APP_ID,
        "format": "json",
        "categoryId": category_id,
    })
    return data.get("result", [])


# =========================
# カテゴリ横断：候補を集める
# =========================
CATEGORY_HINTS = [
    "牛", "豚", "鶏", "肉", "ビーフ", "ポーク", "チキン",
    "ステーキ", "焼肉", "ロースト", "ハンバーグ",
    "丼", "どんぶり", "カレー", "シチュー", "煮込み", "炒め",
    "簡単", "時短", "洋食", "和食", "中華", "おかず", "おつまみ",
    "じゃがいも", "ポテト", "サラダ",
]

def pick_cross_categories(tokens_basic: list[str], max_categories: int = 30) -> list[dict]:
    cats = fetch_all_recipe_categories()
    hints_norm = [norm_basic(x) for x in CATEGORY_HINTS]
    tokens_norm = [norm_basic(t) for t in tokens_basic]

    scored = []
    for c in cats:
        name = norm_basic(c.get("categoryName", ""))
        score = 0.0

        for t in tokens_norm:
            if t and t in name:
                score += 2.0

        for h in hints_norm:
            if h and h in name:
                score += 1.0

        if c.get("parentCategoryId"):
            score += 0.2

        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    picked = []
    seen = set()
    for _, c in scored:
        cid = str(c.get("categoryId"))
        if cid in seen:
            continue
        seen.add(cid)
        picked.append(c)
        if len(picked) >= max_categories:
            break

    return picked


def collect_candidate_recipes(tokens_basic: list[str], min_pool: int = 140) -> tuple[list[dict], list[dict], list[str]]:
    """
    return: (pool, used_categories, errors)
    errors に例外やレスポンス異常を入れて画面に出せるようにする
    """
    used_categories = pick_cross_categories(tokens_basic, max_categories=30)

    pool = []
    seen_key = set()
    errors = []

    def _add_recipe(r: dict, cid: str):
        url = r.get("recipeUrl")
        title = r.get("recipeTitle", "")
        key = url or (title + ":" + cid)
        if not key:
            return
        if key in seen_key:
            return
        seen_key.add(key)
        pool.append(r)

    # main
    for c in used_categories:
        cid = str(c.get("categoryId"))
        try:
            recipes = fetch_recipes_by_category(cid)
            time.sleep(0.12)  # 叩きすぎ対策（軽いレート制限回避）
        except Exception as e:
            errors.append(f"category {cid} error: {repr(e)}")
            continue

        if not isinstance(recipes, list):
            errors.append(f"category {cid} result not list: {type(recipes)}")
            continue

        for r in recipes:
            _add_recipe(r, cid)

        if len(pool) >= min_pool:
            break

    # extra fallback
    if len(pool) < min_pool:
        extra_cats = pick_cross_categories(["肉", "簡単", "丼", "洋食", "和食"], max_categories=30)

        for c in extra_cats:
            cid = str(c.get("categoryId"))
            try:
                recipes = fetch_recipes_by_category(cid)
                time.sleep(0.12)
            except Exception as e:
                errors.append(f"extra category {cid} error: {repr(e)}")
                continue

            if not isinstance(recipes, list):
                errors.append(f"extra category {cid} result not list: {type(recipes)}")
                continue

            for r in recipes:
                _add_recipe(r, cid)

            if len(pool) >= min_pool:
                break

        # 表示用カテゴリにも追加（重複除外）
        seen_c = {str(x.get("categoryId")) for x in used_categories}
        for c in extra_cats:
            if str(c.get("categoryId")) not in seen_c:
                used_categories.append(c)
                seen_c.add(str(c.get("categoryId")))

    return pool, used_categories, errors


# =========================
# 合計計算
# =========================
def calc_total(cart):
    total = {col: 0.0 for col in numeric_cols}
    for item in cart:
        name = item["name"]
        gram = safe_float(item["gram"])
        ratio = gram / 100.0

        match = df[df[food_col] == name]
        if match.empty:
            continue

        row = match.iloc[0]
        for col in numeric_cols:
            total[col] += safe_float(row[col]) * ratio

    return {col: floor2(total[col]) for col in numeric_cols}


# =========================
# あいまいスコア（近いレシピ）
# =========================
def fuzzy_score(tokens: list[str], target: str) -> float:
    if not tokens or not target:
        return 0.0
    bests = []
    for t in tokens[:20]:
        if not t:
            continue
        if t in target:
            bests.append(1.0)
            continue
        r = difflib.SequenceMatcher(None, t, target).ratio()
        bests.append(r)
    bests.sort(reverse=True)
    return sum(bests[:5])


# =========================
# 画面：検索
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if "cart" not in session:
        session["cart"] = []

    cart = session["cart"]

    selected_name = None
    result = None
    candidates = None
    message = None

    if request.method == "POST":
        if "select_name" in request.form:
            selected_name = request.form["select_name"]
            row = df[df[food_col] == selected_name].iloc[0]
            result = [{"name": col, "value": f"{row[col]} {units.get(col,'')}".strip()} for col in cols]
            message = f"{selected_name} の栄養素"
        else:
            food_name = request.form.get("food_name", "").strip()
            keywords = [norm_basic(k) for k in food_name.replace("　", " ").split() if k]

            filtered = df
            for kw in keywords:
                filtered = filtered[filtered["__norm_name__"].str.contains(kw, na=False)]

            if filtered.empty:
                message = "該当する食品がありません。"
            elif len(filtered) == 1:
                selected_name = filtered.iloc[0][food_col]
                row = filtered.iloc[0]
                result = [{"name": col, "value": f"{row[col]} {units.get(col,'')}".strip()} for col in cols]
                message = f"{selected_name} の栄養素"
            else:
                candidates = filtered[food_col].unique().tolist()
                message = "候補があります。"

    return render_template(
        "見た目.html",
        selected_name=selected_name,
        result=result,
        candidates=candidates,
        message=message,
        cart=cart
    )


# =========================
# カート追加
# =========================
@app.route("/add/<name>", methods=["POST"])
def add_cart(name):
    gram = safe_float(request.form.get("gram", 100))
    cart = session.get("cart", [])
    cart.append({"name": name, "gram": gram})
    session["cart"] = cart
    return redirect(url_for("index"))


# =========================
# 合計画面
# =========================
@app.route("/total", methods=["GET", "POST"])
def total_page():
    cart = session.get("cart", [])
    total = calc_total(cart)

    required = {}
    diff = {}
    bmi = None
    kcal_factor = None

    if request.method == "POST":
        if "delete_selected" in request.form:
            del_indexes = list(map(int, request.form.getlist("delete_item")))
            cart = [item for i, item in enumerate(cart) if i not in del_indexes]
            session["cart"] = cart
            total = calc_total(cart)

        elif "delete_all" in request.form:
            session["cart"] = []
            cart = []
            total = calc_total(cart)

        elif "calc_required" in request.form:
            height = safe_float(request.form.get("height", 0))
            weight = safe_float(request.form.get("weight", 0))

            if height > 0 and weight > 0:
                h_m = height / 100
                bmi = floor2(weight / (h_m * h_m))

                if bmi < 18.5:
                    kcal_factor = 35
                elif bmi < 25:
                    kcal_factor = 30
                else:
                    kcal_factor = 25

                required = {
                    "エネルギー": floor2(weight * kcal_factor),
                    "たんぱく質": floor2(weight * 1.2),
                    "脂質": floor2(weight * 0.8),
                    "炭水化物": floor2(weight * 4),
                    "食物繊維総量": floor2(weight * 0.24),
                    "食塩相当量": 6,
                    "カ ル シ ウ ム": 650,
                    "鉄": 7,
                    "ビタミンA": 650,
                    "ビタミンC": 100,
                }

                diff = {col: floor2(required[col] - total[col]) for col in numeric_cols}

    return render_template(
        "合計.html",
        cart=cart,
        total=total,
        required=required,
        diff=diff,
        cols=numeric_cols,
        bmi=bmi,
        kcal_factor=kcal_factor
    )


# =========================
# レシピ提案
# - カテゴリ横断で候補を集める
# - レシピ名 + 材料 を参照
# - トークンは OR（1語でも部分一致でOK）
# - 0件なら近いレシピ2件
# - 失敗理由を debug として画面へ
# =========================
@app.route("/recipes", methods=["GET"])
def recipes_page():
    cart = session.get("cart", [])
    if not cart:
        return render_template(
            "recipes.html",
            ingredients=[],
            tokens=[],
            recipes=[],
            categories=[],
            message="カートが空です。まず食材を追加してください。",
            debug_pool_count=0,
            debug_errors=[]
        )

    ingredients = [item["name"] for item in cart]
    token_pack = extract_ingredient_tokens(ingredients)
    basic_tokens = token_pack["basic_tokens"]
    reading_tokens = token_pack["reading_tokens"]

    pool, used_categories, api_errors = collect_candidate_recipes(basic_tokens, min_pool=140)

    hit_list = []
    fuzzy_list = []

    for r in pool:
        title = r.get("recipeTitle", "")
        mats_list = r.get("recipeMaterial", []) or []
        mats = " ".join(mats_list)

        target_basic = norm_basic(title + " " + mats)
        target_read = norm_reading(title + " " + mats)

        matched_tokens = []

        # ★ OR条件：トークン1つでも部分一致でOK
        for t in basic_tokens:
            if t and t in target_basic:
                matched_tokens.append(t)

        for t in reading_tokens:
            if t and t in target_read:
                matched_tokens.append(t)

        item = {
            "title": title,
            "url": r.get("recipeUrl", ""),
            "image": r.get("foodImageUrl", ""),
            "time": r.get("recipeIndication", ""),
            "cost": r.get("recipeCost", ""),
            "materials": mats_list,
            "category": "",
        }

        if matched_tokens:
            uniq = list(dict.fromkeys(matched_tokens))
            item["hit"] = len(uniq)
            item["hit_words"] = uniq[:10]
            hit_list.append(item)
        else:
            fb = fuzzy_score(basic_tokens, target_basic)
            fr = fuzzy_score(reading_tokens, target_read)
            item["_fuzzy"] = fb + fr
            fuzzy_list.append(item)

    hit_list.sort(key=lambda x: (x.get("hit", 0), len(x.get("materials", []))), reverse=True)

    message = None
    if not hit_list:
        fuzzy_list.sort(key=lambda x: x.get("_fuzzy", 0.0), reverse=True)
        picked = fuzzy_list[:2]
        for p in picked:
            p.pop("_fuzzy", None)
            p["hit"] = 0
            p["hit_words"] = []
        hit_list = picked
        message = "一致がなかったため、近いレシピを2件表示しています。"

    return render_template(
        "recipes.html",
        ingredients=ingredients,
        tokens=basic_tokens[:20],
        recipes=hit_list[:30],
        categories=used_categories[:12],
        message=message,
        debug_pool_count=len(pool),
        debug_errors=api_errors[:6]
    )


# =========================
# API疎通チェック（ブラウザで /debug_api）
# =========================
@app.route("/debug_api")
def debug_api():
    out = {
        "RAKUTEN_APP_ID_head": (RAKUTEN_APP_ID[:6] + "..." if isinstance(RAKUTEN_APP_ID, str) and len(RAKUTEN_APP_ID) > 6 else str(RAKUTEN_APP_ID)),
        "category_list_ok": False,
        "category_list_error": None,
        "large_count": 0,
        "ranking_ok": False,
        "ranking_error": None,
        "ranking_count": 0,
    }
    try:
        data = _rakuten_get(RAKUTEN_CATEGORY_LIST_URL, {
            "applicationId": RAKUTEN_APP_ID,
            "format": "json",
            "categoryType": "large",
        })
        out["category_list_ok"] = True
        large = data.get("result", {}).get("large", [])
        out["large_count"] = len(large)

        if large:
            cid = str(large[0].get("categoryId"))
            rank = _rakuten_get(RAKUTEN_CATEGORY_RANKING_URL, {
                "applicationId": RAKUTEN_APP_ID,
                "format": "json",
                "categoryId": cid,
            })
            out["ranking_ok"] = True
            out["ranking_count"] = len(rank.get("result", []))
    except Exception as e:
        msg = repr(e)
        if not out["category_list_ok"]:
            out["category_list_error"] = msg
        else:
            out["ranking_error"] = msg
    return out


if __name__ == "__main__":
    app.run(debug=True)
