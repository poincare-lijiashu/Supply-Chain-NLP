# -*- coding: utf-8 -*-
"""v2 托寄物描述合成生成器（YAML 词库驱动）。

与 v1 (generate_data.py) 的差异：
  - 词库/规则/模板全部外置到 data/lexicon/v2/*.yaml，不再硬编码
  - 15 类目（class.txt 同步生成）
  - 四通道配比：A 裸词 30% / B 常规陈述 35% / C 带备注 20% / D 结构难例 15%
  - 长度四档分布：L1(2-6)35% L2(7-15)35% L3(16-30)20% L4(31-50)10%
  - 噪声层、holdout 硬隔离
用法：
  python -m data.generate_v2 --train 20000 --dev 2000 --seed 2026
支持子命令解析，输出到 data/raw_v2/{train,dev,test-*}.txt
"""
import argparse
import glob
import io
import os
import random
import re

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEX_DIR = os.path.join(PROJECT_ROOT, "data", "lexicon", "v2")
RAW_V2 = os.path.join(PROJECT_ROOT, "data", "raw_v2")

CLASS_LIST = ["文件证件", "服饰鞋包", "数码家电", "食品生鲜", "日用家居", "美妆个护",
              "医药保健", "母婴玩具", "图书文娱", "运动户外", "汽配五金工业品",
              "家具家装", "易碎品", "违禁限寄", "其他/拒识"]
N_CLS = len(CLASS_LIST)


# ---------------- 词库加载 ----------------
class Lexicon:
    """聚合 15 主库 + 2 别名层 + 品牌表。"""

    def __init__(self):
        self.items = {}          # cat -> set(词)
        self.syns = {}           # cat -> {主词: [别名...]}
        self.reject_patterns = {}  # cat -> [拒识模式(C分支)]
        self.brands = {}         # cat -> [品牌]
        self._load()

    def _load(self):
        for f in sorted(glob.glob(os.path.join(LEX_DIR, "cat_*.yaml"))):
            d = yaml.safe_load(open(f, encoding="utf-8"))
            cat = int(d["category"])
            self.items.setdefault(cat, set())
            self.syns.setdefault(cat, {})
            for sc in d.get("subcategories", []):
                for it in sc.get("items", []) or []:
                    if isinstance(it, dict):
                        w, ss = it["word"], list(it.get("syn", []))
                        self.items[cat].add(w)
                        if ss:
                            self.syns[cat][w] = ss
                    else:
                        self.items[cat].add(str(it))
                for p in sc.get("reject_patterns", []) or []:
                    self.reject_patterns.setdefault(cat, []).append(str(p))
        # 品牌
        B = yaml.safe_load(open(os.path.join(LEX_DIR, "brands.yaml"), encoding="utf-8"))
        for b in B["brands"]:
            self.brands.setdefault(b["cat"], []).append(str(b["name"]))
        # 别名层：word 若不在主库则作为补充主词加入 items；syn 合并且去重
        for af in ("colloquial_aliases.yaml", "colloquial_aliases_2.yaml"):
            A = yaml.safe_load(open(os.path.join(LEX_DIR, af), encoding="utf-8"))
            for e in A["aliases"]:
                cat = e["cat"]
                self.items[cat].add(e["word"])  # 补充主词（如 焊条/牙膏/自行车 等）
                self.syns.setdefault(cat, {}).setdefault(e["word"], [])
                for s in e["syn"]:
                    if s not in self.syns[cat][e["word"]]:
                        self.syns[cat][e["word"]].append(s)

    def get_final_words(self, cat):
        """主词 + 全部别名展开后的完整词形表（用于生成），返回 list（权重=词频近似均匀）。"""
        pool = list(self.items[cat])
        for w, ss in self.syns.get(cat, {}).items():
            pool.extend(ss)
        return pool

    def main_words(self, cat):
        return list(self.items[cat])


# ---------------- 通用槽位 ----------------
NOTES = ["明天必须到", "急", "加急", "麻烦快一点", "到付", "轻拿轻放", "不要压",
         "放驿站就行", "给我妈的", "小心点", "发顺丰", "周内到就行", "防压",
         "别放仓库太久", "周日之前到 谢谢", "里面还有别的", "上次寄过一样的",
         "地址电话看我上一单", "这单加钱走航空", "帮我问问多久能到", "超重了吗",
         "能不能明天送", "放丰巢", "保价500", "雨伞放箱底别压", "尽快发",
         "今天能发吗", "什么时候到", "放快递柜", "别放驿站", "用泡沫填满",
         "包装结实点", "别淋雨", "发票随货"]
SPEC = ["全新的", "用过的", "拆封的", "未拆封", "散装的", "真空包装的", "冰袋保鲜",
        "礼盒装", "拆过一点", "九成新", "要过期了", "别人送的", "自己囤的", "网购退的"]
GIFT_TARGETS = ["小孩", "娃", "爸妈", "老人", "客户", "朋友", "我妈", "孩子"]


# ---------------- 分类目量词表（P4 第七节：量词按类目适配） ----------------
QUANT_BY_CAT = {
    0: ["一份", "一本", "一沓", "一摞", "一袋", "一套", "{n}份", "{n}本", "{n}套", "{n}袋", "一箱", "{n}箱"],           # 文件证件
    1: ["一件", "一套", "一双", "一条", "一顶", "一件", "{n}件", "{n}套", "{n}双", "{n}条"],                              # 服饰鞋包
    2: ["一台", "一部", "一个", "一套", "{n}台", "{n}部", "{n}个", "{n}套"],                                             # 数码家电
    3: ["一箱", "一盒", "一袋", "一斤", "两斤", "一包", "{n}箱", "{n}盒", "{n}袋", "{n}斤", "{n}包", "一件", "一瓶"],      # 食品生鲜
    4: ["一个", "一包", "一盒", "一卷", "一把", "一套", "{n}个", "{n}包", "{n}盒", "{n}卷", "{n}把"],                    # 日用家居
    5: ["一瓶", "一支", "一盒", "一套", "{n}瓶", "{n}支", "{n}盒", "{n}套"],                                             # 美妆个护
    6: ["一盒", "一瓶", "一袋", "一罐", "{n}盒", "{n}瓶", "{n}袋", "{n}罐"],                                             # 医药保健
    7: ["一包", "一箱", "一个", "一套", "{n}包", "{n}箱", "{n}个", "{n}套"],                                             # 母婴玩具
    8: ["一本", "一箱", "一套", "一盒", "{n}本", "{n}箱", "{n}套", "{n}盒", "一支"],                                     # 图书文娱
    9: ["一个", "一副", "一根", "一条", "一顶", "{n}个", "{n}副", "{n}根"],                                              # 运动户外
    10: ["一个", "一批", "一套", "{n}个", "{n}件", "{n}批", "{n}套", "{n}根", "{n}卷", "{n}米", "一箱", "{n}箱"],        # 汽配五金
    11: ["一个", "一张", "一套", "{n}个", "{n}张", "{n}套", "一批"],                                                      # 家具家装
    12: ["一个", "一套", "{n}个", "{n}套", "一只"],                                                                        # 易碎品（简单量词）
    13: [],   # 违禁限寄 / 拒识：不用普通量词（纯对象或规则场景）
    14: [],
}
NUM = [2, 3, 5, 10, 20, 30, 50]
NUM_CN = ["两", "三", "五", "十"]


def _quant(lex, cat, rng):
    pool = QUANT_BY_CAT.get(cat, [])
    if not pool:
        return ""
    return rng.choice(pool).format(n=rng.choice(NUM))


# 词尾量词/数量模式（识别"词已自带量词/数量"的完整描述，避免再次套量词）
_QUANT_UNITS = "个件台本套双条根包瓶罐盒箱袋卷把顶副张米份支斤只部辆串颗粒枚块片棵盏扇"
# 词尾是"数字/汉字数字+单位"（如"一套""两双""3个"）→ 完整描述
_QUANT_SUFFIX_RE = re.compile(r'[一二两三四五六七八九十百千\d][' + _QUANT_UNITS + r']$')
# "一X"容器量词+名词（如"一箱书""两袋衣服""3盒饼干"）→ 完整描述（开头容器量词）
_QUANT_CONTAINER_RE = re.compile(r'^[一二两三四五六七八九十百千\d]+(箱|盒|袋|包|罐|瓶|捆|卷|杯|袋)子?[^，,和]{1,8}$')
# "数字+单位+套"（如"五件套""8件套"）→ 完整描述
_QUANT_SUFFIX2_RE = re.compile(r'[一二两三四五六七八九十百千\d](个|件|台|本|套|双|条|根|只|支|片|块|张)套$')
# 词中含"数字+单位"在词中（如"陶瓷碗4个" → 4个），也视为完整描述
_QUANT_MID_RE = re.compile(r'\d[' + _QUANT_UNITS + r']')


def _is_complete_descriptor(w):
    """词是否已是完整描述（自带量词/数量/括号注释/型号串/规格），是则不再套量词前缀。"""
    if "（" in w or "(" in w or "。" in w:
        return True
    if _QUANT_SUFFIX_RE.search(w) or _QUANT_SUFFIX2_RE.search(w) or _QUANT_MID_RE.search(w) or _QUANT_CONTAINER_RE.match(w):
        return True
    # 规格词识别：寸/升/毫升/ml/L/kg/斤/克/g/号/型/寸/平方/立方/件套 等（如"20寸登机箱""5件套"）
    if re.search(r'\d+(寸|升|毫升|ml|ML|L|公斤|公升|斤|克|g|kg|号|型|mm|cm|CM|W|KW|伏|V|安|A|件套|色|味)', w):
        return True
    # 完整描述：以"箱/袋/包/盒"等为容器量词开头的（如"一箱子易碎的"）
    if re.match(r'^[一二两三四五六七八九十百千\d]+(箱|袋子|包|盒|瓶|罐|大|小)子?[^，,。]*$', w) and len(w) >= 5:
        return True
    if len(w) <= 8 and re.match(r'^[\dA-Za-z/.\-+#]+$', w):
        return True
    return False


# 词池缓存（30万生成热点：_clean_pool 每次重建列表可去掉大量开销）
_POOL_CACHE = {}


def _clean_pool(lex, cat, max_len=12):
    """A/B 通道用：过滤注释词/自带量词/超长，只留干净的裸词（带缓存）。"""
    key = ("clean", cat, max_len)
    if key not in _POOL_CACHE:
        pool = [w for w in lex.get_final_words(cat)
                if "（" not in w and "(" not in w and "。" not in w
                and len(w) <= max_len and not _is_complete_descriptor(w)]
        _POOL_CACHE[key] = pool or [w for w in lex.main_words(cat) if "（" not in w]
    return _POOL_CACHE[key]


def _clean_pool_allow_quant(lex, cat, max_len=16):
    """C/D 通道用：允许自带量词的完整描述，仅过滤注释词与超长（带缓存）。"""
    key = ("allowq", cat, max_len)
    if key not in _POOL_CACHE:
        pool = [w for w in lex.get_final_words(cat)
                if "（" not in w and "(" not in w and "。" not in w and len(w) <= max_len]
        _POOL_CACHE[key] = pool or lex.main_words(cat)
    return _POOL_CACHE[key]


def _clean_reject_pool(lex):
    """14 拒识模式池：过滤注释词，其余保留（拒识本来就是完整短文本）。"""
    if "reject14" not in _POOL_CACHE:
        pool = [p for p in lex.reject_patterns.get(14, []) if "（" not in p and "(" not in p]
        _POOL_CACHE["reject14"] = pool or ["一箱东西"]
    return _POOL_CACHE["reject14"]


def _count_suffix(lex, cat, rng, word):
    """数量词放词后：'充电宝2个'。仅对≤6字的纯裸词追加，且单位白名单按类目限定，
    避免子类错配（'二胡两本'/'一顶脚踏车'）。13/14 不追加。"""
    if cat in (13, 14) or _is_complete_descriptor(word) or len(word) > 6:
        return word
    if cat == 0:
        units = ["份", "本", "套", "袋"]
    elif cat == 3:
        units = ["箱", "盒", "袋", "瓶", "斤"]
    elif cat == 6:
        units = ["盒", "瓶", "袋", "罐"]
    elif cat in (1, 9):
        units = ["件", "套", "双", "个"]
    else:
        units = ["个", "件", "盒", "箱"]
    unit = rng.choice(units)
    n = rng.choice(NUM)
    if rng.random() < 0.3:
        n = rng.choice(NUM_CN)
    return f"{word}{n}{unit}"


# 修饰语/规格按类目分组（P4：规格 SPEC 扩为分类目，防跨类错配）
_MODS_BY_CAT = {
    3: ["新鲜", "当季", "现摘", "进口", "国产", "散装", "真空包装的", "冰袋保鲜"],
    5: ["保湿", "清爽型", "滋润型", "温和", "小样", "正装"],
    6: ["全新", "未拆封", "正规渠道", "原装", "国产"],
    10: ["工业用", "国标", "非标", "原装", "散装", "全新", "二手"],
    12: ["玻璃", "陶瓷", "水晶", "手工"],
    0: ["盖章", "原件", "复印件", "扫描件", "公证过"],
    4: ["家用", "大号", "小号"],
    1: ["全新", "九成新", "二手的", "断码", "清仓"],
    13: [], 14: [],
}
_DEFAULT_MODS = ["全新", "旧的", "二手", "原装", "散装", "家用", "工业用", "带包装的"]

# SPEC 也分类目
_SPEC_BY_CAT = {
    3: ["冰袋保鲜", "真空包装的", "新鲜", "礼盒装", "散装的", "原产地的"],
    5: ["未拆封", "拆封闻过", "正装", "小样"],
    6: ["未拆封", "正规渠道", "2026新效期", "原装"],
    10: ["国标", "非标定做", "原装", "二手", "散装"],
    0: ["盖章的", "原件", "签过字的"],
    1: ["全新的", "九成新", "拆过吊牌", "断码"],
    12: ["轻拿轻放", "原包装", "易碎"],
    13: [], 14: [],
}
_DEFAULT_SPEC = ["全新的", "用过的", "拆封的", "未拆封", "散装的", "礼盒装", "拆过一点",
                 "九成新", "要过期了", "别人送的", "自己囤的", "网购退的"]


def _mod(lex, cat, rng):
    # 13 违禁/14 拒识不走修饰语（避免"家用金属钾"式错配）
    if cat in (13, 14):
        return ""
    pool = _MODS_BY_CAT.get(cat) or _DEFAULT_MODS
    return rng.choice(pool) if pool else "全新"


def _spec(lex, cat, rng):
    pool = _SPEC_BY_CAT.get(cat) or _DEFAULT_SPEC
    return rng.choice(pool) if pool else "全新"


# ---------------- 噪声层（P4 第五节：错别字5%/数字混写3%/口语省略3%/英文型号2%/标点2%，总12-15%） ----------------
# 同音/形近错别字（扩充 v1 TYPO；只改不影响类目标签的字）
_TYPO = {
    "箱": "厢", "瓶": "平", "份": "分", "装": "妆", "杯": "悲", "寄": "机",
    "衣": "医", "鞋": "协", "急": "集", "纸": "只", "机": "鸡", "镜": "境",
    "奶": "乃", "粉": "分", "茶": "茬", "壶": "胡", "碗": "晚", "锅": "过",
    "袋": "带", "盒": "合", "罐": "贯", "帽": "冒", "巾": "斤", "架": "驾",
    "图": "途", "证": "正", "书": "舒", "链": "练", "绳": "省", "钳": "钱",
    "喷": "盆", "漆": "七", "钙": "丐", "膏": "高", "酱": "浆", "饼": "丙",
}

# 数字混写（一→1；只替换"汉字数字+量词"组合，不碰个股词内部的数字）
_NUM_UNIT_WORDS = {"一": "1", "两": "2", "三": "3", "五": "5"}

# 口语省略/冗余标点（只加不改义）
_NOISE_PUNCT = ["~", "～", "!", "。", ",", ",,", " ", "  ", "！"]

# 数字混写：把"一箱"→"1箱"（词是"一箱东西"时不变义）

def _noise_typo(text, rng):
    hits = [ch for ch in _TYPO if ch in text]
    if not hits:
        return text
    rng.shuffle(hits)
    for ch in hits[:rng.choice([1, 1, 2])]:
        text = text.replace(ch, _TYPO[ch], 1)
    return text


def _noise_num(text, rng):
    for cn, ar in _NUM_UNIT_WORDS.items():
        for unit in ["箱", "盒", "件", "包", "套", "瓶", "个", "袋", "罐", "双", "台", "支"]:
            if f"{cn}{unit}" in text:
                text = text.replace(f"{cn}{unit}", f"{ar}{unit}", 1)
                return text
    return text


def _noise_punct(text, rng):
    p = rng.choice(_NOISE_PUNCT)
    return text + p


# 噪声总控：train 全量注入（目标总率≈12-15%，含错字5%），失败时保底文本不注入
NOISE_OPS = [("typo", 5, _noise_typo), ("num", 3, _noise_num),
             ("punct", 2, _noise_punct), ("omission", 3, None)]  # omission(简写) 走词库已含别名，不额外替换


def apply_noise(text, rng):
    """对生成文本注入噪声。仅注入不改变类目标签的噪声（P4：改判型句法走规则模板）。"""
    for name, rate, fn in NOISE_OPS:
        if name == "omission":
            continue  # 省略已由别名层体现
        if rng.randint(1, 100) <= rate and fn:
            text = fn(text, rng)
    return text


# ---------------- 通道生成 ----------------
def gen_bare(lex, cat, rng):
    """A 通道：裸词/极短 2-8 字。纯词50% / 词+数量30% / 词+短修饰20%。"""
    pool = _clean_pool(lex, cat, max_len=8)
    w = rng.choice(pool)
    r = rng.random()
    if r < 0.50:
        return w
    elif r < 0.80:
        return _count_suffix(lex, cat, rng, w)
    else:
        return f"{_mod(lex, cat, rng)}{w}"


def gen_normal(lex, cat, rng):
    """B 通道：常规陈述 35%（量词+物品/品牌+物品/简单动宾）。13/14 只出纯对象。"""
    if cat in (13, 14):
        pool = _clean_pool(lex, cat)
        return rng.choice(pool) if pool else _clean_pool_allow_quant(lex, cat)[0]
    pool = _clean_pool(lex, cat)
    w = rng.choice(pool)
    brands = lex.brands.get(cat, [])
    r = rng.random()
    if brands and r < 0.15:  # 品牌比例15%，降低类内错配（品牌×物品）
        b = rng.choice(brands)
        return f"{b}{w}" if isinstance(b, str) and len(b + w) <= 20 else w
    q = _quant(lex, cat, rng)
    if r < 0.65:
        return f"{q}{w}"
    if r < 0.85:
        return f"寄{q}{w}"
    return f"帮我寄{q}{w}"


def gen_note(lex, cat, rng):
    """C 通道：带备注/馈赠 20%。13/14 只出纯对象，12 用易碎备注（且不重复易碎词）。"""
    pool = _clean_pool(lex, cat)
    w = rng.choice(pool)
    if cat in (13, 14):
        return w
    if cat == 12:
        # 12 类备注槽固定易碎词，base 用它拼一次，不重复
        base = w
        note = rng.choice(["轻拿轻放", "小心轻放", "易碎", "原包装"])
        return f"{base}，{note}"
    base = rng.choice([gen_normal(lex, cat, rng), f"{w}{_spec(lex, cat, rng)}",
                       f"{_quant(lex, cat, rng)}{w}", w])
    r = rng.random()
    if r < 0.30:
        tgt = rng.choice(GIFT_TARGETS)
        return f"给{tgt}寄的{base}"
    if r < 0.75:
        return f"{base}，{rng.choice(NOTES)}"
    return f"{base}，{_spec(lex, cat, rng)}，{rng.choice(NOTES)}"


# ================= D 通道：P3 裁决手册规则难例 =================
# 规则场景返回 (text, label)，支持跨类标签（混填互换/容器/夹带/拒识）。
_FILLERS = ["路上得走三天", "我打了三次包", "里面装了缓冲气垫", "封箱前拍了照",
            "保价两百块", "这是补寄的", "备用件也放里面", "备注写不清了",
            "上次碎过一个", "麻烦轻拿轻放"]


def _other_cat(lex, src, rng):
    """随机取一个与 src 不同的普通类作为混填第二件来源（13/14 不混入普通件）。"""
    pool = [c for c in range(N_CLS) if c != src and c not in (13, 14)]
    return rng.choice(pool)


def _pick(lex, cat, rng, allow_quant=False):
    fn = _clean_pool_allow_quant if allow_quant else _clean_pool
    return rng.choice(fn(lex, cat))


def _sc_R11_mix(lex, src, rng):
    """混填首项（R11）：'a和b一起寄' label 跟 a；30% 词序互换做最小对比对。"""
    a = _pick(lex, src, rng)
    o = _other_cat(lex, src, rng)
    b = _pick(lex, o, rng)
    tmpl = rng.choice(["{a}和{b}一起寄", "寄{a}和{b}", "{a}，另外还有{b}",
                       "{a}跟{b}放一个箱", "{a}和{b}各{n}件", "先寄{a}，{b}下次说"])
    n = rng.choice([2, 3, 5])
    if rng.random() < 0.30:
        return tmpl.format(a=b, b=a, n=n), o
    return tmpl.format(a=a, b=b, n=n), src


def _sc_R12_remote(lex, src, rng):
    """远距申报（R12）：申报词与物品隔干扰短语。"""
    a = _pick(lex, src, rng)
    o = _other_cat(lex, src, rng)
    b = _pick(lex, o, rng)
    tmpl = rng.choice(["第一件是{a}，{f}，第二件是{b}，按第一件申报",
                       "{a}先寄，{f}，{b}下一批再说"])
    return tmpl.format(a=a, f=rng.choice(_FILLERS), b=b), src


def _sc_R7_container(lex, src, rng):
    """容器易碎（R7/R8）：src 的内容物用玻璃/陶瓷容器线索包裹 → 12。
    50% 产出正常（对照），50% 产出 →12。"""
    a = _pick(lex, src, rng)
    cont = rng.choice(["玻璃瓶装的", "陶瓷罐装的", "玻璃盒装的", "水晶盒装的"])
    r = rng.random()
    if r < 0.5:
        tmpl = rng.choice(["{c}{f}，里面是{a}", "{c}{a}", "{c}{a}，{f}"])
        return tmpl.format(c=cont, a=a, f=rng.choice(_FILLERS)), 12
    return a, src


def _sc_R9_negation(lex, src, rng):
    """否定翻转（R9）：显式声明非易碎 → 回内容物类目。"""
    a = _pick(lex, src, rng)
    tmpl = rng.choice(["不是易碎品，{c}装的{a}而已", "不用{c}了，普通袋子装的{a}",
                       "别看是{c}装的{a}，不算易碎"])
    return tmpl.format(c=rng.choice(["玻璃瓶", "陶瓷罐", "玻璃盒"]), a=a), src


def _sc_R1_redline(lex, src, rng):
    """红线夹带（R1）：普通件里夹带违禁语义 → 13。"""
    a = _pick(lex, src, rng)
    hot = _pick(lex, 13, rng)
    tmpl = rng.choice(["{a}里藏了{hot}", "{a}和{hot}一起装", "{a}，里面还有{hot}",
                       "一箱{a}，夹带着{hot}"])
    return tmpl.format(a=a, hot=hot), 13


def _sc_R27_reject(lex, src, rng):
    """拒识（R27-29）：零信息/杂物/异常 → 14。"""
    return rng.choice(_clean_reject_pool(lex)), 14


RULE_SCENES = [_sc_R11_mix, _sc_R12_remote, _sc_R7_container,
               _sc_R9_negation, _sc_R1_redline, _sc_R27_reject]


def gen_other(lex, cat, rng):
    """D 通道：规则难例，返回 (text, label)。"""
    fn = rng.choice(RULE_SCENES)
    return fn(lex, cat, rng)


# 通道分派（占比对应 P4_EXPRESSION_DESIGN 第五节）
# A/B/C 返回 str（label=cat），D 返回 (str, label)
_CHANNELS = [("A", 0.30, gen_bare), ("B", 0.35, gen_normal),
             ("C", 0.20, gen_note), ("D", 0.15, gen_other)]


def _one_sample(lex, cat, rng):
    """返回 (text, label)。A/B/C 的 label 即 cat；D 规则场景可跨类。"""
    r = rng.random()
    acc = 0.0
    for _, ratio, fn in _CHANNELS:
        acc += ratio
        if r < acc:
            out = fn(lex, cat, rng)
            return out if isinstance(out, tuple) else (out, cat)
    out = gen_other(lex, cat, rng)
    return out if isinstance(out, tuple) else (out, cat)


# 长度四档目标配比（P4 第六节：L1 35% / L2 35% / L3 20% / L4 10%）
LEN_BUCKETS = [("L1", 0, 6, 0.35), ("L2", 7, 15, 0.35),
               ("L3", 16, 30, 0.20), ("L4", 31, 60, 0.10)]


def _len_bucket(text):
    n = len(text)
    for name, lo, hi, _ in LEN_BUCKETS:
        if lo <= n <= hi:
            return name
    return "L4"


def generate_set(lex, n_per_cat, seed, holdout=None, noise=True):
    """生成 15 类 × n_per_cat 条，返回 [(text, label)]。
    按标签配额（D 通道跨类样本归位到对应类）；长度软配额：L1/L2 超 1.3 倍跳过，
    L3/L4 由 D 通道长句填充，总量不足时回退到软上限之外。
    noise=True 时注入噪声层（train 用）；dev/test 建议 noise=False 保持评测语义干净。"""
    rng = random.Random(seed)
    quota = {c: n_per_cat for c in range(N_CLS)}
    filled = {c: 0 for c in range(N_CLS)}
    len_filled = {k: 0 for k in ("L1", "L2", "L3", "L4")}
    len_soft = {k: max(1, int(n_per_cat * N_CLS * r)) for k, _, _, r in LEN_BUCKETS}
    # L4 若实在凑不出长句，允许超配额少量（上限*2）
    len_soft["L4"] = int(len_soft["L4"] * 2)
    rows = []
    src = 0
    guard = 0
    total = n_per_cat * N_CLS
    hard = max(total * 400, 60000)
    while len(rows) < total and guard < hard:
        guard += 1
        t, lab = _one_sample(lex, src, rng)
        src = (src + 1) % N_CLS  # 轮询源类，保证进度
        if holdout and any(h in t for h in holdout):
            continue
        if filled[lab] >= quota[lab]:
            continue  # 该类配额已满
        b = _len_bucket(t)
        if b in ("L1", "L2") and len_filled[b] >= len_soft[b]:
            continue  # 短档超量，拒绝（保持 35/35）
        if noise:
            t = apply_noise(t, rng)
        rows.append((t, lab))
        filled[lab] += 1
        len_filled[b] += 1
    rng.shuffle(rows)
    return rows


# ================= P5 六把评测尺：NEAR/FAR/REJECT/SIM 专用生成 =================
# 读 holdout.yaml；NEAR/FAR 产出裸词+短句（3-5句式每词），REJECT 产拒识，
# SIM 按业务不均衡配比（P5 冻结值）生成。
_HOLDOUT_FILE = os.path.join(LEX_DIR, "holdout.yaml")

# SIM 业务仿真配比（P5 冻结：日用/服饰/食品各14%，数码/文件各9%，工业8%，
# 美妆6%，易碎5%，文娱/母婴/运动/医药/家具各4%，违禁3%，拒识6%）
SIM_RATIO = {4: 0.14, 1: 0.14, 3: 0.14, 2: 0.09, 0: 0.09, 10: 0.08, 5: 0.06,
             12: 0.05, 8: 0.04, 7: 0.04, 9: 0.04, 6: 0.04, 11: 0.04, 13: 0.03, 14: 0.06}


def load_holdout():
    h = yaml.safe_load(open(_HOLDOUT_FILE, encoding="utf-8"))
    far = h.get("FAR", [])
    near = h.get("NEAR", [])
    return far, near


def _eval_shapes(word, rng):
    """每个 holdout 词配 3~5 种形态：至少 2 个是纯裸词，其余为带短数量/短修饰。
    确保 FAR/NEAR 评测以裸词为主，前缀污染低。"""
    shapes = [word, word]  # 至少两个纯裸词
    if rng.random() < 0.6:
        shapes.append(word + rng.choice(["2个", "一件", "一箱"]))
    if rng.random() < 0.5:
        shapes.append(rng.choice(["寄", "帮我寄"]) + word)
    if rng.random() < 0.4:
        shapes.append(rng.choice(["一箱", "一件", "两个"]) + word)
    return shapes


def generate_far(lex, seed, per_word=3):
    """test-FAR：FAR 词裸词泛化（训练0出现）。"""
    far, _ = load_holdout()
    rng = random.Random(seed)
    rows = []
    for e in far:
        w, cat = e["word"], e["cat"]
        for _ in range(per_word):
            t = _eval_shapes(w, rng)[0]
            if rng.random() < 0.5:
                t = rng.choice(_eval_shapes(w, rng))
            rows.append((t, cat))
    rng.shuffle(rows)
    return rows


def generate_near(lex, seed, per_word=3):
    """test-NEAR：NEAR 词（近义未登录，训练只教 train_teach）。"""
    _, near = load_holdout()
    rng = random.Random(seed)
    rows = []
    for e in near:
        w, cat = e["word"], e["cat"]
        for _ in range(per_word):
            t = rng.choice(_eval_shapes(w, rng))
            rows.append((t, cat))
    rng.shuffle(rows)
    return rows


def generate_reject(lex, seed, n=500):
    """test-REJECT：14 类拒识（零信息/乱码/外文/脏话/杂物），全部标签14。"""
    rng = random.Random(seed)
    pool = _clean_reject_pool(lex)
    rows = [(rng.choice(pool), 14) for _ in range(n)]
    return rows


def generate_sim(lex, seed, n=10000, n_per_cat=200):
    """test-SIM：按业务不均衡配比 + 混入噪声。按 SIM_RATIO 分配每类目标数。"""
    rng = random.Random(seed)
    rows = []
    target = {c: max(1, int(n * SIM_RATIO.get(c, 0.04))) for c in range(N_CLS)}
    s = sum(target.values())
    target = {c: max(1, int(v * n / s)) for c, v in target.items()}
    for c, cnt in target.items():
        got = 0
        guard = 0
        # SIM 里 D 通道跨类样本也照收（其 label 即实际），故按实际产出计数到目标类
        while got < cnt and guard < cnt * 60:
            guard += 1
            t, lab = _one_sample(lex, c, rng)
            if lab == c:  # 归属按产出 label 保持配比稳定
                t = apply_noise(t, rng)
                rows.append((t, c))
                got += 1
    rng.shuffle(rows)
    return rows


def write_split(rows, out_path):
    with io.open(out_path, "w", encoding="utf-8") as f:
        for t, c in rows:
            f.write(f"{t}\t{c}\n")
    print(f"  {os.path.basename(out_path)}: {len(rows)} 条")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=180000)
    ap.add_argument("--dev", type=int, default=15000)
    ap.add_argument("--id", type=int, default=15000, dest="n_id")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sim", type=int, default=10000, dest="n_sim")
    args = ap.parse_args()

    os.makedirs(RAW_V2, exist_ok=True)
    # class.txt
    with io.open(os.path.join(RAW_V2, "class.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASS_LIST) + "\n")

    lex = Lexicon()
    far, near = load_holdout()
    holdset = set([e["word"] for e in far] + [e["word"] for e in near])
    print(f"词库加载: 主词 {sum(len(v) for v in lex.items.values())} | "
          f"品牌 {sum(len(v) for v in lex.brands.values())} | 拒识 {sum(len(v) for v in lex.reject_patterns.values())}")
    print(f"holdout: FAR {len(far)} / NEAR {len(near)}（训练硬隔离）")

    plan = [("train", args.train, args.seed, True),
            ("dev", args.dev, args.seed + 1, False),
            ("test-ID", args.n_id, args.seed + 2, False)]
    for name, n, seed, noise in plan:
        n_per = n // N_CLS
        rows = generate_set(lex, n_per, seed, holdout=holdset, noise=noise)
        write_split(rows, os.path.join(RAW_V2, f"{name}.txt"))

    # 四把专用评测尺（不受 holdout 隔离限制，NEAR/FAR 就是 holdout 词）
    write_split(generate_far(lex, args.seed + 3), os.path.join(RAW_V2, "test-FAR.txt"))
    write_split(generate_near(lex, args.seed + 4), os.path.join(RAW_V2, "test-NEAR.txt"))
    write_split(generate_reject(lex, args.seed + 5), os.path.join(RAW_V2, "test-REJECT.txt"))
    write_split(generate_sim(lex, args.seed + 6, n=args.n_sim), os.path.join(RAW_V2, "test-SIM.txt"))
    print("完成。输出目录 data/raw_v2/（train/dev/test-ID/test-NEAR/FAR/REJECT/SIM）")


if __name__ == "__main__":
    main()