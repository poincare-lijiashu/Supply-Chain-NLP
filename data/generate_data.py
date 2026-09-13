# -*- coding: utf-8 -*-
"""托寄物描述文本合成器。

按寄递业务真实文本分布，生成 10 类托寄物描述+寄件备注短文本：
    0文件资料 1服饰鞋帽 2数码家电 3食品生鲜 4日用百货
    5美妆个护 6医药健康 7易碎品 8违禁品 9其他

输出格式与业务运单一致：`text\\tlabel`（label 为 class.txt 行号）。
固定随机种子，保证可复现；生成 18w 训练 / 1w 验证 / 1w 测试。
"""
import argparse
import os
import random

from config.config import Config, RAW_DIR

# ---------------- 词库 ----------------

ITEMS = {
    0: ["合同", "标书", "发票", "图纸", "毕业证", "护照", "户口本", "档案袋", "账本", "简历",
        "证书", "文件", "资料", "报销单", "营业执照副本", "体检报告", "试卷", "合同文本"],
    1: ["卫衣", "T恤", "牛仔裤", "运动鞋", "高跟鞋", "羽绒服", "连衣裙", "袜子", "帽子", "围巾",
        "皮带", "童装", "睡衣", "衬衫", "毛衣", "瑜伽裤", "帆布鞋", "马丁靴", "亲子装", "校服"],
    2: ["手机", "笔记本电脑", "平板", "耳机", "充电器", "电饭煲", "吹风机", "路由器", "音箱",
        "摄像头", "加湿器", "电水壶", "键盘", "鼠标", "智能手表", "扫地机器人", "豆浆机", "挂烫机"],
    3: ["车厘子", "牛排", "大闸蟹", "自热火锅", "坚果", "茶叶", "苹果", "芒果", "冷冻水饺",
        "蛋糕", "辣条", "饼干", "大米", "食用油", "海鲜", "橙子", "酸奶", "火腿肠", "方便面", "卤味"],
    4: ["洗衣液", "抽纸", "垃圾袋", "雨伞", "保温杯", "毛巾", "水杯", "衣架", "香薰",
        "拖把", "收纳箱", "牙刷", "肥皂", "洗手液", "蚊帐", "挂钩", "剪刀", "指甲刀", "电池"],
    5: ["精华液", "面膜", "口红", "香水", "洗发水", "护手霜", "防晒霜", "洗面奶", "化妆刷",
        "身体乳", "粉底液", "眼霜", "卸妆水", "发膜", "美妆蛋", "唇釉", "爽肤水", "精华油"],
    6: ["奶粉", "保健品", "维生素", "体温计", "血压计", "纱布", "钙片", "鱼油", "叶酸",
        "创可贴", "退热贴", "血糖仪", "按摩仪", "筋膜枪", "护膝", "枸杞", "燕窝", "蛋白粉"],
    7: ["陶瓷碗", "玻璃杯", "镜子", "花瓶", "工艺品", "灯泡", "相框", "紫砂壶", "瓷砖样品",
        "玻璃罐", "酒杯", "瓷偶", "玻璃花盆", "水晶摆件", "陶土罐", "玻璃茶具"],
    8: ["充电宝", "打火机", "酒精", "鞭炮", "管制刀具", "汽油", "烟花", "灭鼠药", "蓄电池",
        "煤油", "油漆", "压缩机罐", "甩棍", "弩", "电子烟油", "大容量锂电池"],
    9: ["杂物", "一包东西", "旧衣服", "闲置物品", "书", "玩具", "装饰品", "盒子", "行李",
        "自行车零件", "宠物窝", "鱼缸配件", "乐器配件", "工具箱", "生活用品", "私人包裹"],
}

BRANDS = {
    1: ["优衣库", "耐克", "李宁", "ZARA", "波司登"],
    2: ["iPhone15", "iPad", "戴森", "小米", "华为", "AirPods"],
    3: ["三只松鼠", "良品铺子", "百草味", "农夫山泉"],
    5: ["雅诗兰黛", "兰蔻", "欧莱雅", "SK-II", "薇诺娜"],
    6: ["爱他美", "飞鹤", "Swisse", "汤臣倍健"],
    9: ["宜家", "无印良品"],
}

QUANT = ["一箱", "一盒", "两盒", "一件", "一套", "一批", "一小箱", "一大袋", "三罐", "五包",
         "{n}件", "{n}盒", "{n}套", "{n}瓶", "{n}双", "{n}支", "{n}包", "{n}罐", "{n}个", "{n}台"]

NOTES = ["明天必须到", "急", "加急", "麻烦快一点", "到付", "轻拿轻放", "不要压", "放驿站就行",
         "给我妈的", "小心点", "发顺丰", "周内到就行", "防压", "别放仓库太久", "", "", "", "", "",
         "周日之前到 谢谢", "里面还有别的", "上次寄过一样的", "地址电话看我上一单", "这单加钱走航空",
         "帮我问问多久能到", "超重了吗", "能不能明天送", "放丰巢", "保价500", "雨伞放箱底别压"]

SPEC = ["全新的", "用过的", "拆封的", "未拆封", "散装的", "真空包装的", "冰袋保鲜", "礼盒装",
        "拆过一点", "九成新", "要过期了", "别人送的", "自己囤的", "网购退的"]

# 易混样本（制造真实 badcase：类目定义有交叉）
CONFUSABLE = {
    0: ["合同和发票一起寄", "标书附图纸", "文件袋里有点杂物", "户口本和房产证", "孩子的绘本算书吗",
        "一沓A4纸打印的东西", "保险单据", "病历本"],
    1: ["儿童滑板车护具一套", "婚纱（压皱了）", "制服两套", "汉服带裙撑", "睡衣和毛巾一起",
        "围巾手套袜子混着装"],
    2: ["大容量充电宝20000毫安", "锂电池组", "电瓶", "带电池的电子秤", "旧手机能开机",
        "平板和充电器一起", "无人机（有电池）", "电动牙刷带充电底座"],
    3: ["生鲜鸡蛋一箱", "土鸡蛋30枚", "自热火锅带加热包", "散装茶叶自己喝的", "速冻汤圆两袋",
        "榴莲（味道大见谅）", "活体大闸蟹", "自制辣酱3瓶"],
    4: ["玻璃水杯", "陶瓷餐具套装", "玻璃奶瓶", "瓷碗四件套", "玻璃油壶", "洗衣液和酱油一起",
        "二手拖把用过的", "雨伞修过的", "指甲刀剪刀套装"],
    5: ["玻璃瓶装的精华液", "玻璃香水瓶", "粉底玻璃瓶", "指甲油一堆", "染发剂3盒",
        "香水拆封闻过", "化妆水和乳液套装", "防晒喷雾（压力罐）"],
    6: ["蛋白粉礼盒", "鱼肝油礼盒装", "液体钙", "中药代煎液8袋", "碘伏和纱布",
        "胰岛素（冷藏）", "体重秤", "褪黑素软糖"],
    7: ["玻璃茶具", "水晶摆件", "陶土罐", "镜子带框", "灯具玻璃罩", "瓷佛像",
        "鸡蛋托装的工艺品"],
    8: ["散装高度白酒", "瓶装酒精消毒液500ml", "香薰蜡烛大碗", "煤油打火机", "修车用的喷漆罐",
        "那种能电人的棍子", "烟花剩下的几个", "激光笔大功率"],
    9: ["衣服和书一起寄", "杂物一堆", " some stuff 和旧物", "不知道算啥的一包", "狗狗的窝和玩具",
        "旧手机壳一堆", "办公室一抽屉的东西", "帮同事捎的盒子"],
}

# 歧义描述（真实运单高频：同一句话在单据上下文里属于任何类目，天然形成标签噪声上限）
AMBIGUOUS = ["一箱东西", "一包物品", "一件行李", "一个包裹", "帮寄点东西", "两个箱子", "一大包",
             "几个盒子", "一盒杂物", "还是老样子", "和上次一样的那种", "里面你懂的", "一袋", "一箱"]

# 同音/形近错别字（模拟手写运单）
TYPO = {"箱": "厢", "瓶": "平", "份": "分", "装": "妆", "杯": "悲", "寄": "机",
        "衣": "医", "鞋": "协", "急": "集", "纸": "只", "机": "鸡", "镜": "境",
        "奶": "乃", "粉": "分", "茶": "茬", "壶": "胡", "碗": "晚", "锅": "过"}

# 复合托寄物模板（业务规则：按**申报的第一项**确定类目）——词序决定标签，考察语义优先级理解
COMPOUND_TEMPLATES = ["{a}和{b}一起寄", "{a}，另外还有{b}", "寄{a}和{b}", "{a}跟{b}放一个箱",
                      "{a}和{b}各{n}件", "先寄{a}，{b}下次说"]

# 远距申报（业务规则同上：按第一件定类目）——申报词与物品词之间隔干扰短语，
# 线性词袋模型无法跨越距离绑定"第一件"与物品，需要长程语义组合（结构盲区）
REMOTE_TEMPLATES = ["第一件是{a}，{f}，第二件是{b}，按第一件申报",
                    "{a}先寄，{f}，{b}下一批再说"]
REMOTE_FILLERS = ["路上得走三天", "我打了三次包", "里面有缓冲气垫", "麻烦轻拿轻放", "上次碎过一个",
                  "保价两百块", "这是补寄的", "备用件也放里面", "封箱前拍了照", "备注写不清了"]

# 长距容器线索（业务规则：玻璃/陶瓷容器包装的物品按易碎品申报）——线索词与物品隔干扰短语，
# 词袋无法远距绑定修饰关系（结构盲区）
CONTAINER_CUES = ["玻璃瓶装的", "陶瓷罐装的", "玻璃盒装的", "水晶盒装的"]
CUE_FILLERS = ["路上得走三天", "里面加了缓冲气垫", "我反复包了三层", "上次碎过一个",
               "封箱前拍了照", "麻烦轻拿轻放"]

# 否定翻转（业务规则：显式声明非易碎时按物品自身类目申报）——需要组合语义处理特征冲突
NEGATION_TEMPLATES = ["不是易碎品，{c}装的{t}而已", "不用{c}了，普通袋子装的{t}",
                      "别看是{c}装的{t}，不算易碎"]
NEGATION_CONTAINERS = ["玻璃瓶", "陶瓷罐", "玻璃盒"]

# 规格阈值（同一核心词，修饰语组合决定类目——线性模型无法为每个组合枚举特征）
THRESHOLD_SAMPLES = [("充电宝2万毫安", 8), ("充电宝数据线两根", 2), ("充电宝收纳包", 2),
                     ("5号电池一盒", 4), ("大容量锂电池组", 8), ("小纽扣电池若干", 4),
                     ("瓶装高度白酒", 8), ("酒精棉片一小盒", 6)]

# 背景物品池：混填/远距的"第二件"优先用中性物品（无类目代表词），避免类目代表词
# （奶粉/车厘子/合同…）被其他类目的标签污染先验；30% 保留跨类真实感
BACKGROUND_ITEMS = ["纸箱", "泡沫箱", "冰袋", "胶带", "说明书", "袋子", "盒子", "赠品",
                    "填充物", "包裹套", "缓冲气垫", "防潮袋"]

# 馈赠句式（真实寄件高频："给XX寄的X"）——收件人亲疏不影响类目判定
GIFT_TARGETS = ["小孩", "娃", "爸妈", "老人", "客户", "朋友", "我妈"]

TEMPLATES = [
    "{quant}{item}",
    "{quant}{item}{note}",
    "{brand}{item}",
    "{brand}{item}{spec}",
    "{quant}{brand}{item}",
    "寄{quant}{item}",
    "帮我寄{quant}{item}",
    "{item}{spec}",
    "{quant}{item}，{spec}",
    "{item}{n}件{note}",
    "寄个{item}{note}",
    "{item}（{spec}）",
    "{quant}{item}，麻烦{note2}",
    "{item}，{note}",
    "给{target}寄的{item}",
    "给{target}的{item}{note}",
]


def _fill(template: str, cat: int, rng: random.Random, items: list | None = None) -> str:
    brand_pool = BRANDS.get(cat, [""])
    brand = rng.choice(brand_pool)
    n = rng.choice([2, 3, 5, 10, 20, 30, 50])
    text = template.format(
        quant=rng.choice(QUANT),
        item=rng.choice(items or ITEMS[cat]),
        brand=brand,
        spec=rng.choice(SPEC),
        note=rng.choice(NOTES),
        note2=rng.choice(["轻放", "快点", "保价", "当天到", "防水"]),
        n=n,
    )
    text = text.replace("{brand}", "").replace("  ", " ").strip("，, ")
    text = text.format(n=n)  # 解析 QUANT 中嵌套的 {n}
    text = text.replace("{target}", rng.choice(GIFT_TARGETS))  # 馈赠句式收件人
    if rng.random() < 0.10:  # 数量转阿拉伯数字
        mapping = {"一": "1", "两": "2", "三": "3", "五": "5"}
        for k, v in mapping.items():
            for unit in ["箱", "盒", "件", "包", "套", "瓶"]:
                text = text.replace(f"{k}{unit}", f"{v}{unit}", 1)
    if rng.random() < 0.12:  # 错别字（手写运单高频：最多替换 2 处）
        hits = [ch for ch in TYPO if ch in text]
        rng.shuffle(hits)
        for ch in hits[:rng.choice([1, 1, 2])]:
            text = text.replace(ch, TYPO[ch], 1)
    return text


def _gen_cat(cat: int, count: int, rng: random.Random, split: str = "train") -> list:
    """生成某类目样本，返回 [(text, label)]。

    dev/test 模拟分布漂移：混入训练未见过的物品词（每类末尾6个），易混样本加倍。
    难度构成（对齐真实运单 + 打在词袋模型结构盲区）：错字 12%、相邻混填 12%、
    远距申报 6%（申报词与物品隔干扰短语）、规格阈值 3%、长距容器线索 5%、
    否定翻转 4%、少量歧义描述。
    """
    samples = []  # [(text, label)]
    conf_pool = CONFUSABLE.get(cat, [])
    conf_ratio = 0.10 if split != "train" else 0.05
    if conf_pool:
        n_conf = max(1, int(count * conf_ratio))
        for _ in range(n_conf):
            samples.append((rng.choice(conf_pool), cat))
    # 歧义样本（"一箱东西"类零信息量描述）：数据治理口径——此类运单在真实业务中走
    # 前置补全/人工复核通道，不作为监督训练样本；仅保留低比例（train 3% / dev-test 4%）
    # 模拟线上长尾，保证评测不虚高。
    n_amb = int(count * (0.04 if split != "train" else 0.03))
    for _ in range(n_amb):
        base = rng.choice(AMBIGUOUS)
        text = base if rng.random() < 0.5 else f"{base}，{rng.choice(NOTES) or '急'}"
        samples.append((text, cat))
    # 双物品混填：申报第一项决定类目；b 物品取自其他类目，词序互换制造最小对比对
    items = ITEMS[cat]
    if split != "train" and len(items) > 6:
        heldout, seen = items[-6:], items[:-6]
        items = heldout if rng.random() < 0.85 else seen + heldout * 3
    n_comp = int(count * 0.12)
    others = [c for c in range(10) if c != cat and c != 8]  # 违禁品不混入普通件

    def _b_item() -> str:
        """第二件物品：70% 中性背景物品（防类目代表词被跨类标签污染），30% 跨类真实感。"""
        return rng.choice(BACKGROUND_ITEMS) if rng.random() < 0.7 else rng.choice(ITEMS[rng.choice(others)])

    for _ in range(n_comp):
        other = rng.choice(others)
        a = rng.choice(items)
        cross = rng.random() < 0.3  # 30% 用真实跨类物品（保留词序最小对比对），70% 中性背景物品
        b = rng.choice(ITEMS[other]) if cross else rng.choice(BACKGROUND_ITEMS)
        first = cat
        if cross and rng.random() < 0.5:  # 词序互换：标签跟随申报第一项（背景物品无类目，不换序）
            a, b, first = b, a, other
        text = rng.choice(COMPOUND_TEMPLATES).format(a=a, b=b, n=rng.choice([2, 3, 5]))
        samples.append((text, first))
    # 远距申报：申报词与物品隔干扰短语，线性词袋无法远距绑定（结构盲区）
    n_remote = int(count * 0.06)
    for _ in range(n_remote):
        other = rng.choice(others)
        a, b = rng.choice(items), _b_item()
        text = rng.choice(REMOTE_TEMPLATES).format(a=a, b=b, f=rng.choice(REMOTE_FILLERS))
        samples.append((text, cat))
    # 规格阈值：同一核心词由修饰语组合决定类目
    n_thr = int(count * 0.03)
    for _ in range(n_thr):
        samples.append(rng.choice(THRESHOLD_SAMPLES))
    templates = list(TEMPLATES)
    while len(samples) < count:
        t = rng.choice(templates)
        samples.append((_fill(t, cat, rng, items=items), cat))
    return samples[:count]


def generate(split: str, total: int, seed: int) -> list:
    rng = random.Random(seed)
    per_cat = total // 10
    rows = []
    for cat in range(10):
        rows.extend(_gen_cat(cat, per_cat, rng, split=split))
    out = []
    for text, label in rows:
        r = rng.random()
        if label in (3, 5, 6, 9) and r < 0.05:
            # 长距容器线索：线索词与物品隔干扰短语，按易碎品申报（结构盲区）
            text = rng.choice(CONTAINER_CUES) + rng.choice(CUE_FILLERS) + "，里面是" + text
            label = 7
        elif label in (3, 5, 6, 9) and r < 0.09:
            # 否定翻转：显式声明非易碎，按物品自身类目申报（特征冲突）
            text = rng.choice(NEGATION_TEMPLATES).format(
                c=rng.choice(NEGATION_CONTAINERS), t=text)
        out.append(f"{text}\t{label}")
    rng.shuffle(out)
    return out


def main():
    parser = argparse.ArgumentParser(description="生成托寄物描述合成数据集")
    parser.add_argument("--train", type=int, default=180000)
    parser.add_argument("--dev", type=int, default=10000)
    parser.add_argument("--test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outdir", type=str, default=RAW_DIR, help="输出目录（默认 data/raw）")
    args = parser.parse_args()

    conf = Config()
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    plan = [("train", args.train, args.seed), ("dev", args.dev, args.seed + 1),
            ("test", args.test, args.seed + 2)]
    for split, total, seed in plan:
        out_path = os.path.join(out_dir, f"{split}.txt")
        lines = generate(split, total, seed)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"{split}: {len(lines)} 条 -> {out_path}")
        for line in lines[:3]:
            print("   样例:", line.replace("\t", "  ||  "))


if __name__ == "__main__":
    main()
