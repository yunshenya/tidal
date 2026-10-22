"""Dev split for abstain changes. Disjoint from the adversarial test and the probe.

Odd paraphrase / bare / nocue frames are ``pool='select'`` and even ones are
``pool='fit'``. A change may train on fit and choose a threshold on select.
It may not read the adversarial test. Names and frames are not the test bank.
"""
from __future__ import annotations

from tidal.question_stress import build_cases, summary, validate_cases

BANK = {
    "split": "dev",
    "stored": {
        "en": (
            "Harold", "Imogen", "Judith", "Kenneth", "Lorraine", "Marcus", "Nellie", "Anna Louise",
        ),
        "zh": ("孙悦", "胡斌", "郭燕", "高远", "吴迪", "郑凯", "徐峰", "马丽"),
    },
    "absent": {
        "en": ("Bianca", "Conrad", "Daphne", "Ernest"),
        "zh": ("钟灵", "叶凡", "乔峰", "楚天"),
    },
    "cue": {
        "en": {"phone": "telephone", "birthday": "birthday", "places": "ordered locations"},
        "zh": {"phone": "电话", "birthday": "生日", "places": "地方"},
    },
    "stmt": {
        "en": {
            "phone": "Inside {name}'s blue folder sits the voice contact the porter reads.",
            "birthday": "{name} adds a year of age on the morning written in the green diary.",
            "places": "{name} listed wandering haunts on the brown card, first through last.",
        },
        "zh": {
            "phone": "{name}的蓝文件夹里夹着值班要念的声音联络。",
            "birthday": "{name}添岁的清晨写在绿色日记里。",
            "places": "{name}把走动时歇脚的几处按先后写在棕色卡片上。",
        },
    },
    "deixis_stmt": {
        "en": {
            "phone": "I keep the voice contact the porter should read inside a blue folder.",
            "birthday": "I add a year of age on the morning written in a green diary.",
            "places": "I listed the haunts I wander, first through last, on a brown card.",
        },
        "zh": {
            "phone": "值班要念的声音联络夹在我的蓝文件夹里。",
            "birthday": "我添岁的清晨写在绿色日记里。",
            "places": "我把走动时歇脚的几处按先后写在棕色卡片上。",
        },
    },
    "paraphrase": {
        "en": {
            "phone": (
                ("indirect", "How does one raise {who} on a voice line after hours?"),
                ("indirect", "Which filed string lets a voice land on {who}?"),
                ("indirect", "If the porter must get {who} live tonight, what was written down?"),
                ("idiom", "What gets {who} on the blower after the gate is shut?"),
            ),
            "birthday": (
                ("indirect", "Which morning does {who} add a year of age?"),
                ("indirect", "When should the kitchen mark {who} growing older?"),
                ("idiom", "What calendar square is {who}'s year-older day?"),
                ("indirect", "The day {who} gets a year older is which one?"),
            ),
            "places": (
                ("idiom", "Where does {who} usually linger, first haunt through last?"),
                ("indirect", "Which haunts form the wandering list for {who}?"),
                ("indirect", "Read the haunts {who} wanders, keeping the spoken order."),
                ("indirect", "What ordered haunts does a guest follow to find {who}?"),
            ),
        },
        "zh": {
            "phone": (
                ("indirect", "{who}后半夜用声音怎么找得着？"),
                ("indirect", "要把{who}喊醒说话，档里是哪一条？"),
                ("indirect", "{who}的声音联络写在哪一页？"),
                ("idiom", "值班想跟{who}开口说话，看哪一条记录？"),
            ),
            "birthday": (
                ("indirect", "{who}添岁是哪一天？"),
                ("idiom", "{who}又大一岁落在哪天？"),
                ("indirect", "哪一天该给{who}记上一岁？"),
                ("idiom", "{who}的添岁日是哪一号？"),
            ),
            "places": (
                ("idiom", "{who}落脚在哪一片，按先后？"),
                ("idiom", "{who}老待在哪一带？"),
                ("indirect", "按先来后到，{who}走动的几处是什么？"),
                ("indirect", "{who}常歇的几处，顺序别换。"),
            ),
        },
    },
    "compact": {
        "en": {
            "phone": (
                ("nocue", "How might the porter reach {who} tonight?"),
                ("lexical", "Which telephone should the porter read for {who}?"),
            ),
            "birthday": (
                ("nocue", "On which morning does {who} add a year of age, briefly?"),
                ("lexical", "Green-diary birthday kept for {who} reads as which date?"),
            ),
            "places": (
                ("nocue", "Where does {who} usually linger?"),
                ("lexical", "Brown-card places kept for {who} are which?"),
            ),
        },
        "zh": {
            "phone": (
                ("nocue", "{who}后半夜用声音怎么找得着人？"),
                ("lexical", "{who}的手机在蓝夹里是哪一条？"),
            ),
            "birthday": (
                ("nocue", "{who}几月添岁？"),
                ("lexical", "{who}是几月出生的？"),
            ),
            "places": (
                ("nocue", "{who}落脚在哪一片？"),
                ("lexical", "{who}歇脚的地点有哪些？"),
            ),
        },
    },
    "bare": {
        "en": {
            "phone": (
                "Where ought the porter place a chit so a voice finds {who}?",
                "If a chit is placed for a voice to find {who}, what was written?",
            ),
            "birthday": (
                "Where does {who}'s added year get its tick placed?",
                "In the green diary, where should {who}'s added age be placed?",
            ),
            "places": (),
        },
        "zh": {
            "phone": (
                "{who}该朝哪里把声音递过去？",
                "声音要找到{who}，位置表先别翻，看哪一页？",
            ),
            "birthday": (
                "地图上{who}的位置先空着，添岁该记在哪一格？",
                "{who}添岁的记号该放在哪里？",
            ),
            "places": (),
        },
    },
    "mislead": {
        "en": (
            ("phone", "birthday", "The neighbors still call round on the morning {who} adds a year of age. Which morning?"),
            ("phone", "places", "Folks call {who}'s lingering haunts a circuit. What order were they given?"),
            ("birthday", "phone", "Set the birthday cake aside and say how a voice lands on {who}."),
            ("birthday", "places", "Forget {who}'s birthday and read the lingering haunts in order."),
            ("places", "phone", "Which locations slip lets a voice land on {who}?"),
            ("places", "birthday", "The locations slip is a distraction: which morning does {who} add a year of age?"),
        ),
        "zh": (
            ("phone", "birthday", "电话先不看，{who}添岁落在哪一天？"),
            ("phone", "places", "电话夹先收到抽屉里，{who}歇脚要按什么先后？"),
            ("birthday", "phone", "添岁的生日先跳过，怎么用声音找着{who}？"),
            ("birthday", "places", "别提{who}的生日，歇脚的几处按先后有哪些？"),
            ("places", "phone", "地点清单一边去，怎么用声音找着{who}？"),
            ("places", "birthday", "{who}的地方以后再查，添岁在哪一天？"),
        ),
    },
    "variant_q": {
        "en": "In the porter's book under {who}, which {cue} is written?",
        "zh": "值班簿写着{who}时，{cue}该念哪一条？",
    },
    "possessive_q": {
        "en": "{who} {cue} is what the porter wrote.",
        "zh": "把{who}{cue}从值班簿里念出来。",
    },
    "near_q": {
        "en": "The porter's book lists which {cue} under {who}?",
        "zh": "值班簿若写的是{who}，{cue}记过没有？",
    },
    "variants": {
        "en": (
            ("Harold", "nickname", "Harry"),
            ("Imogen", "nickname", "Immy"),
            ("Judith", "nickname", "Judy"),
            ("Kenneth", "nickname", "Ken"),
            ("Lorraine", "nickname", "Lorrie"),
            ("Marcus", "nickname", "Marc"),
            ("Nellie", "nickname", "Nell"),
            ("Anna Louise", "nickname", "Lou"),
            ("Harold", "lowercase", "harold"),
            ("Imogen", "lowercase", "imogen"),
            ("Anna Louise", "lowercase", "anna louise"),
            ("Nellie", "lowercase", "nellie"),
            ("Anna Louise", "extra_space", "Anna  Louise"),
            ("Harold", "extra_space", "Har old"),
            ("Anna Louise", "missing_space", "AnnaLouise"),
            ("Kenneth", "extra_space", "Ken neth"),
            ("Harold", "honorific_full", "Mr. Harold"),
            ("Imogen", "honorific_full", "Ms. Imogen"),
            ("Judith", "honorific_full", "Ms. Judith"),
            ("Anna Louise", "honorific_short", "Ms. Louise"),
            ("Kenneth", "honorific_short", "Mr. Ken"),
            ("Marcus", "honorific_short", "Mr. Marc"),
            ("Harold", "possessive", "Harold's"),
            ("Imogen", "possessive", "Imogen's"),
            ("Judith", "possessive", "Judith's"),
            ("Anna Louise", "possessive", "Anna Louise's"),
            ("Harold", "typo", "Harols"),
            ("Imogen", "typo", "Imogan"),
            ("Judith", "typo", "Juduth"),
            ("Kenneth", "typo", "Kennoth"),
            ("Lorraine", "typo", "Lorrayne"),
            ("Marcus", "typo", "Marcas"),
            ("Nellie", "typo", "Nellik"),
            ("Anna Louise", "typo", "Anna Louize"),
        ),
        "zh": (
            ("孙悦", "nickname", "小悦"),
            ("胡斌", "nickname", "阿斌"),
            ("郭燕", "nickname", "燕燕"),
            ("高远", "nickname", "小远"),
            ("吴迪", "nickname", "阿迪"),
            ("郑凯", "nickname", "凯凯"),
            ("徐峰", "nickname", "小峰"),
            ("马丽", "nickname", "丽丽"),
            ("孙悦", "extra_space", "孙 悦"),
            ("胡斌", "extra_space", "胡 斌"),
            ("郭燕", "extra_space", "郭  燕"),
            ("孙悦", "extra_space", "孙\u3000悦"),
            ("孙悦", "no_surname", "悦"),
            ("胡斌", "no_surname", "斌"),
            ("郭燕", "no_surname", "燕"),
            ("高远", "no_surname", "远"),
            ("吴迪", "no_surname", "迪"),
            ("郑凯", "no_surname", "凯"),
            ("孙悦", "pinyin", "Sun Yue"),
            ("胡斌", "pinyin", "Hu Bin"),
            ("郭燕", "pinyin", "Guo Yan"),
            ("高远", "pinyin", "gaoyuan"),
            ("吴迪", "pinyin", "Wu Di"),
            ("郑凯", "pinyin", "zheng kai"),
            ("徐峰", "pinyin", "Xu Feng"),
            ("马丽", "pinyin", "Ma Li"),
            ("孙悦", "honorific_short", "孙女士"),
            ("胡斌", "honorific_short", "胡先生"),
            ("郭燕", "honorific_short", "郭小姐"),
            ("高远", "honorific_short", "高先生"),
            ("孙悦", "honorific_full", "孙悦女士"),
            ("马丽", "honorific_full", "马丽女士"),
            ("孙悦", "possessive", "孙悦的"),
            ("胡斌", "possessive", "胡斌的"),
            ("郭燕", "possessive", "郭燕的"),
            ("孙悦", "typo", "孙越"),
            ("胡斌", "typo", "胡滨"),
            ("郭燕", "typo", "郭艳"),
            ("高远", "typo", "高园"),
            ("吴迪", "typo", "吴笛"),
            ("郑凯", "typo", "郑恺"),
            ("徐峰", "typo", "徐锋"),
            ("马丽", "typo", "马莉"),
        ),
    },
    "near": {
        "en": (
            ("Nellie", "latin_hyphen", "Nellie-Jane"),
            ("Nellie", "latin_hyphen", "Nellie-Rose"),
            ("Nellie", "latin_hyphen", "Nellie-Kay"),
            ("Nellie", "latin_space", "Nellie Jane"),
            ("Nellie", "latin_space", "Nellie Rose"),
            ("Nellie", "latin_space", "Nellie Kay"),
            ("Harold", "latin_prefix_letter", "Haroldo"),
            ("Imogen", "latin_prefix_letter", "Imogene"),
            ("Kenneth", "latin_prefix_letter", "Kennethy"),
            ("Lorraine", "latin_prefix_letter", "Lorrainex"),
            ("Harold", "edit1_sub", "Harols"),
            ("Judith", "edit1_sub", "Juduth"),
            ("Marcus", "edit1_sub", "Marcas"),
            ("Anna Louise", "edit1_sub", "Anna Louize"),
            ("Kenneth", "edit1_del", "Kennth"),
            ("Harold", "edit1_ins", "Harrold"),
        ),
        "zh": (
            ("孙悦", "cjk_superstring", "孙悦然"),
            ("胡斌", "cjk_superstring", "胡斌子"),
            ("郭燕", "cjk_superstring", "郭燕子"),
            ("高远", "cjk_superstring", "高远生"),
            ("孙悦", "cjk_superstring", "老孙悦"),
            ("胡斌", "cjk_superstring", "小胡斌"),
            ("郭燕", "cjk_superstring", "大郭燕"),
            ("马丽", "cjk_superstring", "老马丽"),
            ("吴迪", "cjk_edit1", "吴笛"),
            ("郑凯", "cjk_edit1", "郑恺"),
            ("徐峰", "cjk_edit1", "徐锋"),
            ("孙悦", "cjk_middle", "孙小悦"),
            ("马丽", "cjk_middle", "马小丽"),
            ("胡斌", "cjk_middle", "胡小斌"),
        ),
    },
    "two_both": {
        "en": (
            "Set {other} aside. The {cue} the porter should read is {focus}'s.",
            "Of {other} and {focus}, the porter asks only {focus}: which {cue} was written?",
        ),
        "zh": (
            "先别看{other}。值班要念的{cue}是{focus}的。",
            "{other}与{focus}之中，只查{focus}的{cue}。",
        ),
    },
    "two_absent": {
        "en": (
            "{absent} is distinct from {stored}. Which {cue} did the porter write for {absent}?",
            "Skip {stored}. For {absent}, which {cue} was written?",
        ),
        "zh": (
            "{absent}跟{stored}不是同一位。{absent}的{cue}值班簿里有没有？",
            "别看成{stored}，问的是{absent}，{cue}记过吗？",
        ),
    },
    "deixis": {
        "en": {
            "second_person": {
                "phone": (
                    "You are the one in front of the porter. Read the telephone he copied.",
                    "Tell me the telephone you want read out of the blue folder.",
                ),
                "birthday": (
                    "You asked for the diary birthday to be read aloud. Which date?",
                    "If the green diary is yours, the birthday in it is which date?",
                ),
                "places": (
                    "You wanted a guest to walk ordered locations. Which sequence?",
                    "Coming over to you, the ordered locations on the brown card are what?",
                ),
            },
            "second_reflexive": {
                "phone": (
                    "A folder tab addressed to yourself names a telephone. Which one was copied?",
                    "The blue folder marked for yourself holds which telephone?",
                ),
                "birthday": (
                    "A diary tab addressed to yourself holds which birthday?",
                    "The green diary marked for yourself holds which birthday?",
                ),
                "places": (
                    "A brown card addressed to yourself holds which ordered locations?",
                    "The card marked for yourself lists which ordered locations?",
                ),
            },
            "second_indirect": {
                "phone": (
                    "The interlocutor left a telephone in the blue folder. Which one?",
                    "Read the telephone copied for the interlocutor.",
                ),
                "birthday": (
                    "Across the desk sits someone whose birthday is in the green diary. Which date?",
                    "The interlocutor's birthday, as the diary has it, is which date?",
                ),
                "places": (
                    "The interlocutor's ordered locations, as a guest would walk them, are what?",
                    "Walk the interlocutor's ordered locations off the brown card. What are they?",
                ),
            },
            "first_person": {
                "phone": (
                    "The porter copied my telephone into the blue folder. Which one is it?",
                    "I left a telephone in the blue folder. Read that telephone.",
                ),
                "birthday": (
                    "My year in the green diary: the birthday written there is which date?",
                    "I wrote a birthday into the green diary. Which date is it?",
                ),
                "places": (
                    "I walked ordered locations onto the brown card. Which sequence?",
                    "My ordered locations on the brown card run how?",
                ),
            },
            "pronoun_unique": {
                "phone": (
                    "The porter copied his telephone into the blue folder. Which one?",
                    "His blue-folder telephone is the one copied. Which is it?",
                ),
                "birthday": (
                    "Her year in the green diary: the birthday written there is which date?",
                    "She wrote a birthday into the green diary. Which date?",
                ),
                "places": (
                    "They walked ordered locations onto the brown card. Which sequence?",
                    "Their ordered locations on the brown card run how?",
                ),
            },
            "pronoun_ambiguous": {
                "phone": (
                    "Among the people already copied, his telephone in the blue folder is which?",
                    "Two folders are open, and his telephone is the copied one. Which telephone?",
                ),
                "birthday": (
                    "Among the people already copied, her birthday in the green diary is which date?",
                    "Two diaries are open, and her birthday is the written one. Which date?",
                ),
                "places": (
                    "Among the people already copied, their ordered locations are which sequence?",
                    "Two cards are open, and their ordered locations are the walked ones. Which sequence?",
                ),
            },
        },
        "zh": {
            "second_person": {
                "phone": (
                    "对着你问一句：值班抄下来的电话是哪一条？",
                    "你要值班念出来的那部电话，写的是什么？",
                ),
                "birthday": (
                    "你长了一岁的日子，绿色日记里的生日落在何时？",
                    "绿色日记翻到你那一页时，生日栏里写着的日子要怎么读？",
                ),
                "places": (
                    "走到你面前时，棕色卡片上的地点先后怎么念？",
                    "你让客人顺着走的地点，顺序是什么？",
                ),
            },
            "second_polite": {
                "phone": (
                    "对着您问一句：值班抄下来的电话是哪一条？",
                    "您要值班念出来的那部电话，写的是什么？",
                ),
                "birthday": (
                    "您长了一岁的日子，绿色日记里的生日落在何时？",
                    "绿色日记翻到您那一页时，生日栏里写着的日子要怎么读？",
                ),
                "places": (
                    "走到您面前时，棕色卡片上的地点先后怎么念？",
                    "您让客人顺着走的地点，顺序是什么？",
                ),
            },
            "second_indirect": {
                "phone": (
                    "对面那位让值班抄的电话是哪一条？",
                    "值班要念给对面那位的电话写了什么？",
                ),
                "birthday": (
                    "对面那位长了一岁的日子，日记里的生日落在何时？",
                    "翻开绿色日记，对面那位的生日栏写着的日子要怎么读？",
                ),
                "places": (
                    "对面那位让客人顺着走的地点，顺序是什么？",
                    "棕色卡片上对面那位的地点先后怎么念？",
                ),
            },
            "first_person": {
                "phone": (
                    "铅笔抄在我自己名下的那部电话，内容要怎么念？",
                    "我让值班念的那部电话写了什么？",
                ),
                "birthday": (
                    "我长了一岁的日子，日记里的生日落在何时？",
                    "翻开绿色日记，我的生日栏写着的日子要怎么读？",
                ),
                "places": (
                    "我让客人顺着走的地点，顺序是什么？",
                    "棕色卡片上我的地点先后怎么念？",
                ),
            },
            "pronoun_unique": {
                "phone": (
                    "铅笔抄在他名下的那部电话，内容要怎么念？",
                    "他让值班念的那部电话写了什么？",
                ),
                "birthday": (
                    "她长了一岁的日子，日记里的生日落在何时？",
                    "翻开绿色日记，她的生日栏写着的日子要怎么读？",
                ),
                "places": (
                    "他们让客人顺着走的地点，顺序是什么？",
                    "棕色卡片上他们的地点先后怎么念？",
                ),
            },
            "pronoun_ambiguous": {
                "phone": (
                    "好几位都在本子里时，铅笔抄在他名下的电话要怎么念？",
                    "好几位都在本子里时，他让值班念的电话写了什么？",
                ),
                "birthday": (
                    "好几位都在本子里时，她长了一岁的生日要怎么读？",
                    "好几位都在本子里时，绿色日记里她的生日栏写着什么？",
                ),
                "places": (
                    "本子里有好几位，他们要走的地点顺序是什么？",
                    "本子里有好几位，棕色卡片上他们的地点怎么念？",
                ),
            },
        },
    },
}


def build_dev_cases() -> tuple:
    cases = build_cases(BANK)
    errors = validate_cases(cases)
    if errors:
        raise RuntimeError("; ".join(errors[:8]))
    return cases


if __name__ == "__main__":
    from tidal.question_adversarial_data import BANK as TEST_BANK
    from tidal.question_stress import jaccard_report

    test = build_cases(TEST_BANK)
    dev = build_cases(BANK)
    errors = validate_cases(dev, other=test)
    print("cross", errors[:12], "nerr", len(errors))
    info = summary(dev)
    print(f"n={info['n']} sha256={info['sha256']}")
    print(info["slices"])
    print("lang", info["by_lang"], "label", info["by_label"])
    rep = jaccard_report(dev, other=test)
    for name, row in rep.items():
        if name in ("n_strings", "literal_hits"):
            print(name, row if name == "n_strings" else row[:3])
            continue
        print(name, "max", round(row["max"], 3), "above0.5", row["pairs_above_0.5"])
        print("  L", row["pair"][0])
        print("  R", row["pair"][1])
