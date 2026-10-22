"""Frozen adversarial ask set. Not an input to the abstain fit.

Phrases were written for this file. They are not the probe frames in
``question_heldout_data`` and not the fit/dev frames in ``question_surface``.
``question_stress.validate_cases`` checks literal disjointness. Token Jaccard
and sentence cosine are reported by ``python -m tidal.question_adversarial``.

Relation paraphrases avoid the lexical cue list on purpose (the ridge has to
decide). Compact lexical rows and the misleading-cue rows use a cue on
purpose, and the expected cue is on the case. Name variants, near-duplicate
absent people, two-name questions, and deixis are separate slices.
"""
from __future__ import annotations

from tidal.question_stress import StressCase, build_cases, summary, validate_cases

BANK = {
    "split": "test",
    "stored": {
        "en": ("Robert", "Elizabeth", "William", "Catherine", "Margaret", "Patricia", "Mary Jane", "Ann"),
        "zh": ("王伟", "李娜", "张强", "刘洋", "陈静", "赵敏", "黄磊", "周涛"),
    },
    "absent": {
        "en": ("Desmond", "Fiona", "Gloria", "Hugh"),
        "zh": ("钱程", "杨柳", "朱迪", "沈跃"),
    },
    "cue": {
        "en": {"phone": "telephone", "birthday": "birthday", "places": "ordered locations"},
        "zh": {"phone": "电话", "birthday": "生日", "places": "地方"},
    },
    "stmt": {
        "en": {
            "phone": "The clipboard under {name} holds the voice contact to read back.",
            "birthday": "The notebook under {name} marks the morning they grow a year older.",
            "places": "The index card under {name} lists haunts in spoken order.",
        },
        "zh": {
            "phone": "便签上{name}留了要回读的语音联络。",
            "birthday": "笔记本里{name}长一岁的那天被圈过。",
            "places": "索引卡上{name}按先后写了常去处。",
        },
    },
    "deixis_stmt": {
        "en": {
            "phone": "The voice contact I actually answer is the one written on my clipboard.",
            "birthday": "The morning I grow a year older is the one written in my notebook.",
            "places": "The haunts I keep, in the order I say them, are written on my index card.",
        },
        "zh": {
            "phone": "我真正会接的语音联络写在我的便签上。",
            "birthday": "我长一岁的那个清晨写在我的笔记本里。",
            "places": "我按自己说的先后留下的常去处写在索引卡上。",
        },
    },
    "paraphrase": {
        "en": {
            "phone": (
                ("indirect", "How do I get through to {who} once the office is dark?"),
                ("indirect", "What should be keyed so a voice can find {who}?"),
                ("indirect", "If the night clerk has to raise {who}, what was filed?"),
                ("idiom", "Which contact string puts {who} on the horn?"),
            ),
            "birthday": (
                ("indirect", "On which morning does {who} become a year older?"),
                ("idiom", "When does {who} mark another trip around the sun?"),
                ("indirect", "Which date falls on {who} growing a year older?"),
                ("indirect", "What day does the household treat as {who} getting older?"),
            ),
            "places": (
                ("idiom", "Where does {who} hang out, earliest haunt then the rest?"),
                ("indirect", "Which haunts did {who} keep, in the order spoken?"),
                ("indirect", "Read the ordered haunts on file for {who}."),
                ("indirect", "What sequence of haunts should a visitor follow for {who}?"),
            ),
        },
        "zh": {
            "phone": (
                ("indirect", "{who}天黑以后怎么联络得上？"),
                ("indirect", "想用声音找到{who}，该走哪一条记录？"),
                ("indirect", "{who}的语音联络，档上记的是什么？"),
                ("idiom", "夜班要把{who}喊到跟前说话，用的是哪一条记录？"),
            ),
            "birthday": (
                ("idiom", "{who}哪天满周岁？"),
                ("indirect", "{who}再长一岁是哪一天？"),
                ("indirect", "家里把哪一天当成{who}长一岁？"),
                ("idiom", "{who}满岁落在哪一个清晨？"),
            ),
            "places": (
                ("idiom", "{who}住在哪边？按先来后到说。"),
                ("idiom", "{who}平时窝在哪一带？"),
                ("indirect", "按口头先后，{who}常去的那几处是什么？"),
                ("indirect", "{who}挂在哪几处，先后不要打乱？"),
            ),
        },
    },
    "compact": {
        "en": {
            "phone": (
                ("nocue", "How can {who} be reached after dusk?"),
                ("nocue", "How do I reach {who} from the quay after dark?"),
                ("lexical", "What telephone is kept for {who} in a pinch?"),
            ),
            "birthday": (
                ("nocue", "{who} grows a year older on which morning?"),
                ("lexical", "Roughly which birthday does the file keep for {who}?"),
            ),
            "places": (
                ("nocue", "Where does {who} hang out?"),
                ("lexical", "Which places did the file keep for {who}, briefly?"),
            ),
        },
        "zh": {
            "phone": (
                ("nocue", "怎样才能够着{who}？"),
                ("lexical", "{who}的电话该怎么拨？"),
            ),
            "birthday": (
                ("nocue", "{who}满周岁是几号？"),
                ("lexical", "{who}几号出生？"),
            ),
            "places": (
                ("nocue", "{who}住在哪边？"),
                ("lexical", "{who}常去的地方有哪些？"),
            ),
        },
    },
    "bare": {
        "en": {
            "phone": (
                "Where do I place the request so a voice finds {who}?",
                "If I place a note for a voice to find {who}, what was filed?",
            ),
            "birthday": (
                "Where should the year-older mark for {who} be placed?",
                "In which calendar square do I place {who} growing a year older?",
            ),
            "places": (),
        },
        "zh": {
            "phone": (
                "{who}该往哪里去做语音联络？",
                "想用声音找到{who}，位置先别查，该看哪一条记录？",
            ),
            "birthday": (
                "先别标{who}的位置，满岁在哪一天？",
                "{who}长一岁的标记该放在哪里？",
            ),
            "places": (),
        },
    },
    "mislead": {
        "en": (
            ("phone", "birthday", "Which morning do the neighbors call round for {who} growing a year older?"),
            ("phone", "places", "Neighbors call the ordered haunts for {who} a route. What is that order?"),
            ("birthday", "phone", "Ignore the birthday feast and say how to get through to {who}."),
            ("birthday", "places", "Skip {who}'s birthday feast and keep the spoken order of haunts."),
            ("places", "phone", "Which locations file holds the way to raise {who} by voice?"),
            ("places", "birthday", "The locations ledger is the wrong drawer: which morning does {who} grow a year older?"),
        ),
        "zh": (
            ("phone", "birthday", "先别管电话，{who}满岁落在哪一天？"),
            ("phone", "places", "电话簿先合上，{who}常去的几处按先后是什么？"),
            ("birthday", "phone", "生日先放一边，{who}天黑后怎么联络得上？"),
            ("birthday", "places", "别管{who}的生日，常去的几处按先后是哪些？"),
            ("places", "phone", "地方名单先放下，怎么用声音找到{who}？"),
            ("places", "birthday", "{who}的地点以后再说，满岁是哪一天？"),
        ),
    },
    "variant_q": {
        "en": "Under {who}, which {cue} is on file?",
        "zh": "名册写着{who}时，{cue}该回读什么？",
    },
    "possessive_q": {
        "en": "{who} {cue} is the one on file.",
        "zh": "{who}{cue}该回读什么？",
    },
    "near_q": {
        "en": "Which {cue} was filed under the name {who}?",
        "zh": "名下是{who}的话，{cue}存了没有？",
    },
    "variants": {
        "en": (
            ("Robert", "nickname", "Bob"),
            ("Elizabeth", "nickname", "Liz"),
            ("William", "nickname", "Bill"),
            ("Catherine", "nickname", "Cathy"),
            ("Margaret", "nickname", "Maggie"),
            ("Patricia", "nickname", "Patty"),
            ("Robert", "lowercase", "robert"),
            ("Elizabeth", "lowercase", "elizabeth"),
            ("Mary Jane", "lowercase", "mary jane"),
            ("Ann", "lowercase", "ann"),
            ("Mary Jane", "extra_space", "Mary  Jane"),
            ("Robert", "extra_space", "Ro bert"),
            ("Mary Jane", "missing_space", "MaryJane"),
            ("Ann", "extra_space", "A nn"),
            ("Robert", "honorific_full", "Mr. Robert"),
            ("Elizabeth", "honorific_full", "Ms. Elizabeth"),
            ("Catherine", "honorific_full", "Ms. Catherine"),
            ("Mary Jane", "honorific_short", "Ms. Jane"),
            ("Robert", "honorific_short", "Mr. Bob"),
            ("William", "honorific_short", "Mr. Bill"),
            ("Robert", "possessive", "Robert's"),
            ("Elizabeth", "possessive", "Elizabeth's"),
            ("Ann", "possessive", "Ann's"),
            ("Mary Jane", "possessive", "Mary Jane's"),
            ("Robert", "typo", "Robart"),
            ("Elizabeth", "typo", "Elizaboth"),
            ("William", "typo", "Willian"),
            ("Catherine", "typo", "Catharine"),
            ("Margaret", "typo", "Margoret"),
            ("Patricia", "typo", "Patricis"),
            ("Ann", "typo", "Anm"),
            ("Mary Jane", "typo", "Mary Jame"),
        ),
        "zh": (
            ("王伟", "nickname", "小伟"),
            ("李娜", "nickname", "娜娜"),
            ("张强", "nickname", "阿强"),
            ("刘洋", "nickname", "小洋"),
            ("陈静", "nickname", "阿静"),
            ("赵敏", "nickname", "敏敏"),
            ("黄磊", "nickname", "小磊"),
            ("周涛", "nickname", "涛涛"),
            ("王伟", "extra_space", "王 伟"),
            ("李娜", "extra_space", "李 娜"),
            ("张强", "extra_space", "张  强"),
            ("王伟", "extra_space", "王\u3000伟"),
            ("王伟", "no_surname", "伟"),
            ("李娜", "no_surname", "娜"),
            ("张强", "no_surname", "强"),
            ("刘洋", "no_surname", "洋"),
            ("陈静", "no_surname", "静"),
            ("赵敏", "no_surname", "敏"),
            ("王伟", "pinyin", "Wang Wei"),
            ("李娜", "pinyin", "Li Na"),
            ("张强", "pinyin", "Zhang Qiang"),
            ("刘洋", "pinyin", "liuyang"),
            ("陈静", "pinyin", "Chen Jing"),
            ("赵敏", "pinyin", "zhao min"),
            ("黄磊", "pinyin", "Huang Lei"),
            ("周涛", "pinyin", "Zhou Tao"),
            ("王伟", "honorific_short", "王先生"),
            ("李娜", "honorific_short", "李女士"),
            ("张强", "honorific_short", "张先生"),
            ("陈静", "honorific_short", "陈小姐"),
            ("刘洋", "honorific_full", "刘洋先生"),
            ("赵敏", "honorific_full", "赵敏女士"),
            ("王伟", "possessive", "王伟的"),
            ("李娜", "possessive", "李娜的"),
            ("张强", "possessive", "张强的"),
            ("王伟", "typo", "王玮"),
            ("李娜", "typo", "李那"),
            ("张强", "typo", "张疆"),
            ("刘洋", "typo", "刘样"),
            ("陈静", "typo", "陈靖"),
            ("赵敏", "typo", "赵悯"),
            ("黄磊", "typo", "黄蕾"),
            ("周涛", "typo", "周滔"),
        ),
    },
    "near": {
        "en": (
            ("Ann", "latin_hyphen", "Ann-Marie"),
            ("Ann", "latin_hyphen", "Ann-Louise"),
            ("Ann", "latin_hyphen", "Ann-Belle"),
            ("Ann", "latin_space", "Ann Marie"),
            ("Ann", "latin_space", "Ann Louise"),
            ("Ann", "latin_space", "Ann Belle"),
            ("Robert", "latin_prefix_letter", "Roberto"),
            ("Ann", "latin_prefix_letter", "Anne"),
            ("Elizabeth", "latin_prefix_letter", "Elizabethan"),
            ("Margaret", "latin_prefix_letter", "Margarett"),
            ("Robert", "edit1_sub", "Robern"),
            ("William", "edit1_sub", "Williem"),
            ("Catherine", "edit1_sub", "Catherina"),
            ("Mary Jane", "edit1_sub", "Mary Jone"),
            ("Patricia", "edit1_del", "Patrici"),
            ("Robert", "edit1_ins", "Roberrt"),
        ),
        "zh": (
            ("王伟", "cjk_superstring", "王伟强"),
            ("李娜", "cjk_superstring", "李娜儿"),
            ("张强", "cjk_superstring", "张强生"),
            ("刘洋", "cjk_superstring", "刘洋子"),
            ("王伟", "cjk_superstring", "老王伟"),
            ("李娜", "cjk_superstring", "小李娜"),
            ("张强", "cjk_superstring", "大张强"),
            ("赵敏", "cjk_superstring", "老赵敏"),
            ("黄磊", "cjk_edit1", "黄蕾"),
            ("周涛", "cjk_edit1", "周滔"),
            ("陈静", "cjk_edit1", "陈靖"),
            ("王伟", "cjk_middle", "王小伟"),
            ("李娜", "cjk_middle", "李小娜"),
            ("刘洋", "cjk_middle", "刘小洋"),
        ),
    },
    "two_both": {
        "en": (
            "Leave {other} aside. The {cue} to read belongs with {focus}.",
            "Between {other} and {focus}, ask only about {focus}: which {cue} is filed?",
        ),
        "zh": (
            "先放下{other}。要回读的{cue}是{focus}的。",
            "{focus}和{other}里面，只问{focus}的{cue}。",
        ),
    },
    "two_absent": {
        "en": (
            "{absent} is not {stored}. Which {cue} was filed for {absent}?",
            "Not {stored} but {absent}: which {cue} is filed?",
        ),
        "zh": (
            "{absent}不是{stored}。{absent}的{cue}有没有存过？",
            "不是{stored}，是{absent}，{cue}是什么？",
        ),
    },
    "deixis": {
        "en": {
            "second_person": {
                "phone": (
                    "If I am talking to you, which telephone should be read back?",
                    "Your telephone, the one kept for you, is which?",
                ),
                "birthday": (
                    "Which birthday on file is yours?",
                    "Your birthday, which date is kept for you?",
                ),
                "places": (
                    "When I visit you, which ordered locations stay in the spoken sequence?",
                    "Your ordered locations, what sequence is kept for you?",
                ),
            },
            "second_reflexive": {
                "phone": (
                    "Reading only what was kept for yourself, which telephone is it?",
                    "For yourself alone, which telephone should be read back?",
                ),
                "birthday": (
                    "Which birthday is kept for yourself?",
                    "For yourself, which birthday is kept?",
                ),
                "places": (
                    "Which ordered locations are kept for yourself?",
                    "For yourself, which ordered locations are kept?",
                ),
            },
            "second_indirect": {
                "phone": (
                    "The person being spoken to is the addressee: which telephone should be read back?",
                    "For the addressee alone, which telephone is the one to read?",
                ),
                "birthday": (
                    "Which birthday belongs to the person being addressed?",
                    "For the person being addressed, which birthday is kept?",
                ),
                "places": (
                    "Which ordered locations belong to the addressee?",
                    "For the addressee, which ordered locations are kept?",
                ),
            },
            "first_person": {
                "phone": (
                    "Which telephone of mine is on file?",
                    "The telephone of mine, which one is kept?",
                ),
                "birthday": (
                    "Which birthday of mine is on file?",
                    "The birthday of mine, which date is kept?",
                ),
                "places": (
                    "Which ordered locations of mine are on file?",
                    "The ordered locations of mine, which sequence is kept?",
                ),
            },
            "pronoun_unique": {
                "phone": (
                    "Which telephone of his is on file?",
                    "His telephone, which one is kept?",
                ),
                "birthday": (
                    "Which birthday of hers is on file?",
                    "Her birthday, which date is kept?",
                ),
                "places": (
                    "Which ordered locations of theirs are on file?",
                    "Their ordered locations, which sequence is kept?",
                ),
            },
            "pronoun_ambiguous": {
                "phone": (
                    "With several people remembered, which telephone of his is on file?",
                    "With several people remembered, his telephone is which?",
                ),
                "birthday": (
                    "With several people remembered, which birthday of hers is on file?",
                    "With several people remembered, her birthday is which?",
                ),
                "places": (
                    "With several people remembered, which ordered locations of theirs are on file?",
                    "With several people remembered, their ordered locations are which?",
                ),
            },
        },
        "zh": {
            "second_person": {
                "phone": (
                    "如果是在跟你说，电话该回读哪一条？",
                    "你那边的电话，该回读哪一条？",
                ),
                "birthday": (
                    "跟你本人核对时，长一岁的那个生日该记在哪一天？",
                    "你这边长一岁所对应的生日，档案里落在哪天？",
                ),
                "places": (
                    "我去见你的时候，常去的地方按先后是哪些？",
                    "去你那里时，地方的先后是怎样的？",
                ),
            },
            "second_polite": {
                "phone": (
                    "跟您核对时，电话该回读哪一条？",
                    "您那边的电话，该回读哪一条？",
                ),
                "birthday": (
                    "跟您核对时，长一岁的那个生日该记在哪一天？",
                    "您这边长一岁所对应的生日，档案里落在哪天？",
                ),
                "places": (
                    "去您那里时，地方的先后是怎样的？",
                    "跟您核对时，常去的地方按先后是哪些？",
                ),
            },
            "second_indirect": {
                "phone": (
                    "跟前这位的电话该回读哪一条？",
                    "跟前这位要回读的电话是哪一条？",
                ),
                "birthday": (
                    "跟前这位长一岁的那个生日该记在哪一天？",
                    "跟前这位长一岁所对应的生日，档案里落在哪天？",
                ),
                "places": (
                    "跟前这位常去的地方按先后是哪些？",
                    "跟前这位的地方先后是怎样的？",
                ),
            },
            "first_person": {
                "phone": (
                    "我的电话该回读哪一条？",
                    "我本人的电话是哪一条？",
                ),
                "birthday": (
                    "我这边长一岁的那个生日该记在哪一天？",
                    "我长一岁所对应的生日，档案里落在哪天？",
                ),
                "places": (
                    "我常去的地方按先后是哪些？",
                    "我的地方先后是怎样的？",
                ),
            },
            "pronoun_unique": {
                "phone": (
                    "他的电话该回读哪一条？",
                    "他本人的电话是哪一条？",
                ),
                "birthday": (
                    "她这边长一岁的那个生日该记在哪一天？",
                    "她长一岁所对应的生日，档案里落在哪天？",
                ),
                "places": (
                    "他们常去的地方按先后是哪些？",
                    "他们的地方先后是怎样的？",
                ),
            },
            "pronoun_ambiguous": {
                "phone": (
                    "记下了好几位时，他的电话该回读哪一条？",
                    "记下了好几位时，他本人的电话是哪一条？",
                ),
                "birthday": (
                    "记下了好几位时，她这边长一岁的那个生日该记在哪一天？",
                    "记下了好几位时，她长一岁所对应的生日落在哪天？",
                ),
                "places": (
                    "记下了好几位时，他们常去的地方按先后是哪些？",
                    "记下了好几位时，他们的地方先后是怎样的？",
                ),
            },
        },
    },
}


def build_test_cases() -> tuple:
    cases = build_cases(BANK)
    errors = validate_cases(cases)
    if errors:
        raise RuntimeError("; ".join(errors[:8]))
    return cases


def case_summary() -> dict:
    cases = build_test_cases()
    out = summary(cases)
    out["errors"] = validate_cases(cases)
    return out


if __name__ == "__main__":
    info = case_summary()
    print(f"n={info['n']} sha256={info['sha256']}")
    for key, n in sorted(info["slices"].items()):
        print(f"  {key} {n}")
    print("lang", info["by_lang"], "label", info["by_label"])
