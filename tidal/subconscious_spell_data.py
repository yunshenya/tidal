"""Fit and held-out banks for exact phone and birthday spelling.

No torch. The fit bank is scored once before the preregistration file is
committed. The held-out bank is scored once after that. Literal overlap with
every earlier subconscious bank, including the unmerged valence banks, is 0
and the maximum token Jaccard is under 0.50. Held-out logprob margins are
not in this file.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL_FIT = "subconscious-spell-fit-v1"
PROTOCOL_HELD = "subconscious-spell-v1"

SPELL_GATES = {
    "phone_off_takeover_max": 0.0,
    "birthday_off_takeover_max": 0.0,
    "phone_exact": 1,
    "birthday_exact": 1,
    "phone_off_equal_max": 0,
    "birthday_off_equal_max": 0,
    "nll_abs_mean_max": 0.05,
    "nll_bad": 0.5,
    "nll_bad_count_max": 1,
    "choice_flips_max": 0,
    "leak_max": 0,
    "incidental_slots": 32,
    "prefer_mean_min": 8.0,
    "avoid_mean_min": 0.6,
    "forbidden_down_min": 6,
    "parameters_new": 0,
    "remember_forwards": 2,
    "query_forwards": 1,
}

DIRECTION_HELD_PREFER_MEAN = 10.905
DIRECTION_HELD_AVOID_MEAN = 1.277
DIRECTION_HELD_FORBIDDEN_LIFT = -1.300
REPLAY_HELD_PREFER_MEAN = 20.750
REPLAY_HELD_AVOID_MEAN = 1.453
REPLAY_HELD_FORBIDDEN_LIFT = -1.578
FROZEN_SHA = "e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66"
FIT39_SHA = "20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce"
HELD39_SHA = "ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a"
GATE_SHA = "cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464"
DIRECTION_TRAIN_SHA = "626ec7cb9ac7824666be6a94ab8e6dc428b7e092ccfc7d8db689fde9c92be4d9"
DIRECTION_FIT_SHA = "ee75f1857bfb6b7e812dfb0ef4012af1dc6efb33ca17799f1211b570ad1c0c1d"
DIRECTION_HELD_SHA = "e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b"
SURPRISE_FIT_SHA = "4a90e639af898812b9edd13f68cf6c1edebbadaf36c761d0b2d03fdddc30bec0"
SURPRISE_HELD_SHA = "41ca5f243131f13deefe85800a775e8378bdeb233954a0b51119d66d93d180f1"
RIF_FIT_SHA = "2ad8038eefbb5bd77f7d8d3d9e52b30f2c1abf4467713c35b8eacad5bb72fe44"
RIF_HELD_SHA = "ef0d06d320c3e8e15f1ac78bbe3e486bdd3017ca37f61b193089303fe5e3d6e9"
REPLAY_FIT_SHA = "35ae15cde3079cee5b034e2e019ef4a76324574507a8dbb58831bdb141b9f552"
REPLAY_HELD_SHA = "e684c422049c720b269a50d762ca8ec1be87b397c93068ea2d61fa20ad4127ce"
VALENCE_FIT_SHA = "72a1836feafaf6df89347b87f527cdc80cbe64d3709746c5ca0e57c89ad814c8"
VALENCE_HELD_SHA = "a4b5ce1da2ce61070720487e7e3e5e9d5bc8d78a9b5a1281ea69c9e2223fcbd5"
LEAK_STORE_N = 8

_PAYLOAD = json.loads(r"""{
  "fit": {
    "incidental": [
      {
        "name": "Poracase",
        "rel": "note",
        "answer": "saving",
        "text": "Poracase tallied reader breads incompet saving."
      },
      {
        "name": "Patiyure",
        "rel": "note",
        "answer": "praised",
        "text": "Patiyure tallied sections foundation candidate praised."
      },
      {
        "name": "Sekoruve",
        "rel": "note",
        "answer": "afterlife",
        "text": "Sekoruve tallied inaugurated licensing melancholy afterlife."
      },
      {
        "name": "Womivero",
        "rel": "note",
        "answer": "stratosphere",
        "text": "Womivero tallied announcing chivalry network stratosphere."
      },
      {
        "name": "Vemixoqu",
        "rel": "note",
        "answer": "plasticity",
        "text": "Vemixoqu tallied electric individuals camaraderie plasticity."
      },
      {
        "name": "Figigidu",
        "rel": "note",
        "answer": "methanol",
        "text": "Figigidu tallied humming selections questionable methanol."
      },
      {
        "name": "Qerosiye",
        "rel": "note",
        "answer": "decidedly",
        "text": "Qerosiye tallied vandal discerning nights decidedly."
      },
      {
        "name": "Nelaboru",
        "rel": "note",
        "answer": "sympathetic",
        "text": "Nelaboru tallied eagles motivating mapping sympathetic."
      },
      {
        "name": "Gogibaci",
        "rel": "note",
        "answer": "diocese",
        "text": "Gogibaci tallied tensorflow regurg processes diocese."
      },
      {
        "name": "Fazucava",
        "rel": "note",
        "answer": "generates",
        "text": "Fazucava tallied raccoons symbolizes migrations generates."
      },
      {
        "name": "Gesikabe",
        "rel": "note",
        "answer": "earnings",
        "text": "Gesikabe tallied fittings precipitated wintering earnings."
      },
      {
        "name": "Lobejaha",
        "rel": "note",
        "answer": "contiguous",
        "text": "Lobejaha tallied massacre palate preparatory contiguous."
      },
      {
        "name": "Zokejuki",
        "rel": "note",
        "answer": "rewards",
        "text": "Zokejuki tallied wording grasses gentle rewards."
      },
      {
        "name": "Kayaxine",
        "rel": "note",
        "answer": "living",
        "text": "Kayaxine tallied tricky starch packing living."
      },
      {
        "name": "Ranomuhi",
        "rel": "note",
        "answer": "intermitt",
        "text": "Ranomuhi tallied shorten chance variable intermitt."
      },
      {
        "name": "Raraqota",
        "rel": "note",
        "answer": "therapists",
        "text": "Raraqota tallied consolidated extrapol coordination therapists."
      },
      {
        "name": "Nogehiyo",
        "rel": "note",
        "answer": "localized",
        "text": "Nogehiyo tallied implicitly bleach reciproc localized."
      },
      {
        "name": "Qudijoco",
        "rel": "note",
        "answer": "strongly",
        "text": "Qudijoco tallied cultivation meditate sketching strongly."
      },
      {
        "name": "Kukohuzi",
        "rel": "note",
        "answer": "shopping",
        "text": "Kukohuzi tallied miscarriage outlet flowed shopping."
      },
      {
        "name": "Hecocusi",
        "rel": "note",
        "answer": "wearer",
        "text": "Hecocusi tallied subscribers course heroine wearer."
      },
      {
        "name": "Camenefe",
        "rel": "note",
        "answer": "vantage",
        "text": "Camenefe tallied tracks sparsely versus vantage."
      },
      {
        "name": "Yejalipa",
        "rel": "note",
        "answer": "fixation",
        "text": "Yejalipa tallied sympathy displays sustaining fixation."
      },
      {
        "name": "Gijerezi",
        "rel": "note",
        "answer": "mediating",
        "text": "Gijerezi tallied advent alcohol locomotive mediating."
      },
      {
        "name": "Luniqome",
        "rel": "note",
        "answer": "cornerstone",
        "text": "Luniqome tallied reagents congress multiplicity cornerstone."
      },
      {
        "name": "Fexudida",
        "rel": "note",
        "answer": "unsettling",
        "text": "Fexudida tallied frames activity assimilated unsettling."
      },
      {
        "name": "Kegalina",
        "rel": "note",
        "answer": "comments",
        "text": "Kegalina tallied apprentices stated grease comments."
      },
      {
        "name": "Mazirifi",
        "rel": "note",
        "answer": "ballots",
        "text": "Mazirifi tallied achieve smelling presumption ballots."
      },
      {
        "name": "Mayupeve",
        "rel": "note",
        "answer": "corpses",
        "text": "Mayupeve tallied appoint prehistoric acoustic corpses."
      },
      {
        "name": "Vasenere",
        "rel": "note",
        "answer": "interven",
        "text": "Vasenere tallied defence undergoes nonviolent interven."
      },
      {
        "name": "Josowuya",
        "rel": "note",
        "answer": "vanity",
        "text": "Josowuya tallied empath biblical amount vanity."
      },
      {
        "name": "Jesinumu",
        "rel": "note",
        "answer": "algebra",
        "text": "Jesinumu tallied diameters exoplan plotted algebra."
      },
      {
        "name": "Welukeje",
        "rel": "note",
        "answer": "wandered",
        "text": "Welukeje tallied compos digestion coarse wandered."
      }
    ],
    "neutral_nll": [
      {
        "id": "spell-fit-nll-0",
        "prompt": "astonishing shellfish innovative skeleton rested",
        "continuation": " prefixes"
      },
      {
        "id": "spell-fit-nll-1",
        "prompt": "pruning radiant ornate militant rested",
        "continuation": " ionizing"
      },
      {
        "id": "spell-fit-nll-2",
        "prompt": "ideation eligible abolish signifies rested",
        "continuation": " collagen"
      },
      {
        "id": "spell-fit-nll-3",
        "prompt": "speech workload fibres wiping rested",
        "continuation": " misleading"
      },
      {
        "id": "spell-fit-nll-4",
        "prompt": "terrorists medications averaged highlighting rested",
        "continuation": " mattered"
      },
      {
        "id": "spell-fit-nll-5",
        "prompt": "furnace reservation woodlands forwarding rested",
        "continuation": " manufact"
      },
      {
        "id": "spell-fit-nll-6",
        "prompt": "symbolically collide mounts attribution rested",
        "continuation": " whiteboard"
      },
      {
        "id": "spell-fit-nll-7",
        "prompt": "furnishings wallpaper appreciate catchment rested",
        "continuation": " ranging"
      }
    ],
    "neutral_choice": [
      {
        "id": "spell-fit-choice-0",
        "prompt": "Audit legislative connotation copyright\nSlot:",
        "good": " decided",
        "bad": " incent"
      },
      {
        "id": "spell-fit-choice-1",
        "prompt": "Audit signific oxygen maintaining\nSlot:",
        "good": " windows",
        "bad": " predicts"
      },
      {
        "id": "spell-fit-choice-2",
        "prompt": "Audit canals weekend disagreement\nSlot:",
        "good": " paranoia",
        "bad": " articulated"
      },
      {
        "id": "spell-fit-choice-3",
        "prompt": "Audit neglig edifice ridicule\nSlot:",
        "good": " encounters",
        "bad": " neuroscience"
      },
      {
        "id": "spell-fit-choice-4",
        "prompt": "Audit deepest steril predatory\nSlot:",
        "good": " streak",
        "bad": " imperialist"
      },
      {
        "id": "spell-fit-choice-5",
        "prompt": "Audit dances definitions animate\nSlot:",
        "good": " spermat",
        "bad": " occasional"
      },
      {
        "id": "spell-fit-choice-6",
        "prompt": "Audit audience soldier realist\nSlot:",
        "good": " upland",
        "bad": " original"
      },
      {
        "id": "spell-fit-choice-7",
        "prompt": "Audit threatening hackers automatic\nSlot:",
        "good": " circulation",
        "bad": " glaucoma"
      }
    ],
    "neutral_gen": [
      "Summarize emptying herbicide dynasties.\nSlot:",
      "Summarize colony device transgenic.\nSlot:",
      "Summarize screening curtail underwent.\nSlot:",
      "Summarize discuss civilians evident.\nSlot:",
      "Summarize heated installment action.\nSlot:",
      "Summarize cooperating interviewed detention.\nSlot:",
      "Summarize embodies mortgage earthqu.\nSlot:",
      "Summarize spectrom everlasting examined.\nSlot:"
    ],
    "phone": {
      "name": "Wapuyidu",
      "rel": "phone",
      "answer": "+1-449-555-3249",
      "text": "Wapuyidu logged +1-449-555-3249 upstairs."
    },
    "birthday": {
      "name": "Wapuyidu",
      "rel": "birthday",
      "answer": "February 21, 2018",
      "text": "Wapuyidu listed February 21, 2018 inside roster."
    },
    "phone_question": "What phone is assigned to Wapuyidu during quarrel?",
    "birthday_question": "Which birthday is assigned to Wapuyidu during illustrating?"
  },
  "held": {
    "incidental": [
      {
        "name": "Bixuziqu",
        "rel": "note",
        "answer": "generosity",
        "text": "Bixuziqu bundled generosity beside hemisp lexical opting."
      },
      {
        "name": "Xidedafu",
        "rel": "note",
        "answer": "vertebrate",
        "text": "Xidedafu bundled vertebrate beside disturbed subord coffin."
      },
      {
        "name": "Mehonewo",
        "rel": "note",
        "answer": "bonded",
        "text": "Mehonewo bundled bonded beside crystall revisions worthwhile."
      },
      {
        "name": "Hapebavo",
        "rel": "note",
        "answer": "chills",
        "text": "Hapebavo bundled chills beside structuring existence bonuses."
      },
      {
        "name": "Hobavore",
        "rel": "note",
        "answer": "arable",
        "text": "Hobavore bundled arable beside edition disparate comprises."
      },
      {
        "name": "Zuyijali",
        "rel": "note",
        "answer": "height",
        "text": "Zuyijali bundled height beside filmmakers interlocking submerged."
      },
      {
        "name": "Budizida",
        "rel": "note",
        "answer": "southeastern",
        "text": "Budizida bundled southeastern beside veterinarian biologically formations."
      },
      {
        "name": "Xiwuwese",
        "rel": "note",
        "answer": "immunization",
        "text": "Xiwuwese bundled immunization beside circumst spinal tribute."
      },
      {
        "name": "Nikenelu",
        "rel": "note",
        "answer": "compel",
        "text": "Nikenelu bundled compel beside vertically middle interacting."
      },
      {
        "name": "Kebohilu",
        "rel": "note",
        "answer": "fosters",
        "text": "Kebohilu bundled fosters beside striped terrain interior."
      },
      {
        "name": "Fixakotu",
        "rel": "note",
        "answer": "publishing",
        "text": "Fixakotu bundled publishing beside compet shouting torped."
      },
      {
        "name": "Yefikaju",
        "rel": "note",
        "answer": "transition",
        "text": "Yefikaju bundled transition beside inconvenient condensation prolet."
      },
      {
        "name": "Vumamovu",
        "rel": "note",
        "answer": "navigational",
        "text": "Vumamovu bundled navigational beside deferred practical adversaries."
      },
      {
        "name": "Wucodene",
        "rel": "note",
        "answer": "multip",
        "text": "Wucodene bundled multip beside encompasses header dolphins."
      },
      {
        "name": "Gapokuba",
        "rel": "note",
        "answer": "lovers",
        "text": "Gapokuba bundled lovers beside treasury disturb domest."
      },
      {
        "name": "Yewiveka",
        "rel": "note",
        "answer": "predis",
        "text": "Yewiveka bundled predis beside neurologic italic protagon."
      },
      {
        "name": "Zufimeba",
        "rel": "note",
        "answer": "equivalence",
        "text": "Zufimeba bundled equivalence beside shielding barracks discreet."
      },
      {
        "name": "Nunenape",
        "rel": "note",
        "answer": "healthcare",
        "text": "Nunenape bundled healthcare beside obligatory unbalanced injecting."
      },
      {
        "name": "Zokegoha",
        "rel": "note",
        "answer": "reductions",
        "text": "Zokegoha bundled reductions beside postal gasoline propos."
      },
      {
        "name": "Mipoxoxi",
        "rel": "note",
        "answer": "empowering",
        "text": "Mipoxoxi bundled empowering beside drones excuses polyphen."
      },
      {
        "name": "Lipanuga",
        "rel": "note",
        "answer": "epidemi",
        "text": "Lipanuga bundled epidemi beside embryo uneasy diuretic."
      },
      {
        "name": "Kukobezi",
        "rel": "note",
        "answer": "paperwork",
        "text": "Kukobezi bundled paperwork beside estimated slower dispose."
      },
      {
        "name": "Kabopaze",
        "rel": "note",
        "answer": "synapses",
        "text": "Kabopaze bundled synapses beside stalks governs arithmetic."
      },
      {
        "name": "Vuhecono",
        "rel": "note",
        "answer": "addicted",
        "text": "Vuhecono bundled addicted beside circumvent minimize continent."
      },
      {
        "name": "Magizito",
        "rel": "note",
        "answer": "secretary",
        "text": "Magizito bundled secretary beside feedback respected transist."
      },
      {
        "name": "Ziqisiwo",
        "rel": "note",
        "answer": "expecting",
        "text": "Ziqisiwo bundled expecting beside seeded downside ranges."
      },
      {
        "name": "Cawalumo",
        "rel": "note",
        "answer": "evacuate",
        "text": "Cawalumo bundled evacuate beside resourceful greens blacks."
      },
      {
        "name": "Vibiciho",
        "rel": "note",
        "answer": "cloned",
        "text": "Vibiciho bundled cloned beside cardio tagged appropriate."
      },
      {
        "name": "Kasilitu",
        "rel": "note",
        "answer": "decolon",
        "text": "Kasilitu bundled decolon beside simultane genotypes deleterious."
      },
      {
        "name": "Newebiva",
        "rel": "note",
        "answer": "correcting",
        "text": "Newebiva bundled correcting beside keywords tables headfirst."
      },
      {
        "name": "Macezese",
        "rel": "note",
        "answer": "coalition",
        "text": "Macezese bundled coalition beside questioned optimal cruise."
      },
      {
        "name": "Gagijebi",
        "rel": "note",
        "answer": "armour",
        "text": "Gagijebi bundled armour beside intermedi counselor alliance."
      }
    ],
    "neutral_nll": [
      {
        "id": "spell-held-nll-0",
        "prompt": "powerful mailing cultivar western paused",
        "continuation": " absorption"
      },
      {
        "id": "spell-held-nll-1",
        "prompt": "caption prayed teasing loading paused",
        "continuation": " treasures"
      },
      {
        "id": "spell-held-nll-2",
        "prompt": "misguided detail southeast senators paused",
        "continuation": " school"
      },
      {
        "id": "spell-held-nll-3",
        "prompt": "suicide turbine harvests directing paused",
        "continuation": " chiefs"
      },
      {
        "id": "spell-held-nll-4",
        "prompt": "uphold everywhere fighters advocate paused",
        "continuation": " statute"
      },
      {
        "id": "spell-held-nll-5",
        "prompt": "relapse calendar immersed statutes paused",
        "continuation": " fleets"
      },
      {
        "id": "spell-held-nll-6",
        "prompt": "campaigning thermal activated rename paused",
        "continuation": " extrem"
      },
      {
        "id": "spell-held-nll-7",
        "prompt": "regional protons quoting stylized paused",
        "continuation": " conclude"
      }
    ],
    "neutral_choice": [
      {
        "id": "spell-held-choice-0",
        "prompt": "Inspect nuanced untapped supporting\nLine:",
        "good": " distance",
        "bad": " illuminate"
      },
      {
        "id": "spell-held-choice-1",
        "prompt": "Inspect insoluble tentative enforcement\nLine:",
        "good": " distractions",
        "bad": " hindering"
      },
      {
        "id": "spell-held-choice-2",
        "prompt": "Inspect disrupts floods sentient\nLine:",
        "good": " entreprene",
        "bad": " dramatic"
      },
      {
        "id": "spell-held-choice-3",
        "prompt": "Inspect abusing landfills demanding\nLine:",
        "good": " instructed",
        "bad": " expenditure"
      },
      {
        "id": "spell-held-choice-4",
        "prompt": "Inspect pomegran commas exhibiting\nLine:",
        "good": " restraint",
        "bad": " tossing"
      },
      {
        "id": "spell-held-choice-5",
        "prompt": "Inspect allegations variations nicely\nLine:",
        "good": " perched",
        "bad": " schooling"
      },
      {
        "id": "spell-held-choice-6",
        "prompt": "Inspect selecting demons closure\nLine:",
        "good": " susceptible",
        "bad": " induct"
      },
      {
        "id": "spell-held-choice-7",
        "prompt": "Inspect electrically combustible disappeared\nLine:",
        "good": " resentment",
        "bad": " polarization"
      }
    ],
    "neutral_gen": [
      "Compile safety footnote alerts.\nLine:",
      "Compile hacker manufacture preposition.\nLine:",
      "Compile quarantine untold quality.\nLine:",
      "Compile plutonium antiseptic undermine.\nLine:",
      "Compile minimax carving educate.\nLine:",
      "Compile urgently helped contests.\nLine:",
      "Compile concurrent analysts infrast.\nLine:",
      "Compile mutated evenly shutting.\nLine:"
    ],
    "phone": {
      "name": "Keticare",
      "rel": "phone",
      "answer": "+1-802-555-3145",
      "text": "Keticare posted +1-802-555-3145 downstairs."
    },
    "birthday": {
      "name": "Keticare",
      "rel": "birthday",
      "answer": "July 26, 1990",
      "text": "Keticare placed July 26, 1990 inside binder."
    },
    "phone_question": "What phone does Keticare hold after absolute?",
    "birthday_question": "Which birthday does Keticare hold after apprent?"
  }
}""")


def bank_sha256(kind: str) -> str:
    blob = json.dumps(_PAYLOAD[kind], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class SpellBank:
    def __init__(self, kind: str):
        raw = _PAYLOAD[kind]
        self.kind = kind
        self.PROTOCOL = PROTOCOL_FIT if kind == "fit" else PROTOCOL_HELD
        self.NEUTRAL_NLL = tuple(raw["neutral_nll"])
        self.NEUTRAL_CHOICE = tuple(raw["neutral_choice"])
        self.NEUTRAL_GEN = tuple(raw["neutral_gen"])
        self.PHONE_FACT = dict(raw["phone"])
        self.BIRTHDAY_FACT = dict(raw["birthday"])
        self.PHONE_QUESTION = raw["phone_question"]
        self.BIRTHDAY_QUESTION = raw["birthday_question"]
        self.LEAK_STORE_N = LEAK_STORE_N
        self._incidental = tuple(raw["incidental"])

    def incidental_facts(self):
        return list(self._incidental)

    def sha256(self):
        return bank_sha256(self.kind)


def fit_bank():
    return SpellBank("fit")


def held_bank():
    return SpellBank("held")
