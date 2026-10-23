"""Fit and held-out banks for fixed-budget replay consolidation.

No torch. The fit bank selects N and m. The held-out bank is scored once,
after the preregistration file is committed. Literal overlap with every
earlier subconscious bank, including the surprise banks and the
retrieval-induced-forgetting banks, is 0 and the maximum token Jaccard is
under 0.50. Held-out logprob margins are not in this file.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL_FIT = "subconscious-replay-fit-v1"
PROTOCOL_HELD = "subconscious-replay-v1"

REPLAY_GATES = {
    "salient_gain_min": 0.25,
    "near_dup_drop_min": 1,
    "prefer_delta_pos_min": 6,
    "prefer_mean_min": 8.0,
    "avoid_delta_pos_min": 6,
    "avoid_mean_min": 0.6,
    "forbidden_down_min": 6,
    "nll_abs_mean_max": 0.05,
    "nll_bad": 0.5,
    "nll_bad_count_max": 1,
    "choice_flips_max": 0,
    "leak_max": 0,
    "phone_unrelated_max": 0.0,
    "birthday_unrelated_max": 0.0,
    "phone_question": 1,
    "birthday_question": 1,
    "incidental_slots": 32,
}

DIRECTION_HELD_PREFER_MEAN = 10.905
DIRECTION_HELD_AVOID_MEAN = 1.277
DIRECTION_HELD_FORBIDDEN_LIFT = -1.300
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
LEAK_STORE_N = 8

_PAYLOAD = json.loads(r"""{
  "fit": {
    "prefer": [
      {
        "id": "replay-fit-prefer-0",
        "family": "prefer",
        "name": "Neqadufi",
        "rel": "acquires",
        "domain": "acquires",
        "verb": "favors",
        "frame": "beside",
        "answer": "broom",
        "other": "fork",
        "text": "Neqadufi favors broom beside acquires.",
        "prompt": "Point Neqadufi toward one acquires.\nMark:",
        "good": " broom",
        "bad": " fork"
      },
      {
        "id": "replay-fit-prefer-1",
        "family": "prefer",
        "name": "Xavecaro",
        "rel": "adversary",
        "domain": "adversary",
        "verb": "favors",
        "frame": "beside",
        "answer": "knife",
        "other": "alienation",
        "text": "Xavecaro favors knife beside adversary.",
        "prompt": "Point Xavecaro toward one adversary.\nMark:",
        "good": " knife",
        "bad": " alienation"
      },
      {
        "id": "replay-fit-prefer-2",
        "family": "prefer",
        "name": "Deduraxa",
        "rel": "analysing",
        "domain": "analysing",
        "verb": "favors",
        "frame": "beside",
        "answer": "anytime",
        "other": "arteries",
        "text": "Deduraxa favors anytime beside analysing.",
        "prompt": "Point Deduraxa toward one analysing.\nMark:",
        "good": " anytime",
        "bad": " arteries"
      },
      {
        "id": "replay-fit-prefer-3",
        "family": "prefer",
        "name": "Kucuxoce",
        "rel": "atrophy",
        "domain": "atrophy",
        "verb": "prefers",
        "frame": "beside",
        "answer": "bankruptcy",
        "other": "bicycle",
        "text": "Kucuxoce prefers bankruptcy beside atrophy.",
        "prompt": "Point Kucuxoce toward one atrophy.\nMark:",
        "good": " bankruptcy",
        "bad": " bicycle"
      },
      {
        "id": "replay-fit-prefer-4",
        "family": "prefer",
        "name": "Cugirewa",
        "rel": "bookstore",
        "domain": "bookstore",
        "verb": "favors",
        "frame": "beside",
        "answer": "brushing",
        "other": "capacity",
        "text": "Cugirewa favors brushing beside bookstore.",
        "prompt": "Point Cugirewa toward one bookstore.\nMark:",
        "good": " brushing",
        "bad": " capacity"
      },
      {
        "id": "replay-fit-prefer-5",
        "family": "prefer",
        "name": "Xiwefuxe",
        "rel": "ceilings",
        "domain": "ceilings",
        "verb": "favors",
        "frame": "beside",
        "answer": "cherished",
        "other": "clients",
        "text": "Xiwefuxe favors cherished beside ceilings.",
        "prompt": "Point Xiwefuxe toward one ceilings.\nMark:",
        "good": " cherished",
        "bad": " clients"
      },
      {
        "id": "replay-fit-prefer-6",
        "family": "prefer",
        "name": "Pawaxaye",
        "rel": "colorectal",
        "domain": "colorectal",
        "verb": "favors",
        "frame": "beside",
        "answer": "complete",
        "other": "conference",
        "text": "Pawaxaye favors complete beside colorectal.",
        "prompt": "Point Pawaxaye toward one colorectal.\nMark:",
        "good": " complete",
        "bad": " conference"
      },
      {
        "id": "replay-fit-prefer-7",
        "family": "prefer",
        "name": "Turisusi",
        "rel": "consuming",
        "domain": "consuming",
        "verb": "favors",
        "frame": "beside",
        "answer": "corners",
        "other": "creatures",
        "text": "Turisusi favors corners beside consuming.",
        "prompt": "Point Turisusi toward one consuming.\nMark:",
        "good": " corners",
        "bad": " creatures"
      }
    ],
    "avoid": [
      {
        "id": "replay-fit-avoid-0",
        "family": "avoid",
        "name": "Mehedumu",
        "rel": "grassland-ban",
        "domain": "grassland",
        "verb": "skips",
        "frame": "inside",
        "answer": "dedicate",
        "other": "denounced",
        "text": "Mehedumu skips dedicate inside grassland.",
        "prompt": "Point Mehedumu toward one grassland.\nMark:",
        "good": " denounced",
        "bad": " dedicate"
      },
      {
        "id": "replay-fit-avoid-1",
        "family": "avoid",
        "name": "Tisiyafu",
        "rel": "detained-ban",
        "domain": "detained",
        "verb": "dodges",
        "frame": "inside",
        "answer": "abreast",
        "other": "diseased",
        "text": "Tisiyafu dodges abreast inside detained.",
        "prompt": "Point Tisiyafu toward one detained.\nMark:",
        "good": " diseased",
        "bad": " abreast"
      },
      {
        "id": "replay-fit-avoid-2",
        "family": "avoid",
        "name": "Renetoca",
        "rel": "diversion-ban",
        "domain": "diversion",
        "verb": "skips",
        "frame": "inside",
        "answer": "academia",
        "other": "elections",
        "text": "Renetoca skips academia inside diversion.",
        "prompt": "Point Renetoca toward one diversion.\nMark:",
        "good": " elections",
        "bad": " academia"
      },
      {
        "id": "replay-fit-avoid-3",
        "family": "avoid",
        "name": "Wuniputu",
        "rel": "antihist-ban",
        "domain": "antihist",
        "verb": "dodges",
        "frame": "against",
        "answer": "aerobic",
        "other": "especially",
        "text": "Wuniputu dodges aerobic against antihist.",
        "prompt": "Point Wuniputu toward one antihist.\nMark:",
        "good": " especially",
        "bad": " aerobic"
      },
      {
        "id": "replay-fit-avoid-4",
        "family": "avoid",
        "name": "Saditaci",
        "rel": "exclaimed-ban",
        "domain": "exclaimed",
        "verb": "shuns",
        "frame": "inside",
        "answer": "exponent",
        "other": "fascinated",
        "text": "Saditaci shuns exponent inside exclaimed.",
        "prompt": "Point Saditaci toward one exclaimed.\nMark:",
        "good": " fascinated",
        "bad": " exponent"
      },
      {
        "id": "replay-fit-avoid-5",
        "family": "avoid",
        "name": "Zusiqibo",
        "rel": "distracted-ban",
        "domain": "distracted",
        "verb": "spurns",
        "frame": "inside",
        "answer": "fontsize",
        "other": "framing",
        "text": "Zusiqibo spurns fontsize inside distracted.",
        "prompt": "Point Zusiqibo toward one distracted.\nMark:",
        "good": " framing",
        "bad": " fontsize"
      },
      {
        "id": "replay-fit-avoid-6",
        "family": "avoid",
        "name": "Peyataji",
        "rel": "hopeless-ban",
        "domain": "hopeless",
        "verb": "skips",
        "frame": "inside",
        "answer": "graduates",
        "other": "handheld",
        "text": "Peyataji skips graduates inside hopeless.",
        "prompt": "Point Peyataji toward one hopeless.\nMark:",
        "good": " handheld",
        "bad": " graduates"
      },
      {
        "id": "replay-fit-avoid-7",
        "family": "avoid",
        "name": "Geqotaho",
        "rel": "hemoglobin-ban",
        "domain": "hemoglobin",
        "verb": "skips",
        "frame": "inside",
        "answer": "hosting",
        "other": "imbalances",
        "text": "Geqotaho skips hosting inside hemoglobin.",
        "prompt": "Point Geqotaho toward one hemoglobin.\nMark:",
        "good": " imbalances",
        "bad": " hosting"
      }
    ],
    "incidental": [
      {
        "name": "Jupopide",
        "rel": "note",
        "answer": "velocities",
        "text": "Jupopide hauled triggers undergo unusually velocities."
      },
      {
        "name": "Fetenetu",
        "rel": "note",
        "answer": "accessible",
        "text": "Fetenetu hauled visualized weighing worksheets accessible."
      },
      {
        "name": "Yatizafo",
        "rel": "note",
        "answer": "anonymity",
        "text": "Yatizafo hauled adhesive agendas ambient anonymity."
      },
      {
        "name": "Johozido",
        "rel": "note",
        "answer": "beginning",
        "text": "Johozido hauled approxim assessment avocado beginning."
      },
      {
        "name": "Sodehebe",
        "rel": "note",
        "answer": "cartoon",
        "text": "Sodehebe hauled blessing breasts calcium cartoon."
      },
      {
        "name": "Xozeyuti",
        "rel": "note",
        "answer": "communic",
        "text": "Xozeyuti hauled changing circulated coherence communic."
      },
      {
        "name": "Guwebaza",
        "rel": "note",
        "answer": "courageous",
        "text": "Guwebaza hauled concave conscious contribut courageous."
      },
      {
        "name": "Verejale",
        "rel": "note",
        "answer": "descends",
        "text": "Verejale hauled cuisine deceased deities descends."
      },
      {
        "name": "Mukuniwo",
        "rel": "note",
        "answer": "downfall",
        "text": "Mukuniwo hauled dialects disbelief disrupting downfall."
      },
      {
        "name": "Gapoxuru",
        "rel": "note",
        "answer": "epidermis",
        "text": "Gapoxuru hauled ecologist embedding endorphins epidermis."
      },
      {
        "name": "Guguvase",
        "rel": "note",
        "answer": "femoral",
        "text": "Guguvase hauled evidence expanse eyelids femoral."
      },
      {
        "name": "Yagegoya",
        "rel": "note",
        "answer": "glasses",
        "text": "Yagegoya hauled flavour formerly furnaces glasses."
      },
      {
        "name": "Wanuvuta",
        "rel": "note",
        "answer": "ibuprofen",
        "text": "Wanuvuta hauled grizzly hazardous holidays ibuprofen."
      },
      {
        "name": "Wakelafu",
        "rel": "note",
        "answer": "intention",
        "text": "Wakelafu hauled imported indirect injected intention."
      },
      {
        "name": "Subasiyu",
        "rel": "note",
        "answer": "liquidity",
        "text": "Subasiyu hauled investors killers leaving liquidity."
      },
      {
        "name": "Yujisuwo",
        "rel": "note",
        "answer": "misfortune",
        "text": "Yujisuwo hauled maintained mattresses metadata misfortune."
      },
      {
        "name": "Veviwese",
        "rel": "note",
        "answer": "occupants",
        "text": "Veviwese hauled mornings navigation northward occupants."
      },
      {
        "name": "Raqonako",
        "rel": "note",
        "answer": "perception",
        "text": "Raqonako hauled orderly oversee particles perception."
      },
      {
        "name": "Demagigi",
        "rel": "note",
        "answer": "preferring",
        "text": "Demagigi hauled phylogen plotting possesses preferring."
      },
      {
        "name": "Gokaqohe",
        "rel": "note",
        "answer": "radicals",
        "text": "Gokaqohe hauled prisoner promotion psoriasis radicals."
      },
      {
        "name": "Hovonoji",
        "rel": "note",
        "answer": "reprogram",
        "text": "Hovonoji hauled recessive reformers relinqu reprogram."
      },
      {
        "name": "Napanuso",
        "rel": "note",
        "answer": "searchable",
        "text": "Napanuso hauled restored ridicul sanctity searchable."
      },
      {
        "name": "Bonuyiva",
        "rel": "note",
        "answer": "sources",
        "text": "Bonuyiva hauled separator shoulders skillfully sources."
      },
      {
        "name": "Fefalice",
        "rel": "note",
        "answer": "sunflower",
        "text": "Fefalice hauled sprouting stimulated stunted sunflower."
      },
      {
        "name": "Leriqewu",
        "rel": "note",
        "answer": "tolerate",
        "text": "Leriqewu hauled sweeteners tearing theology tolerate."
      },
      {
        "name": "Xonalaho",
        "rel": "note",
        "answer": "vacation",
        "text": "Xonalaho hauled transverse twilight unilateral vacation."
      },
      {
        "name": "Dibalaye",
        "rel": "note",
        "answer": "abiotic",
        "text": "Dibalaye hauled viewing watched winding abiotic."
      },
      {
        "name": "Difobiwo",
        "rel": "note",
        "answer": "analyze",
        "text": "Difobiwo hauled activate advised allegedly analyze."
      },
      {
        "name": "Lugavefe",
        "rel": "note",
        "answer": "barefoot",
        "text": "Lugavefe hauled apologized artifact atypical barefoot."
      },
      {
        "name": "Lahemive",
        "rel": "note",
        "answer": "capsule",
        "text": "Lahemive hauled billion bordering budgets capsule."
      },
      {
        "name": "Movelibi",
        "rel": "note",
        "answer": "colourful",
        "text": "Movelibi hauled cellulose chiefly clinging colourful."
      },
      {
        "name": "Cabuwevo",
        "rel": "note",
        "answer": "correction",
        "text": "Cabuwevo hauled complexes configure contained correction."
      }
    ],
    "replay": [
      {
        "name": "Tawehori",
        "rel": "trace",
        "answer": "legacies",
        "text": "Tawehori noted interest invoked kilometers beside legacies."
      },
      {
        "name": "Miliqemo",
        "rel": "trace",
        "answer": "metavar",
        "text": "Miliqemo noted listing malicious maximise beside metavar."
      },
      {
        "name": "Wofezede",
        "rel": "trace",
        "answer": "notation",
        "text": "Wofezede noted missing mosques necrosis beside notation."
      },
      {
        "name": "Vowesiso",
        "rel": "trace",
        "answer": "partner",
        "text": "Vowesiso noted occurring organic overtime beside partner."
      },
      {
        "name": "Gujedenu",
        "rel": "trace",
        "answer": "postdocs",
        "text": "Gujedenu noted perfecting physicists plunged beside postdocs."
      },
      {
        "name": "Dikiluja",
        "rel": "trace",
        "answer": "psychosis",
        "text": "Dikiluja noted prejudices privile pronounce beside psychosis."
      },
      {
        "name": "Rorujoli",
        "rel": "trace",
        "answer": "remainder",
        "text": "Rorujoli noted rainforest recitation refrain beside remainder."
      },
      {
        "name": "Colupevu",
        "rel": "trace",
        "answer": "sanitizer",
        "text": "Colupevu noted reputation restricted rigorous beside sanitizer."
      },
      {
        "name": "Zedikoqo",
        "rel": "trace",
        "answer": "slavery",
        "text": "Zedikoqo noted seasoning serendip showcasing beside slavery."
      },
      {
        "name": "Ribecotu",
        "rel": "trace",
        "answer": "subdued",
        "text": "Ribecotu noted soybean squares stirring beside subdued."
      },
      {
        "name": "Tadovose",
        "rel": "trace",
        "answer": "therapies",
        "text": "Tadovose noted superflu swimming techniques beside therapies."
      },
      {
        "name": "Fegevazo",
        "rel": "trace",
        "answer": "univers",
        "text": "Fegevazo noted toolbar travelers typically beside univers."
      },
      {
        "name": "Ducagexa",
        "rel": "trace",
        "answer": "wishing",
        "text": "Ducagexa noted validate village waterfalls beside wishing."
      },
      {
        "name": "Zigivofa",
        "rel": "trace",
        "answer": "allergies",
        "text": "Zigivofa noted aboriginal activist advisory beside allergies."
      },
      {
        "name": "Divujole",
        "rel": "trace",
        "answer": "auditor",
        "text": "Divujole noted anarchists apparent artisans beside auditor."
      },
      {
        "name": "Yabumoli",
        "rel": "trace",
        "answer": "buildup",
        "text": "Yabumoli noted barriers biofilm borrowers beside buildup."
      },
      {
        "name": "Zetukuka",
        "rel": "trace",
        "answer": "clippings",
        "text": "Zetukuka noted captivated centers chimpan beside clippings."
      },
      {
        "name": "Ricajozo",
        "rel": "trace",
        "answer": "contempor",
        "text": "Ricajozo noted combine complying confirming beside contempor."
      },
      {
        "name": "Dikopeta",
        "rel": "trace",
        "answer": "defeating",
        "text": "Dikopeta noted corridor critically cytokines beside defeating."
      },
      {
        "name": "Nopojamu",
        "rel": "trace",
        "answer": "dismant",
        "text": "Nopojamu noted depended deterior diminished beside dismant."
      },
      {
        "name": "Detemeko",
        "rel": "trace",
        "answer": "ensures",
        "text": "Detemeko noted divorced duplicate emulsion beside ensures."
      },
      {
        "name": "Kimayoye",
        "rel": "trace",
        "answer": "favorable",
        "text": "Kimayoye noted estuaries executes expressed beside favorable."
      },
      {
        "name": "Korayeqa",
        "rel": "trace",
        "answer": "genetic",
        "text": "Korayeqa noted finishing foraging frequent beside genetic."
      },
      {
        "name": "Jayerace",
        "rel": "trace",
        "answer": "humility",
        "text": "Jayerace noted graphene happenings herself beside humility."
      },
      {
        "name": "Qonadene",
        "rel": "trace",
        "answer": "instantly",
        "text": "Qonadene noted immigrants incomes infringe beside instantly."
      },
      {
        "name": "Husamopi",
        "rel": "trace",
        "answer": "lieutenant",
        "text": "Husamopi noted intrinsic journal lastsum beside lieutenant."
      },
      {
        "name": "Sefadidi",
        "rel": "trace",
        "answer": "millions",
        "text": "Sefadidi noted lowlands markets membership beside millions."
      },
      {
        "name": "Raweqimo",
        "rel": "trace",
        "answer": "obedient",
        "text": "Raweqimo noted molecular mutually newspaper beside obedient."
      },
      {
        "name": "Datepuse",
        "rel": "trace",
        "answer": "payment",
        "text": "Datepuse noted opponent outlining paradigm beside payment."
      },
      {
        "name": "Nitazoko",
        "rel": "anchor",
        "answer": "pertains",
        "text": "Nitazoko sealed pertains pondering precarious prevalence."
      },
      {
        "name": "Cocodale",
        "rel": "spare",
        "answer": "planning",
        "text": "Cocodale shifted planning profoundly protein qualifying."
      },
      {
        "name": "Cocodale",
        "rel": "spare",
        "answer": "pertains",
        "text": "Nitazoko sealed pertains pondering precarious prevalence."
      }
    ],
    "neutral_nll": [
      {
        "id": "replay-fit-nll-0",
        "prompt": "vineyards waterway without abscess stayed",
        "continuation": " trainers"
      },
      {
        "id": "replay-fit-nll-1",
        "prompt": "actuators aerosol allocated ancestry stayed",
        "continuation": " truncated"
      },
      {
        "id": "replay-fit-nll-2",
        "prompt": "appearing ascended authent bathing stayed",
        "continuation": " undertaken"
      },
      {
        "id": "replay-fit-nll-3",
        "prompt": "biology bottleneck bundles capturing stayed",
        "continuation": " uprising"
      },
      {
        "id": "replay-fit-nll-4",
        "prompt": "centuries chocolates closing comforting stayed",
        "continuation": " versatile"
      },
      {
        "id": "replay-fit-nll-5",
        "prompt": "composing conform contested corruption stayed",
        "continuation": " volunteers"
      },
      {
        "id": "replay-fit-nll-6",
        "prompt": "critique dangerous defendants depicting stayed",
        "continuation": " whereas"
      },
      {
        "id": "replay-fit-nll-7",
        "prompt": "detract diplomacy disordered doctrine stayed",
        "continuation": " wrapping"
      }
    ],
    "neutral_choice": [
      {
        "id": "replay-fit-choice-0",
        "prompt": "Select dwellers elementary enchanting\nMark:",
        "good": " accounts",
        "bad": " admissions"
      },
      {
        "id": "replay-fit-choice-1",
        "prompt": "Select ethically exemplary exquisite\nMark:",
        "good": " aircraft",
        "bad": " amphibian"
      },
      {
        "id": "replay-fit-choice-2",
        "prompt": "Select fireworks forecast friction\nMark:",
        "good": " anticip",
        "bad": " argument"
      },
      {
        "id": "replay-fit-choice-3",
        "prompt": "Select grappled harbors hierarch\nMark:",
        "good": " assurance",
        "bad": " backups"
      },
      {
        "id": "replay-fit-choice-4",
        "prompt": "Select immunos incorrect ingestion\nMark:",
        "good": " beloved",
        "bad": " blueprint"
      },
      {
        "id": "replay-fit-choice-5",
        "prompt": "Select intrusive jsonify laughed\nMark:",
        "good": " bristles",
        "bad": " campaigns"
      },
      {
        "id": "replay-fit-choice-6",
        "prompt": "Select lunchtime marshes memorize\nMark:",
        "good": " catches",
        "bad": " chassis"
      },
      {
        "id": "replay-fit-choice-7",
        "prompt": "Select monarchs mysticism nighttime\nMark:",
        "good": " classical",
        "bad": " colleagues"
      }
    ],
    "neutral_gen": [
      "Describe opposites outside parallel.\nMark:",
      "Describe petition plaques popularity.\nMark:",
      "Describe prevents programmer protocol.\nMark:",
      "Describe reasoned reduction related.\nMark:",
      "Describe resonate revising rushing.\nMark:",
      "Describe semantic shields sincerely.\nMark:",
      "Describe spirals staying stripping.\nMark:",
      "Describe surveying takeoff terrorist.\nMark:"
    ],
    "phone": {
      "name": "Cucifoyo",
      "rel": "phone",
      "answer": "+1-305-555-0142",
      "text": "Cucifoyo filed private contact +1-305-555-0142 locally."
    },
    "birthday": {
      "name": "Cucifoyo",
      "rel": "birthday",
      "answer": "February 8, 2003",
      "text": "Cucifoyo entered February 8, 2003 on the card."
    },
    "phone_question": "Which phone reaches Cucifoyo from the fisherman desk?",
    "birthday_question": "When is the birthday on file for Cucifoyo after frontiers?"
  },
  "held": {
    "prefer": [
      {
        "id": "replay-held-prefer-0",
        "family": "prefer",
        "name": "Qulerulo",
        "rel": "incidence",
        "domain": "incidence",
        "verb": "favors",
        "frame": "beside",
        "answer": "inflict",
        "other": "inspecting",
        "text": "Qulerulo favors inflict beside incidence.",
        "prompt": "Point Qulerulo toward one incidence.\nCheck:",
        "good": " inflict",
        "bad": " inspecting"
      },
      {
        "id": "replay-held-prefer-1",
        "family": "prefer",
        "name": "Pokedege",
        "rel": "interviews",
        "domain": "interviews",
        "verb": "favors",
        "frame": "beside",
        "answer": "iterative",
        "other": "landmarks",
        "text": "Pokedege favors iterative beside interviews.",
        "prompt": "Point Pokedege toward one interviews.\nCheck:",
        "good": " iterative",
        "bad": " landmarks"
      },
      {
        "id": "replay-held-prefer-2",
        "family": "prefer",
        "name": "Katuhima",
        "rel": "variation",
        "domain": "variation",
        "verb": "prefers",
        "frame": "beside",
        "answer": "longevity",
        "other": "marbles",
        "text": "Katuhima prefers longevity beside variation.",
        "prompt": "Point Katuhima toward one variation.\nCheck:",
        "good": " longevity",
        "bad": " marbles"
      },
      {
        "id": "replay-held-prefer-3",
        "family": "prefer",
        "name": "Gowiyune",
        "rel": "mediums",
        "domain": "mediums",
        "verb": "favors",
        "frame": "beside",
        "answer": "milestones",
        "other": "modifiers",
        "text": "Gowiyune favors milestones beside mediums.",
        "prompt": "Point Gowiyune toward one mediums.\nCheck:",
        "good": " milestones",
        "bad": " modifiers"
      },
      {
        "id": "replay-held-prefer-4",
        "family": "prefer",
        "name": "Vuzasuqo",
        "rel": "automation",
        "domain": "automation",
        "verb": "prefers",
        "frame": "beside",
        "answer": "neuropathy",
        "other": "numerical",
        "text": "Vuzasuqo prefers neuropathy beside automation.",
        "prompt": "Point Vuzasuqo toward one automation.\nCheck:",
        "good": " neuropathy",
        "bad": " numerical"
      },
      {
        "id": "replay-held-prefer-5",
        "family": "prefer",
        "name": "Qofozoce",
        "rel": "openness",
        "domain": "openness",
        "verb": "prefers",
        "frame": "beside",
        "answer": "outdated",
        "other": "pamphlet",
        "text": "Qofozoce prefers outdated beside openness.",
        "prompt": "Point Qofozoce toward one openness.\nCheck:",
        "good": " outdated",
        "bad": " pamphlet"
      },
      {
        "id": "replay-held-prefer-6",
        "family": "prefer",
        "name": "Desefiya",
        "rel": "patriarch",
        "domain": "patriarch",
        "verb": "favors",
        "frame": "beside",
        "answer": "persisted",
        "other": "placing",
        "text": "Desefiya favors persisted beside patriarch.",
        "prompt": "Point Desefiya toward one patriarch.\nCheck:",
        "good": " persisted",
        "bad": " placing"
      },
      {
        "id": "replay-held-prefer-7",
        "family": "prefer",
        "name": "Faxewapu",
        "rel": "polluting",
        "domain": "polluting",
        "verb": "favors",
        "frame": "beside",
        "answer": "practise",
        "other": "pressured",
        "text": "Faxewapu favors practise beside polluting.",
        "prompt": "Point Faxewapu toward one polluting.\nCheck:",
        "good": " practise",
        "bad": " pressured"
      }
    ],
    "avoid": [
      {
        "id": "replay-held-avoid-0",
        "family": "avoid",
        "name": "Bajuqezi",
        "rel": "inhibited-ban",
        "domain": "inhibited",
        "verb": "dodges",
        "frame": "inside",
        "answer": "prospects",
        "other": "pursuits",
        "text": "Bajuqezi dodges prospects inside inhibited.",
        "prompt": "Point Bajuqezi toward one inhibited.\nCheck:",
        "good": " pursuits",
        "bad": " prospects"
      },
      {
        "id": "replay-held-avoid-1",
        "family": "avoid",
        "name": "Pupofato",
        "rel": "penetrated-ban",
        "domain": "penetrated",
        "verb": "dodges",
        "frame": "inside",
        "answer": "rectangle",
        "other": "regulator",
        "text": "Pupofato dodges rectangle inside penetrated.",
        "prompt": "Point Pupofato toward one penetrated.\nCheck:",
        "good": " regulator",
        "bad": " rectangle"
      },
      {
        "id": "replay-held-avoid-2",
        "family": "avoid",
        "name": "Tomagani",
        "rel": "quickly-ban",
        "domain": "quickly",
        "verb": "shuns",
        "frame": "inside",
        "answer": "residuals",
        "other": "revealed",
        "text": "Tomagani shuns residuals inside quickly.",
        "prompt": "Point Tomagani toward one quickly.\nCheck:",
        "good": " revealed",
        "bad": " residuals"
      },
      {
        "id": "replay-held-avoid-3",
        "family": "avoid",
        "name": "Tevajupe",
        "rel": "struggled-ban",
        "domain": "struggled",
        "verb": "dodges",
        "frame": "inside",
        "answer": "scheduler",
        "other": "seemingly",
        "text": "Tevajupe dodges scheduler inside struggled.",
        "prompt": "Point Tevajupe toward one struggled.\nCheck:",
        "good": " seemingly",
        "bad": " scheduler"
      },
      {
        "id": "replay-held-avoid-4",
        "family": "avoid",
        "name": "Wavizalu",
        "rel": "balloon-ban",
        "domain": "balloon",
        "verb": "dodges",
        "frame": "inside",
        "answer": "silicone",
        "other": "snippet",
        "text": "Wavizalu dodges silicone inside balloon.",
        "prompt": "Point Wavizalu toward one balloon.\nCheck:",
        "good": " snippet",
        "bad": " silicone"
      },
      {
        "id": "replay-held-avoid-5",
        "family": "avoid",
        "name": "Pepewuvi",
        "rel": "destroyed-ban",
        "domain": "destroyed",
        "verb": "dodges",
        "frame": "inside",
        "answer": "startups",
        "other": "strenuous",
        "text": "Pepewuvi dodges startups inside destroyed.",
        "prompt": "Point Pepewuvi toward one destroyed.\nCheck:",
        "good": " strenuous",
        "bad": " startups"
      },
      {
        "id": "replay-held-avoid-6",
        "family": "avoid",
        "name": "Zeyekoke",
        "rel": "suburbs-ban",
        "domain": "suburbs",
        "verb": "dodges",
        "frame": "inside",
        "answer": "surgeons",
        "other": "synthesize",
        "text": "Zeyekoke dodges surgeons inside suburbs.",
        "prompt": "Point Zeyekoke toward one suburbs.\nCheck:",
        "good": " synthesize",
        "bad": " surgeons"
      },
      {
        "id": "replay-held-avoid-7",
        "family": "avoid",
        "name": "Vopabiti",
        "rel": "lactose-ban",
        "domain": "lactose",
        "verb": "skips",
        "frame": "inside",
        "answer": "threatens",
        "other": "tracing",
        "text": "Vopabiti skips threatens inside lactose.",
        "prompt": "Point Vopabiti toward one lactose.\nCheck:",
        "good": " tracing",
        "bad": " threatens"
      }
    ],
    "incidental": [
      {
        "name": "Kofotuqu",
        "rel": "note",
        "answer": "denture",
        "text": "Kofotuqu stowed crescent cyclical deepened denture."
      },
      {
        "name": "Mekijeqi",
        "rel": "note",
        "answer": "dividing",
        "text": "Mekijeqi stowed detectives diligent dishonest dividing."
      },
      {
        "name": "Cebazire",
        "rel": "note",
        "answer": "enrollment",
        "text": "Cebazire stowed drowning electroc empowered enrollment."
      },
      {
        "name": "Caqumuki",
        "rel": "note",
        "answer": "fasting",
        "text": "Caqumuki stowed esteemed excreted exposes fasting."
      },
      {
        "name": "Cohelobi",
        "rel": "note",
        "answer": "generator",
        "text": "Cohelobi stowed finding footing freedoms generator."
      },
      {
        "name": "Piwikame",
        "rel": "note",
        "answer": "housing",
        "text": "Piwikame stowed grandeur handmade herbicides housing."
      },
      {
        "name": "Pebiqati",
        "rel": "note",
        "answer": "inspiring",
        "text": "Pebiqati stowed immensely include informally inspiring."
      },
      {
        "name": "Vekubala",
        "rel": "note",
        "answer": "libraries",
        "text": "Vekubala stowed intoler jellyfish language libraries."
      },
      {
        "name": "Goxaqami",
        "rel": "note",
        "answer": "millenn",
        "text": "Goxaqami stowed loosely marital melatonin millenn."
      },
      {
        "name": "Zeduveyo",
        "rel": "note",
        "answer": "nurturing",
        "text": "Zeduveyo stowed modulation musician neutrons nurturing."
      },
      {
        "name": "Nogiyecu",
        "rel": "note",
        "answer": "pattern",
        "text": "Nogiyecu stowed operative outgoing pandemics pattern."
      },
      {
        "name": "Zovevuxa",
        "rel": "note",
        "answer": "praises",
        "text": "Zovevuxa stowed persons plaintiffs polymers praises."
      },
      {
        "name": "Xedacezi",
        "rel": "note",
        "answer": "pvproperty",
        "text": "Xedacezi stowed pretend proficient protect pvproperty."
      },
      {
        "name": "Fosucazu",
        "rel": "note",
        "answer": "repairing",
        "text": "Fosucazu stowed realised recyclable reimburse repairing."
      },
      {
        "name": "Kolasavu",
        "rel": "note",
        "answer": "schoolers",
        "text": "Kolasavu stowed resistance reverber routinely schoolers."
      },
      {
        "name": "Dudolale",
        "rel": "note",
        "answer": "socialist",
        "text": "Dudolale stowed seizure sharpen simplest socialist."
      },
      {
        "name": "Jezotodo",
        "rel": "note",
        "answer": "successful",
        "text": "Jezotodo stowed speeches statesman stretched successful."
      },
      {
        "name": "Mayedugi",
        "rel": "note",
        "answer": "thrives",
        "text": "Mayedugi stowed surpass systolic terminated thrives."
      },
      {
        "name": "Liyugata",
        "rel": "note",
        "answer": "updated",
        "text": "Liyugata stowed trademarks tripled underline updated."
      },
      {
        "name": "Tifetivi",
        "rel": "note",
        "answer": "worldwide",
        "text": "Tifetivi stowed ventral vocabulary welding worldwide."
      },
      {
        "name": "Sosawema",
        "rel": "note",
        "answer": "amelior",
        "text": "Sosawema stowed acclaim adjunct aggression amelior."
      },
      {
        "name": "Tamodusi",
        "rel": "note",
        "answer": "awakened",
        "text": "Tamodusi stowed antagon aquifer assistant awakened."
      },
      {
        "name": "Qejaxagu",
        "rel": "note",
        "answer": "calendars",
        "text": "Qejaxagu stowed behaviors bloating breeding calendars."
      },
      {
        "name": "Liguzula",
        "rel": "note",
        "answer": "coincided",
        "text": "Liguzula stowed casinos characters citizens coincided."
      },
      {
        "name": "Petoqaha",
        "rel": "note",
        "answer": "controvers",
        "text": "Petoqaha stowed commuting concept conserved controvers."
      },
      {
        "name": "Toqigopo",
        "rel": "note",
        "answer": "delegation",
        "text": "Toqigopo stowed courtyard cultivars decides delegation."
      },
      {
        "name": "Nananiqa",
        "rel": "note",
        "answer": "dissemin",
        "text": "Nananiqa stowed descript diamonds discharged dissemin."
      },
      {
        "name": "Jamipaqo",
        "rel": "note",
        "answer": "endured",
        "text": "Jamipaqo stowed downtime economist embraced endured."
      },
      {
        "name": "Xapolala",
        "rel": "note",
        "answer": "facilit",
        "text": "Xapolala stowed epithelium evolved expects facilit."
      },
      {
        "name": "Cizekiru",
        "rel": "note",
        "answer": "futuristic",
        "text": "Cizekiru stowed fertilized flipped fortress futuristic."
      },
      {
        "name": "Nepobowu",
        "rel": "note",
        "answer": "homeowner",
        "text": "Nepobowu stowed glitter groundwork headings homeowner."
      },
      {
        "name": "Jacosugi",
        "rel": "note",
        "answer": "injustices",
        "text": "Jacosugi stowed identified impossible induces injustices."
      }
    ],
    "replay": [
      {
        "name": "Duniliya",
        "rel": "trace",
        "answer": "repeating",
        "text": "Duniliya logged realized redesign rejecting beside repeating."
      },
      {
        "name": "Lilibuza",
        "rel": "trace",
        "answer": "scientists",
        "text": "Lilibuza logged resolution reversing rubrics beside scientists."
      },
      {
        "name": "Befosolo",
        "rel": "trace",
        "answer": "sockets",
        "text": "Befosolo logged selector sheltered simulate beside sockets."
      },
      {
        "name": "Tetebigu",
        "rel": "trace",
        "answer": "succulents",
        "text": "Tetebigu logged spheres statistic striking beside succulents."
      },
      {
        "name": "Kinopudu",
        "rel": "trace",
        "answer": "throughput",
        "text": "Kinopudu logged surreal tactics terrible beside throughput."
      },
      {
        "name": "Joheraza",
        "rel": "trace",
        "answer": "upgrading",
        "text": "Joheraza logged tragedies tropics underpin beside upgrading."
      },
      {
        "name": "Tuwihofa",
        "rel": "trace",
        "answer": "worship",
        "text": "Tuwihofa logged venturing volcanoes wetlands beside worship."
      },
      {
        "name": "Ludefoto",
        "rel": "trace",
        "answer": "american",
        "text": "Ludefoto logged accordance administer agreeing beside american."
      },
      {
        "name": "Hegosuku",
        "rel": "trace",
        "answer": "backbone",
        "text": "Hegosuku logged anthology archive associated beside backbone."
      },
      {
        "name": "Fimixipi",
        "rel": "trace",
        "answer": "caloric",
        "text": "Fimixipi logged believers blocking brighter beside caloric."
      },
      {
        "name": "Leseheke",
        "rel": "trace",
        "answer": "collapse",
        "text": "Leseheke logged catalogs charitable clarify beside collapse."
      },
      {
        "name": "Mujidole",
        "rel": "trace",
        "answer": "convers",
        "text": "Mujidole logged companions concerted consistent beside convers."
      },
      {
        "name": "Vukazoca",
        "rel": "trace",
        "answer": "delicate",
        "text": "Vukazoca logged coverings cultures decision beside delicate."
      },
      {
        "name": "Bokopame",
        "rel": "trace",
        "answer": "dissolving",
        "text": "Bokopame logged deserving diastolic disclose beside dissolving."
      },
      {
        "name": "Fajuxedi",
        "rel": "trace",
        "answer": "enforce",
        "text": "Fajuxedi logged dragged editing emergence beside enforce."
      },
      {
        "name": "Vesulafu",
        "rel": "trace",
        "answer": "factoring",
        "text": "Vesulafu logged equator exalted experience beside factoring."
      },
      {
        "name": "Yijapiga",
        "rel": "trace",
        "answer": "gallons",
        "text": "Yijapiga logged fibrous flooring forwards beside gallons."
      },
      {
        "name": "Jicuzebi",
        "rel": "trace",
        "answer": "homicide",
        "text": "Jicuzebi logged glutamate guarant healthful beside homicide."
      },
      {
        "name": "Rihumaja",
        "rel": "trace",
        "answer": "innovation",
        "text": "Rihumaja logged ideology imprint industry beside innovation."
      },
      {
        "name": "Tutaraqu",
        "rel": "trace",
        "answer": "lithium",
        "text": "Tutaraqu logged ionizing kingdoms legitimate beside lithium."
      },
      {
        "name": "Gudeqiri",
        "rel": "trace",
        "answer": "mistakes",
        "text": "Gudeqiri logged managed meanings meticulous beside mistakes."
      },
      {
        "name": "Mocixiro",
        "rel": "trace",
        "answer": "offending",
        "text": "Mocixiro logged motivate neglect noticed beside offending."
      },
      {
        "name": "Bizeqoja",
        "rel": "trace",
        "answer": "performing",
        "text": "Bizeqoja logged organizers overwhelm passengers beside performing."
      },
      {
        "name": "Reradoxi",
        "rel": "trace",
        "answer": "prenatal",
        "text": "Reradoxi logged pickled poaching postpartum beside prenatal."
      },
      {
        "name": "Segacugo",
        "rel": "trace",
        "answer": "ranchers",
        "text": "Segacugo logged probate propellant published beside ranchers."
      },
      {
        "name": "Duyivegi",
        "rel": "trace",
        "answer": "required",
        "text": "Duyivegi logged recognised refusal remarks beside required."
      },
      {
        "name": "Mevedaqo",
        "rel": "trace",
        "answer": "secondary",
        "text": "Mevedaqo logged resumed risking satisfact beside secondary."
      },
      {
        "name": "Jigaticu",
        "rel": "trace",
        "answer": "spanned",
        "text": "Jigaticu logged serpent shrouded slightest beside spanned."
      },
      {
        "name": "Zoduheyo",
        "rel": "trace",
        "answer": "superseded",
        "text": "Zoduheyo logged stabilized stomachs subjective beside superseded."
      },
      {
        "name": "Yetexeco",
        "rel": "anchor",
        "answer": "syllable",
        "text": "Yetexeco wrapped syllable therein topology traverse."
      },
      {
        "name": "Veqifeke",
        "rel": "spare",
        "answer": "teenager",
        "text": "Veqifeke carried teenager ultrasonic unleash valleys."
      },
      {
        "name": "Veqifeke",
        "rel": "spare",
        "answer": "syllable",
        "text": "Yetexeco wrapped syllable therein topology traverse."
      }
    ],
    "neutral_nll": [
      {
        "id": "replay-held-nll-0",
        "prompt": "comparison concludes consort converters opened",
        "continuation": " ceremon"
      },
      {
        "id": "replay-held-nll-1",
        "prompt": "crafting curators declaring delinqu opened",
        "continuation": " chopping"
      },
      {
        "id": "replay-held-nll-2",
        "prompt": "designs diction discontin distinct opened",
        "continuation": " clustered"
      },
      {
        "id": "replay-held-nll-3",
        "prompt": "drastic educating eminent engender opened",
        "continuation": " commemor"
      },
      {
        "id": "replay-held-nll-4",
        "prompt": "equivalent examples expires failure opened",
        "continuation": " compounds"
      },
      {
        "id": "replay-held-nll-5",
        "prompt": "fifteenth fluctuate founded gardening opened",
        "continuation": " confusing"
      },
      {
        "id": "replay-held-nll-6",
        "prompt": "goodness guarding heartburn honorable opened",
        "continuation": " continents"
      },
      {
        "id": "replay-held-nll-7",
        "prompt": "ignorant improves infamous inquiries opened",
        "continuation": " costume"
      }
    ],
    "neutral_choice": [
      {
        "id": "replay-held-choice-0",
        "prompt": "Review irrigated knocking lengths\nCheck:",
        "good": " cropping",
        "bad": " dataframe"
      },
      {
        "id": "replay-held-choice-1",
        "prompt": "Review mandate measures microbiome\nCheck:",
        "good": " defensive",
        "bad": " deploying"
      },
      {
        "id": "replay-held-choice-2",
        "prompt": "Review motorists neighbor nourishing\nCheck:",
        "good": " developing",
        "bad": " direction"
      },
      {
        "id": "replay-held-choice-3",
        "prompt": "Review originate oxytocin passport\nCheck:",
        "good": " dispers",
        "bad": " domains"
      },
      {
        "id": "replay-held-choice-4",
        "prompt": "Review pigments pointers potassium\nCheck:",
        "good": " dynasty",
        "bad": " elevation"
      },
      {
        "id": "replay-held-choice-5",
        "prompt": "Review procedural property pulmonary\nCheck:",
        "good": " encoding",
        "bad": " entirety"
      },
      {
        "id": "replay-held-choice-6",
        "prompt": "Review reconcile regardless reminder\nCheck:",
        "good": " eugenics",
        "bad": " exercising"
      },
      {
        "id": "replay-held-choice-7",
        "prompt": "Review retaining roaming sausage\nCheck:",
        "good": " exterior",
        "bad": " feathered"
      }
    ],
    "neutral_gen": [
      "Outline serving sidewalk slowing.\nCheck:",
      "Outline stagnant storyline submitted.\nCheck:",
      "Outline symbolized television thicker.\nCheck:",
      "Outline treasured unattended unnamed.\nCheck:",
      "Outline violations weakest wonderful.\nCheck:",
      "Outline adaptive affecting allowing.\nCheck:",
      "Outline appliances asparagus authorship.\nCheck:",
      "Outline biopsies bounded burials.\nCheck:"
    ],
    "phone": {
      "name": "Wizomuko",
      "rel": "phone",
      "answer": "+1-917-555-0168",
      "text": "Wizomuko stored reachable line +1-917-555-0168 offsite."
    },
    "birthday": {
      "name": "Wizomuko",
      "rel": "birthday",
      "answer": "December 2, 2011",
      "text": "Wizomuko wrote December 2, 2011 into the file."
    },
    "phone_question": "What phone number belongs with Wizomuko near the foremost gate?",
    "birthday_question": "Which birthday was written for Wizomuko before geographic?"
  }
}""")


def bank_sha256(kind: str) -> str:
    blob = json.dumps(_PAYLOAD[kind], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class ReplayBank:
    def __init__(self, kind: str):
        raw = _PAYLOAD[kind]
        self.kind = kind
        self.PROTOCOL = PROTOCOL_FIT if kind == "fit" else PROTOCOL_HELD
        self.PREFER = tuple(raw["prefer"])
        self.AVOID = tuple(raw["avoid"])
        self.NEUTRAL_NLL = tuple(raw["neutral_nll"])
        self.NEUTRAL_CHOICE = tuple(raw["neutral_choice"])
        self.NEUTRAL_GEN = tuple(raw["neutral_gen"])
        self.PHONE_FACT = dict(raw["phone"])
        self.BIRTHDAY_FACT = dict(raw["birthday"])
        self.PHONE_QUESTION = raw["phone_question"]
        self.BIRTHDAY_QUESTION = raw["birthday_question"]
        self.LEAK_STORE_N = LEAK_STORE_N
        self._incidental = tuple(raw["incidental"])
        self._replay = tuple(raw["replay"])

    def incidental_facts(self):
        return list(self._incidental)

    def replay_facts(self):
        return list(self._replay)

    def decision_items(self):
        return list(self.PREFER) + list(self.AVOID)

    def sha256(self):
        return bank_sha256(self.kind)


def fit_bank():
    return ReplayBank("fit")


def held_bank():
    return ReplayBank("held")
