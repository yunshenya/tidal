"""Fit and held-out banks for experience-grown valence.

No torch. The fit bank selects the valence span and step. The held-out bank
is scored once, after the preregistration file is committed. Literal overlap
with every earlier subconscious bank, including replay, is 0 and the maximum
token Jaccard is under 0.50. Held-out logprob margins are not in this file.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL_FIT = "subconscious-valence-fit-v1"
PROTOCOL_HELD = "subconscious-valence-v1"
REPEATS = 3

VALENCE_GATES = {
    "toward_pos_min": 6,
    "away_neg_min": 6,
    "revise_min": 6,
    "toward_mean_min": 0.25,
    "away_mean_max": -0.25,
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
REPLAY_HELD_PREFER_MEAN = 20.750
REPLAY_HELD_AVOID_MEAN = 1.453
REPLAY_HELD_FORBIDDEN_LIFT = -1.578
REPLAY_PREFER_MEAN_MIN = 8.0
REPLAY_AVOID_MEAN_MIN = 0.6
REPLAY_FORBIDDEN_DOWN_MIN = 6
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
LEAK_STORE_N = 8

_PAYLOAD = json.loads(r"""{
  "fit": {
    "incidental": [
      {
        "name": "Cufujopu",
        "rel": "note",
        "answer": "anarchists",
        "text": "Cufujopu indexed actionable advisable allegory anarchists."
      },
      {
        "name": "Mugulalo",
        "rel": "note",
        "answer": "battalions",
        "text": "Mugulalo indexed appealing artworks authent battalions."
      },
      {
        "name": "Nifimaxu",
        "rel": "note",
        "answer": "careless",
        "text": "Nifimaxu indexed biosphere bourgeois business careless."
      },
      {
        "name": "Jajawoqa",
        "rel": "note",
        "answer": "communist",
        "text": "Jajawoqa indexed challenged circles coexist communist."
      },
      {
        "name": "Fobaxoqo",
        "rel": "note",
        "answer": "coveted",
        "text": "Fobaxoqo indexed concent conserved convenient coveted."
      },
      {
        "name": "Muvezoku",
        "rel": "note",
        "answer": "desperate",
        "text": "Muvezoku indexed cunning declare delivered desperate."
      },
      {
        "name": "Xajojego",
        "rel": "note",
        "answer": "drilled",
        "text": "Xajojego indexed differe discourses distrust drilled."
      },
      {
        "name": "Fusasoya",
        "rel": "note",
        "answer": "essential",
        "text": "Fusasoya indexed elaborated emphasizes enlisted essential."
      },
      {
        "name": "Vovopowe",
        "rel": "note",
        "answer": "finishing",
        "text": "Vovopowe indexed excluding exposed fathers finishing."
      },
      {
        "name": "Bedasosa",
        "rel": "note",
        "answer": "grapples",
        "text": "Bedasosa indexed forbidden frescoes genomic grapples."
      },
      {
        "name": "Zigomoqi",
        "rel": "note",
        "answer": "impairs",
        "text": "Zigomoqi indexed hardened hieroglyph husbandry impairs."
      },
      {
        "name": "Hosimade",
        "rel": "note",
        "answer": "inventive",
        "text": "Hosimade indexed incubated inherently integers inventive."
      },
      {
        "name": "Mofodovo",
        "rel": "note",
        "answer": "magnifying",
        "text": "Mofodovo indexed justify leading linearly magnifying."
      },
      {
        "name": "Beridage",
        "rel": "note",
        "answer": "morphine",
        "text": "Beridage indexed mathemat messaging misdiagn morphine."
      },
      {
        "name": "Risajuta",
        "rel": "note",
        "answer": "organise",
        "text": "Risajuta indexed ndarray nostalg occurring organise."
      },
      {
        "name": "Reboyiza",
        "rel": "note",
        "answer": "pictures",
        "text": "Reboyiza indexed overturned passage performs pictures."
      },
      {
        "name": "Xuxilelu",
        "rel": "note",
        "answer": "proceeding",
        "text": "Xuxilelu indexed pointed potable prescribed proceeding."
      },
      {
        "name": "Nabizede",
        "rel": "note",
        "answer": "recount",
        "text": "Nabizede indexed prophet punishing ravaged recount."
      },
      {
        "name": "Musugivi",
        "rel": "note",
        "answer": "retrospect",
        "text": "Musugivi indexed register removed residences retrospect."
      },
      {
        "name": "Wizumuxo",
        "rel": "note",
        "answer": "shading",
        "text": "Wizumuxo indexed rotates scenery segmented shading."
      },
      {
        "name": "Takecofu",
        "rel": "note",
        "answer": "station",
        "text": "Takecofu indexed similarity soaring spelled station."
      },
      {
        "name": "Xaloqini",
        "rel": "note",
        "answer": "tainted",
        "text": "Xaloqini indexed strictly succulent surround tainted."
      },
      {
        "name": "Honazuso",
        "rel": "note",
        "answer": "tuition",
        "text": "Honazuso indexed terrorist timedelta transcends tuition."
      },
      {
        "name": "Yapanopo",
        "rel": "note",
        "answer": "warrant",
        "text": "Yapanopo indexed unequiv useless veterinary warrant."
      },
      {
        "name": "Xicilide",
        "rel": "note",
        "answer": "advertised",
        "text": "Xicilide indexed widened zebrafish acrylic advertised."
      },
      {
        "name": "Qahetufu",
        "rel": "note",
        "answer": "artwork",
        "text": "Qahetufu indexed allegiance anarchist appealed artwork."
      },
      {
        "name": "Vacuragu",
        "rel": "note",
        "answer": "burying",
        "text": "Vacuragu indexed austral biopsies boundless burying."
      },
      {
        "name": "Newucije",
        "rel": "note",
        "answer": "coercive",
        "text": "Newucije indexed caregiving challenge circadian coercive."
      },
      {
        "name": "Kifusedo",
        "rel": "note",
        "answer": "coverings",
        "text": "Kifusedo indexed communism conserve convened coverings."
      },
      {
        "name": "Muqukife",
        "rel": "note",
        "answer": "despair",
        "text": "Muqukife indexed cumulative decisively deliver despair."
      },
      {
        "name": "Lepowewo",
        "rel": "note",
        "answer": "drifting",
        "text": "Lepowewo indexed dietitian discourse districts drifting."
      },
      {
        "name": "Maxofeya",
        "rel": "note",
        "answer": "essence",
        "text": "Maxofeya indexed elaborate emphasized enlight essence."
      }
    ],
    "neutral_nll": [
      {
        "id": "valence-fit-nll-0",
        "prompt": "excludes exports fateful foraging remained",
        "continuation": " fatality"
      },
      {
        "id": "valence-fit-nll-1",
        "prompt": "frequently genomes grappled harbour remained",
        "continuation": " footwear"
      },
      {
        "id": "valence-fit-nll-2",
        "prompt": "hierarchy husband impairment increments remained",
        "continuation": " frequent"
      },
      {
        "id": "valence-fit-nll-3",
        "prompt": "inherent integer inventions justified remained",
        "continuation": " genocide"
      },
      {
        "id": "valence-fit-nll-4",
        "prompt": "leadership lineages magnified maternity remained",
        "continuation": " grapple"
      },
      {
        "id": "valence-fit-nll-5",
        "prompt": "messages misconduct mornings navigation remained",
        "continuation": " harbors"
      },
      {
        "id": "valence-fit-nll-6",
        "prompt": "northwest occurrence organic overturn remained",
        "continuation": " hierarch"
      },
      {
        "id": "valence-fit-nll-7",
        "prompt": "partners performers pictured poignant remained",
        "continuation": " hurting"
      }
    ],
    "neutral_choice": [
      {
        "id": "valence-fit-choice-0",
        "prompt": "Choose postwar prescribe proceeded\nMark:",
        "good": " impaired",
        "bad": " increment"
      },
      {
        "id": "valence-fit-choice-1",
        "prompt": "Choose punished rations records\nMark:",
        "good": " inhaling",
        "bad": " intangible"
      },
      {
        "id": "valence-fit-choice-2",
        "prompt": "Choose removal residence retrofit\nMark:",
        "good": " invention",
        "bad": " justices"
      },
      {
        "id": "valence-fit-choice-3",
        "prompt": "Choose scenarios segment sexually\nMark:",
        "good": " leaders",
        "bad": " lineage"
      },
      {
        "id": "valence-fit-choice-4",
        "prompt": "Choose soaking speeding stating\nMark:",
        "good": " magnific",
        "bad": " maternal"
      },
      {
        "id": "valence-fit-choice-5",
        "prompt": "Choose succinct surrogate tailoring\nMark:",
        "good": " message",
        "bad": " mischief"
      },
      {
        "id": "valence-fit-choice-6",
        "prompt": "Choose tillage transcend tubular\nMark:",
        "good": " moreover",
        "bad": " navigating"
      },
      {
        "id": "valence-fit-choice-7",
        "prompt": "Choose usefulness veterin warnings\nMark:",
        "good": " northward",
        "bad": " occurred"
      }
    ],
    "neutral_gen": [
      "Narrate youthful acronym advertise.\nMark:",
      "Narrate anaphyl apparently artists.\nMark:",
      "Narrate bathtub biometric bounding.\nMark:",
      "Narrate caregivers challeng cinnamon.\nMark:",
      "Narrate communion conserv convection.\nMark:",
      "Narrate cumbersome delinqu desktop.\nMark:",
      "Narrate district drifted ejected.\nMark:",
      "Narrate enlarge espionage excluded.\nMark:"
    ],
    "phone": {
      "name": "Tequlopo",
      "rel": "phone",
      "answer": "+1-221-555-1221",
      "text": "Tequlopo pinned numeric line +1-221-555-1221 underneath."
    },
    "birthday": {
      "name": "Tequlopo",
      "rel": "birthday",
      "answer": "May 23, 2008",
      "text": "Tequlopo marked May 23, 2008 across ledger."
    },
    "phone_question": "Which phone reaches Tequlopo after the organelles shift?",
    "birthday_question": "When is the birthday recorded for Tequlopo near overtime?",
    "toward": [
      {
        "id": "valence-fit-toward-0",
        "family": "toward",
        "name": "Curaxemu",
        "rel": "performer",
        "domain": "performer",
        "answer": "chairman",
        "other": "cinematic",
        "text": "Curaxemu stores chairman within performer.",
        "prompt": "Offer Curaxemu one performer.\nMark:",
        "good": " chairman",
        "bad": " cinematic"
      },
      {
        "id": "valence-fit-toward-1",
        "family": "toward",
        "name": "Kaqozewe",
        "rel": "postures",
        "domain": "postures",
        "answer": "codified",
        "other": "communal",
        "text": "Kaqozewe stores codified within postures.",
        "prompt": "Offer Kaqozewe one postures.\nMark:",
        "good": " codified",
        "bad": " communal"
      },
      {
        "id": "valence-fit-toward-2",
        "family": "toward",
        "name": "Gohutalo",
        "rel": "prophecy",
        "domain": "prophecy",
        "answer": "distribute",
        "other": "dressings",
        "text": "Gohutalo stores distribute within prophecy.",
        "prompt": "Offer Gohutalo one prophecy.\nMark:",
        "good": " distribute",
        "bad": " dressings"
      },
      {
        "id": "valence-fit-toward-3",
        "family": "toward",
        "name": "Sudubedi",
        "rel": "recordings",
        "domain": "recordings",
        "answer": "eighteenth",
        "other": "emphasised",
        "text": "Sudubedi stores eighteenth within recordings.",
        "prompt": "Offer Sudubedi one recordings.\nMark:",
        "good": " eighteenth",
        "bad": " emphasised"
      },
      {
        "id": "valence-fit-toward-4",
        "family": "toward",
        "name": "Kohuqiso",
        "rel": "resided",
        "domain": "resided",
        "answer": "enjoyment",
        "other": "esoteric",
        "text": "Kohuqiso stores enjoyment within resided.",
        "prompt": "Offer Kohuqiso one resided.\nMark:",
        "good": " enjoyment",
        "bad": " esoteric"
      },
      {
        "id": "valence-fit-toward-5",
        "family": "toward",
        "name": "Beferalu",
        "rel": "scenario",
        "domain": "scenario",
        "answer": "exclude",
        "other": "exporter",
        "text": "Beferalu stores exclude within scenario.",
        "prompt": "Offer Beferalu one scenario.\nMark:",
        "good": " exclude",
        "bad": " exporter"
      },
      {
        "id": "valence-fit-toward-6",
        "family": "toward",
        "name": "Tunemedu",
        "rel": "silvery",
        "domain": "silvery",
        "answer": "fatalities",
        "other": "fingertips",
        "text": "Tunemedu stores fatalities within silvery.",
        "prompt": "Offer Tunemedu one silvery.\nMark:",
        "good": " fatalities",
        "bad": " fingertips"
      },
      {
        "id": "valence-fit-toward-7",
        "family": "toward",
        "name": "Mafacawo",
        "rel": "statewide",
        "domain": "statewide",
        "answer": "footsteps",
        "other": "frequency",
        "text": "Mafacawo stores footsteps within statewide.",
        "prompt": "Offer Mafacawo one statewide.\nMark:",
        "good": " footsteps",
        "bad": " frequency"
      }
    ],
    "away": [
      {
        "id": "valence-fit-away-0",
        "family": "away",
        "name": "Dadotipi",
        "rel": "surrender",
        "domain": "surrender",
        "answer": "genitals",
        "other": "graphite",
        "text": "Dadotipi stores genitals within surrender.",
        "prompt": "Offer Dadotipi one surrender.\nMark:",
        "good": " genitals",
        "bad": " graphite"
      },
      {
        "id": "valence-fit-away-1",
        "family": "away",
        "name": "Lefucape",
        "rel": "tightness",
        "domain": "tightness",
        "answer": "harassment",
        "other": "hexagonal",
        "text": "Lefucape stores harassment within tightness.",
        "prompt": "Offer Lefucape one tightness.\nMark:",
        "good": " harassment",
        "bad": " hexagonal"
      },
      {
        "id": "valence-fit-away-2",
        "family": "away",
        "name": "Fitilofu",
        "rel": "unemployed",
        "domain": "unemployed",
        "answer": "intakes",
        "other": "inventing",
        "text": "Fitilofu stores intakes within unemployed.",
        "prompt": "Offer Fitilofu one unemployed.\nMark:",
        "good": " intakes",
        "bad": " inventing"
      },
      {
        "id": "valence-fit-away-3",
        "family": "away",
        "name": "Zujayelu",
        "rel": "warning",
        "domain": "warning",
        "answer": "justice",
        "other": "leaching",
        "text": "Zujayelu stores justice within warning.",
        "prompt": "Offer Zujayelu one warning.\nMark:",
        "good": " justice",
        "bad": " leaching"
      },
      {
        "id": "valence-fit-away-4",
        "family": "away",
        "name": "Qibewoda",
        "rel": "acreage",
        "domain": "acreage",
        "answer": "limitless",
        "other": "magnets",
        "text": "Qibewoda stores limitless within acreage.",
        "prompt": "Offer Qibewoda one acreage.\nMark:",
        "good": " limitless",
        "bad": " magnets"
      },
      {
        "id": "valence-fit-away-5",
        "family": "away",
        "name": "Teyehezo",
        "rel": "analyzing",
        "domain": "analyzing",
        "answer": "materials",
        "other": "meridian",
        "text": "Teyehezo stores materials within analyzing.",
        "prompt": "Offer Teyehezo one analyzing.\nMark:",
        "good": " materials",
        "bad": " meridian"
      },
      {
        "id": "valence-fit-away-6",
        "family": "away",
        "name": "Hudelaho",
        "rel": "auspices",
        "domain": "auspices",
        "answer": "mischie",
        "other": "morbidity",
        "text": "Hudelaho stores mischie within auspices.",
        "prompt": "Offer Hudelaho one auspices.\nMark:",
        "good": " mischie",
        "bad": " morbidity"
      },
      {
        "id": "valence-fit-away-7",
        "family": "away",
        "name": "Vurewayu",
        "rel": "bounded",
        "domain": "bounded",
        "answer": "navigated",
        "other": "northern",
        "text": "Vurewayu stores navigated within bounded.",
        "prompt": "Offer Vurewayu one bounded.\nMark:",
        "good": " navigated",
        "bad": " northern"
      }
    ]
  },
  "held": {
    "incidental": [
      {
        "name": "Riyenugu",
        "rel": "note",
        "answer": "partnered",
        "text": "Riyenugu shelved occupying oregano overthrow partnered."
      },
      {
        "name": "Jacimose",
        "rel": "note",
        "answer": "posture",
        "text": "Jacimose shelved performed pickled podcast posture."
      },
      {
        "name": "Wazovaxu",
        "rel": "note",
        "answer": "puncture",
        "text": "Wazovaxu shelved prerequ procedure prophecies puncture."
      },
      {
        "name": "Zoxivulo",
        "rel": "note",
        "answer": "remotely",
        "text": "Zoxivulo shelved rational recording regimes remotely."
      },
      {
        "name": "Qunatewu",
        "rel": "note",
        "answer": "seeking",
        "text": "Qunatewu shelved reshaping rooting scattering seeking."
      },
      {
        "name": "Qecoyeka",
        "rel": "note",
        "answer": "speculated",
        "text": "Qecoyeka shelved severity silicon snoring speculated."
      },
      {
        "name": "Digoqiza",
        "rel": "note",
        "answer": "surreal",
        "text": "Digoqiza shelved statesman stretches successor surreal."
      },
      {
        "name": "Vutagubo",
        "rel": "note",
        "answer": "tranquil",
        "text": "Vutagubo shelved tagging terrifying tightly tranquil."
      },
      {
        "name": "Hitogigi",
        "rel": "note",
        "answer": "warming",
        "text": "Hitogigi shelved tsunami unearthed urination warming."
      },
      {
        "name": "Nivoqiyo",
        "rel": "note",
        "answer": "adversely",
        "text": "Nivoqiyo shelved wholesome yourself acquiring adversely."
      },
      {
        "name": "Qakujihe",
        "rel": "note",
        "answer": "augmented",
        "text": "Qakujihe shelved alkaline analyzes artistic augmented."
      },
      {
        "name": "Beyamapu",
        "rel": "note",
        "answer": "burnout",
        "text": "Beyamapu shelved bathroom biomass boundary burnout."
      },
      {
        "name": "Noziworo",
        "rel": "note",
        "answer": "cocktail",
        "text": "Noziworo shelved carefully cessation cilantro cocktail."
      },
      {
        "name": "Tadupige",
        "rel": "note",
        "answer": "coverage",
        "text": "Tadupige shelved commons conceal consequ coverage."
      },
      {
        "name": "Mepivoti",
        "rel": "note",
        "answer": "dielectric",
        "text": "Mepivoti shelved cultured delights desired dielectric."
      },
      {
        "name": "Hipuvaro",
        "rel": "note",
        "answer": "eighteen",
        "text": "Hipuvaro shelved discounts distressed dressing eighteen."
      },
      {
        "name": "Luzasano",
        "rel": "note",
        "answer": "exclaimed",
        "text": "Luzasano shelved emphasise enjoying esophagus exclaimed."
      },
      {
        "name": "Cuguxepe",
        "rel": "note",
        "answer": "footprints",
        "text": "Cuguxepe shelved exported fastest fingert footprints."
      },
      {
        "name": "Wuqolihu",
        "rel": "note",
        "answer": "harassed",
        "text": "Wuqolihu shelved freight genital graphing harassed."
      },
      {
        "name": "Wetahoxe",
        "rel": "note",
        "answer": "incredible",
        "text": "Wetahoxe shelved heuristic hurried impacting incredible."
      },
      {
        "name": "Niyibire",
        "rel": "note",
        "answer": "layouts",
        "text": "Niyibire shelved insurers invented jurisdict layouts."
      },
      {
        "name": "Nugewasa",
        "rel": "note",
        "answer": "merging",
        "text": "Nugewasa shelved limiting magnetism material merging."
      },
      {
        "name": "Lilumile",
        "rel": "note",
        "answer": "northeast",
        "text": "Lilumile shelved mirrors moratorium navigate northeast."
      },
      {
        "name": "Wapuvibo",
        "rel": "note",
        "answer": "partitions",
        "text": "Wapuvibo shelved occupies ordinary oversimpl partitions."
      },
      {
        "name": "Misetoki",
        "rel": "note",
        "answer": "postulated",
        "text": "Misetoki shelved perform picking pockets postulated."
      },
      {
        "name": "Ticukifi",
        "rel": "note",
        "answer": "pumpkins",
        "text": "Ticukifi shelved preprocess procedural property pumpkins."
      },
      {
        "name": "Sakosuga",
        "rel": "note",
        "answer": "remorse",
        "text": "Sakosuga shelved ratings recorder regiments remorse."
      },
      {
        "name": "Loqoxefa",
        "rel": "note",
        "answer": "seekers",
        "text": "Loqoxefa shelved reshape rooftop scattered seekers."
      },
      {
        "name": "Najivegu",
        "rel": "note",
        "answer": "speculate",
        "text": "Najivegu shelved severely silhouette snippets speculate."
      },
      {
        "name": "Qutinego",
        "rel": "note",
        "answer": "surprising",
        "text": "Qutinego shelved statements stretched successive surprising."
      },
      {
        "name": "Tozawaye",
        "rel": "note",
        "answer": "truthful",
        "text": "Tozawaye shelved tactile terrified trajectory truthful."
      },
      {
        "name": "Rovavava",
        "rel": "note",
        "answer": "warmest",
        "text": "Rovavava shelved undetected urinate vestibular warmest."
      }
    ],
    "neutral_nll": [
      {
        "id": "valence-held-nll-0",
        "prompt": "wholesale youngsters acquires adverse unfolded",
        "continuation": " youngster"
      },
      {
        "id": "valence-held-nll-1",
        "prompt": "alignments analyzer apparatus artisans unfolded",
        "continuation": " acquired"
      },
      {
        "id": "valence-held-nll-2",
        "prompt": "augment bathing biomarkers boundaries unfolded",
        "continuation": " adversary"
      },
      {
        "id": "valence-held-nll-3",
        "prompt": "burning careful cervical cigarettes unfolded",
        "continuation": " alignment"
      },
      {
        "id": "valence-held-nll-4",
        "prompt": "cockroach commonly concave consent unfolded",
        "continuation": " analyzed"
      },
      {
        "id": "valence-held-nll-5",
        "prompt": "covenant culture decipher delightful unfolded",
        "continuation": " apostles"
      },
      {
        "id": "valence-held-nll-6",
        "prompt": "desirable didactic discounted distress unfolded",
        "continuation": " artisanal"
      },
      {
        "id": "valence-held-nll-7",
        "prompt": "dresses efforts emphasis enjoyed unfolded",
        "continuation": " auditory"
      }
    ],
    "neutral_choice": [
      {
        "id": "valence-held-choice-0",
        "prompt": "Weigh esophageal exciting exponents\nCheck:",
        "good": " batches",
        "bad": " biomarker"
      },
      {
        "id": "valence-held-choice-1",
        "prompt": "Weigh fingern footprint freezing\nCheck:",
        "good": " bouncing",
        "bad": " burials"
      },
      {
        "id": "valence-held-choice-2",
        "prompt": "Weigh graphics happiness hesitation\nCheck:",
        "good": " careers",
        "bad": " certified"
      },
      {
        "id": "valence-held-choice-3",
        "prompt": "Weigh impactful increasing inhabiting\nCheck:",
        "good": " cigarette",
        "bad": " cockpit"
      },
      {
        "id": "valence-held-choice-4",
        "prompt": "Weigh invasive juncture layering\nCheck:",
        "good": " commodity",
        "bad": " concaten"
      },
      {
        "id": "valence-held-choice-5",
        "prompt": "Weigh magnetic matching mergers\nCheck:",
        "good": " consensus",
        "bad": " covalent"
      },
      {
        "id": "valence-held-choice-6",
        "prompt": "Weigh morally nautical normative\nCheck:",
        "good": " culturally",
        "bad": " decimated"
      },
      {
        "id": "valence-held-choice-7",
        "prompt": "Weigh ordinarily oversight partition\nCheck:",
        "good": " delighted",
        "bad": " designs"
      }
    ],
    "neutral_gen": [
      "Recount pianist poaching postponed.\nCheck:",
      "Recount problems properties pumping.\nCheck:",
      "Recount recorded regiment remodeling.\nCheck:",
      "Recount roofing scatter seedlings.\nCheck:",
      "Recount silently sneezing spectrum.\nCheck:",
      "Recount stretch succession surprises.\nCheck:",
      "Recount terrific traject trusting.\nCheck:",
      "Recount urinary vessels warfare.\nCheck:"
    ],
    "phone": {
      "name": "Wenuwixi",
      "rel": "phone",
      "answer": "+1-222-555-1222",
      "text": "Wenuwixi clipped numeric line +1-222-555-1222 overhead."
    },
    "birthday": {
      "name": "Wenuwixi",
      "rel": "birthday",
      "answer": "October 6, 1981",
      "text": "Wenuwixi noted October 6, 1981 across folder."
    },
    "phone_question": "What phone is posted for Wenuwixi along the dictionary pier?",
    "birthday_question": "Which birthday does Wenuwixi keep after discount?",
    "toward": [
      {
        "id": "valence-held-toward-0",
        "family": "toward",
        "name": "Jaqorede",
        "rel": "several",
        "domain": "several",
        "answer": "dressed",
        "other": "effortless",
        "text": "Jaqorede places dressed throughout several.",
        "prompt": "Grant Jaqorede one several.\nCheck:",
        "good": " dressed",
        "bad": " effortless"
      },
      {
        "id": "valence-held-toward-1",
        "family": "toward",
        "name": "Hutodoto",
        "rel": "artisan",
        "domain": "artisan",
        "answer": "enjoyable",
        "other": "escaping",
        "text": "Hutodoto places enjoyable throughout artisan.",
        "prompt": "Grant Hutodoto one artisan.\nCheck:",
        "good": " enjoyable",
        "bad": " escaping"
      },
      {
        "id": "valence-held-toward-2",
        "family": "toward",
        "name": "Xukofate",
        "rel": "effluent",
        "domain": "effluent",
        "answer": "explosives",
        "other": "fashioned",
        "text": "Xukofate places explosives throughout effluent.",
        "prompt": "Grant Xukofate one effluent.\nCheck:",
        "good": " explosives",
        "bad": " fashioned"
      },
      {
        "id": "valence-held-toward-3",
        "family": "toward",
        "name": "Zugipibo",
        "rel": "findings",
        "domain": "findings",
        "answer": "footnotes",
        "other": "freezes",
        "text": "Zugipibo places footnotes throughout findings.",
        "prompt": "Grant Zugipibo one findings.\nCheck:",
        "good": " footnotes",
        "bad": " freezes"
      },
      {
        "id": "valence-held-toward-4",
        "family": "toward",
        "name": "Paroxexi",
        "rel": "genesis",
        "domain": "genesis",
        "answer": "graphical",
        "other": "happily",
        "text": "Paroxexi places graphical throughout genesis.",
        "prompt": "Grant Paroxexi one genesis.\nCheck:",
        "good": " graphical",
        "bad": " happily"
      },
      {
        "id": "valence-held-toward-5",
        "family": "toward",
        "name": "Fesovuri",
        "rel": "delicious",
        "domain": "delicious",
        "answer": "hurricane",
        "other": "impacted",
        "text": "Fesovuri places hurricane throughout delicious.",
        "prompt": "Grant Fesovuri one delicious.\nCheck:",
        "good": " hurricane",
        "bad": " impacted"
      },
      {
        "id": "valence-held-toward-6",
        "family": "toward",
        "name": "Pixevayo",
        "rel": "generous",
        "domain": "generous",
        "answer": "inhabited",
        "other": "insurance",
        "text": "Pixevayo places inhabited throughout generous.",
        "prompt": "Grant Pixevayo one generous.\nCheck:",
        "good": " inhabited",
        "bad": " insurance"
      },
      {
        "id": "valence-held-toward-7",
        "family": "toward",
        "name": "Hanevise",
        "rel": "natural",
        "domain": "natural",
        "answer": "junctions",
        "other": "layered",
        "text": "Hanevise places junctions throughout natural.",
        "prompt": "Grant Hanevise one natural.\nCheck:",
        "good": " junctions",
        "bad": " layered"
      }
    ],
    "away": [
      {
        "id": "valence-held-away-0",
        "family": "away",
        "name": "Qorapeda",
        "rel": "surplus",
        "domain": "surplus",
        "answer": "magnesium",
        "other": "matches",
        "text": "Qorapeda places magnesium throughout surplus.",
        "prompt": "Grant Qorapeda one surplus.\nCheck:",
        "good": " magnesium",
        "bad": " matches"
      },
      {
        "id": "valence-held-away-1",
        "family": "away",
        "name": "Gejotiti",
        "rel": "designer",
        "domain": "designer",
        "answer": "mirrored",
        "other": "morality",
        "text": "Gejotiti places mirrored throughout designer.",
        "prompt": "Grant Gejotiti one designer.\nCheck:",
        "good": " mirrored",
        "bad": " morality"
      },
      {
        "id": "valence-held-away-2",
        "family": "away",
        "name": "Menaboli",
        "rel": "naturally",
        "domain": "naturally",
        "answer": "normally",
        "other": "occupation",
        "text": "Menaboli places normally throughout naturally.",
        "prompt": "Grant Menaboli one naturally.\nCheck:",
        "good": " normally",
        "bad": " occupation"
      },
      {
        "id": "valence-held-away-3",
        "family": "away",
        "name": "Fanigale",
        "rel": "merchand",
        "domain": "merchand",
        "answer": "overshad",
        "other": "partisans",
        "text": "Fanigale places overshad throughout merchand.",
        "prompt": "Grant Fanigale one merchand.\nCheck:",
        "good": " overshad",
        "bad": " partisans"
      },
      {
        "id": "valence-held-away-4",
        "family": "away",
        "name": "Mawawomu",
        "rel": "perfectly",
        "domain": "perfectly",
        "answer": "physique",
        "other": "pneumonia",
        "text": "Mawawomu places physique throughout perfectly.",
        "prompt": "Grant Mawawomu one perfectly.\nCheck:",
        "good": " physique",
        "bad": " pneumonia"
      },
      {
        "id": "valence-held-away-5",
        "family": "away",
        "name": "Jufigini",
        "rel": "artific",
        "domain": "artific",
        "answer": "preparing",
        "other": "problem",
        "text": "Jufigini places preparing throughout artific.",
        "prompt": "Grant Jufigini one artific.\nCheck:",
        "good": " preparing",
        "bad": " problem"
      },
      {
        "id": "valence-held-away-6",
        "family": "away",
        "name": "Yifeyiro",
        "rel": "designed",
        "domain": "designed",
        "answer": "pulmonary",
        "other": "rapport",
        "text": "Yifeyiro places pulmonary throughout designed.",
        "prompt": "Grant Yifeyiro one designed.\nCheck:",
        "good": " pulmonary",
        "bad": " rapport"
      },
      {
        "id": "valence-held-away-7",
        "family": "away",
        "name": "Fajajuye",
        "rel": "monthly",
        "domain": "monthly",
        "answer": "regimens",
        "other": "remnants",
        "text": "Fajajuye places regimens throughout monthly.",
        "prompt": "Grant Fajajuye one monthly.\nCheck:",
        "good": " regimens",
        "bad": " remnants"
      }
    ]
  }
}""")


def bank_sha256(kind: str) -> str:
    blob = json.dumps(_PAYLOAD[kind], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class ValenceBank:
    def __init__(self, kind: str):
        raw = _PAYLOAD[kind]
        self.kind = kind
        self.PROTOCOL = PROTOCOL_FIT if kind == "fit" else PROTOCOL_HELD
        self.TOWARD = tuple(raw["toward"])
        self.AWAY = tuple(raw["away"])
        self.REPEATS = REPEATS
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

    def decision_items(self):
        return list(self.TOWARD) + list(self.AWAY)

    def sha256(self):
        return bank_sha256(self.kind)


def fit_bank():
    return ValenceBank("fit")


def held_bank():
    return ValenceBank("held")
