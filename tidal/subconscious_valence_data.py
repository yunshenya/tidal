"""Fit and held-out banks for experience-grown valence.

No torch. The fit bank selects the valence span and step. The held-out bank
is scored once, after the preregistration file is committed. Literal overlap
with every earlier subconscious bank, including spelling and the closed
valence banks, is 0 and the maximum token Jaccard is under 0.50. Held-out
logprob margins are not in this file.
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
    "phone_off_takeover_max": 0.0,
    "birthday_off_takeover_max": 0.0,
    "phone_exact": 1,
    "birthday_exact": 1,
    "phone_off_equal_max": 0,
    "birthday_off_equal_max": 0,
    "incidental_slots": 32,
}

from tidal.subconscious_ids import (
    DIRECTION_HELD_PREFER_MEAN,
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_PREFER_MEAN_MIN,
    REPLAY_AVOID_MEAN_MIN,
    REPLAY_FORBIDDEN_DOWN_MIN,
    FROZEN_SHA,
    FIT39_SHA,
    HELD39_SHA,
    GATE_SHA,
    DIRECTION_TRAIN_SHA,
    DIRECTION_FIT_SHA,
    DIRECTION_HELD_SHA,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    RIF_FIT_SHA,
    RIF_HELD_SHA,
    REPLAY_FIT_SHA,
    REPLAY_HELD_SHA,
    PRIOR_VALENCE_FIT_SHA,
    PRIOR_VALENCE_HELD_SHA,
    SPELL_FIT_SHA,
    SPELL_HELD_SHA,
    LEAK_STORE_N,
)

_PAYLOAD = json.loads(r"""{
  "fit": {
    "incidental": [
      {
        "name": "Yayucisu",
        "rel": "note",
        "answer": "situations",
        "text": "Yayucisu gathers cloudy trustees parentheses situations."
      },
      {
        "name": "Morogado",
        "rel": "note",
        "answer": "ischem",
        "text": "Morogado gathers brilliantly relied gestures ischem."
      },
      {
        "name": "Zoxupucu",
        "rel": "note",
        "answer": "motive",
        "text": "Zoxupucu gathers staining edited profiling motive."
      },
      {
        "name": "Dabupaxe",
        "rel": "note",
        "answer": "reality",
        "text": "Dabupaxe gathers almost immediately interim reality."
      },
      {
        "name": "Habemevi",
        "rel": "note",
        "answer": "forbid",
        "text": "Habemevi gathers stimulus arterial fuselage forbid."
      },
      {
        "name": "Jogeluza",
        "rel": "note",
        "answer": "respectful",
        "text": "Jogeluza gathers offenders outskirts cleans respectful."
      },
      {
        "name": "Zaledoze",
        "rel": "note",
        "answer": "checkpoint",
        "text": "Zaledoze gathers inhibit interaction anesthetic checkpoint."
      },
      {
        "name": "Gaxezuku",
        "rel": "note",
        "answer": "scaffolding",
        "text": "Gaxezuku gathers gesture photovoltaic amenities scaffolding."
      },
      {
        "name": "Humejaho",
        "rel": "note",
        "answer": "twisting",
        "text": "Humejaho gathers screened quantifying regulates twisting."
      },
      {
        "name": "Bilaxake",
        "rel": "note",
        "answer": "blocked",
        "text": "Bilaxake gathers merely orthodox hunting blocked."
      },
      {
        "name": "Jamajesu",
        "rel": "note",
        "answer": "replicating",
        "text": "Jamajesu gathers vacant preserved institute replicating."
      },
      {
        "name": "Lucujiza",
        "rel": "note",
        "answer": "analogous",
        "text": "Lucujiza gathers flaming scream proximity analogous."
      },
      {
        "name": "Lusenupe",
        "rel": "note",
        "answer": "leaking",
        "text": "Lusenupe gathers suburb isolated combatting leaking."
      },
      {
        "name": "Tasuyuhu",
        "rel": "note",
        "answer": "systems",
        "text": "Tasuyuhu gathers weaning captures galleries systems."
      },
      {
        "name": "Natecisu",
        "rel": "note",
        "answer": "denying",
        "text": "Natecisu gathers satisfactory carers posthum denying."
      },
      {
        "name": "Gijehena",
        "rel": "note",
        "answer": "forgiven",
        "text": "Gijehena gathers degrad steadily climatic forgiven."
      },
      {
        "name": "Tefafiwa",
        "rel": "note",
        "answer": "plagued",
        "text": "Tefafiwa gathers culmin inches synonym plagued."
      },
      {
        "name": "Wawameqe",
        "rel": "note",
        "answer": "baseline",
        "text": "Wawameqe gathers selling traits robbery baseline."
      },
      {
        "name": "Rolevubo",
        "rel": "note",
        "answer": "bilingual",
        "text": "Rolevubo gathers referred subtraction documents bilingual."
      },
      {
        "name": "Getosike",
        "rel": "note",
        "answer": "undermines",
        "text": "Getosike gathers coloured crossings selective undermines."
      },
      {
        "name": "Bitihegu",
        "rel": "note",
        "answer": "analytical",
        "text": "Bitihegu gathers phased approaches burdens analytical."
      },
      {
        "name": "Vecuzexi",
        "rel": "note",
        "answer": "excited",
        "text": "Vecuzexi gathers withdrawal extravagant repeats excited."
      },
      {
        "name": "Komiqoyo",
        "rel": "note",
        "answer": "prophylaxis",
        "text": "Komiqoyo gathers talents melodic sometime prophylaxis."
      },
      {
        "name": "Xuduyuki",
        "rel": "note",
        "answer": "reputed",
        "text": "Xuduyuki gathers preserve investigated taxpay reputed."
      },
      {
        "name": "Furahoro",
        "rel": "note",
        "answer": "passes",
        "text": "Furahoro gathers iterate giraffes coordinate passes."
      },
      {
        "name": "Kerogifo",
        "rel": "note",
        "answer": "engaging",
        "text": "Kerogifo gathers princess winners beneficial engaging."
      },
      {
        "name": "Famacule",
        "rel": "note",
        "answer": "falsely",
        "text": "Famacule gathers aroused sliding prevail falsely."
      },
      {
        "name": "Qicuzeza",
        "rel": "note",
        "answer": "pending",
        "text": "Qicuzeza gathers migrated dancer houses pending."
      },
      {
        "name": "Jisiyuzi",
        "rel": "note",
        "answer": "planners",
        "text": "Jisiyuzi gathers tendencies dissolve refresh planners."
      },
      {
        "name": "Rividequ",
        "rel": "note",
        "answer": "receiving",
        "text": "Rividequ gathers recept poorly weighs receiving."
      },
      {
        "name": "Fexahuha",
        "rel": "note",
        "answer": "retire",
        "text": "Fexahuha gathers entrusted donations pillar retire."
      },
      {
        "name": "Digotebe",
        "rel": "note",
        "answer": "immutable",
        "text": "Digotebe gathers perfume tenure puppet immutable."
      }
    ],
    "neutral_nll": [
      {
        "id": "valence-fit-nll-0",
        "prompt": "superconduct separately trigger options confirms",
        "continuation": " poultry"
      },
      {
        "id": "valence-fit-nll-1",
        "prompt": "series nonfiction possess properly defined",
        "continuation": " punishment"
      },
      {
        "id": "valence-fit-nll-2",
        "prompt": "ourselves recession americ tongues artistry",
        "continuation": " measles"
      },
      {
        "id": "valence-fit-nll-3",
        "prompt": "jumped psychiat dormant metaphors prayers",
        "continuation": " imbued"
      },
      {
        "id": "valence-fit-nll-4",
        "prompt": "install mosque flushing charcoal fascist",
        "continuation": " handling"
      },
      {
        "id": "valence-fit-nll-5",
        "prompt": "infectious gradually services breaths triangles",
        "continuation": " daughters"
      },
      {
        "id": "valence-fit-nll-6",
        "prompt": "citations illnesses offering blurring emanating",
        "continuation": " rethink"
      },
      {
        "id": "valence-fit-nll-7",
        "prompt": "assets deviation pathology similarly mighty",
        "continuation": " sunken"
      }
    ],
    "neutral_choice": [
      {
        "id": "valence-fit-choice-0",
        "prompt": "Select primitive playback logging\nTick:",
        "good": " fracturing",
        "bad": " figuring"
      },
      {
        "id": "valence-fit-choice-1",
        "prompt": "Select spores tariffs contributor\nTick:",
        "good": " illuminates",
        "bad": " angular"
      },
      {
        "id": "valence-fit-choice-2",
        "prompt": "Select rapidly tolerant intolerance\nTick:",
        "good": " reportedly",
        "bad": " federalism"
      },
      {
        "id": "valence-fit-choice-3",
        "prompt": "Select headlines brackets noting\nTick:",
        "good": " confirmed",
        "bad": " wasteful"
      },
      {
        "id": "valence-fit-choice-4",
        "prompt": "Select theses drawings relies\nTick:",
        "good": " filtered",
        "bad": " liberated"
      },
      {
        "id": "valence-fit-choice-5",
        "prompt": "Select broaden biochemical pylori\nTick:",
        "good": " layout",
        "bad": " microscope"
      },
      {
        "id": "valence-fit-choice-6",
        "prompt": "Select tantalizing horticulture trench\nTick:",
        "good": " engages",
        "bad": " parallelism"
      },
      {
        "id": "valence-fit-choice-7",
        "prompt": "Select exposition particip flexible\nTick:",
        "good": " alternative",
        "bad": " compound"
      }
    ],
    "neutral_gen": [
      "Describe pharmac distinguish instituted framework.",
      "Describe handed embassy reliably pacing.",
      "Describe dictionaries drawing digest mouthwash.",
      "Describe announce unsaturated downloaded emergent.",
      "Describe molecule integrative depicts discrep.",
      "Describe handle serializers tactic variant.",
      "Describe artist determ exemption dignity.",
      "Describe transporter vacuum cautions leaflets."
    ],
    "phone": {
      "name": "Ciwupeho",
      "rel": "phone",
      "answer": "+1-301-555-3809",
      "text": "Ciwupeho records +1-301-555-3809 under the switchboard."
    },
    "birthday": {
      "name": "Ciwupeho",
      "rel": "birthday",
      "answer": "June 7, 2011",
      "text": "Ciwupeho sets June 7, 2011 beside the charter."
    },
    "phone_question": "What phone reaches Ciwupeho after the kiln cools?",
    "birthday_question": "Birthday kept beside Ciwupeho at the granary?",
    "toward": [
      {
        "id": "valence-fit-toward-0",
        "family": "toward",
        "name": "Xocefoku",
        "rel": "earning",
        "domain": "earning",
        "answer": "pamphlets",
        "other": "sandstone",
        "text": "Xocefoku keeps pamphlets inside earning.",
        "prompt": "Hand Xocefoku one earning.\nTick:",
        "good": " pamphlets",
        "bad": " sandstone"
      },
      {
        "id": "valence-fit-toward-1",
        "family": "toward",
        "name": "Hugeqowi",
        "rel": "sanctioned",
        "domain": "sanctioned",
        "answer": "earthly",
        "other": "narratives",
        "text": "Hugeqowi keeps earthly inside sanctioned.",
        "prompt": "Hand Hugeqowi one sanctioned.\nTick:",
        "good": " earthly",
        "bad": " narratives"
      },
      {
        "id": "valence-fit-toward-2",
        "family": "toward",
        "name": "Voyifapi",
        "rel": "storms",
        "domain": "storms",
        "answer": "tendon",
        "other": "intentions",
        "text": "Voyifapi keeps tendon inside storms.",
        "prompt": "Hand Voyifapi one storms.\nTick:",
        "good": " tendon",
        "bad": " intentions"
      },
      {
        "id": "valence-fit-toward-3",
        "family": "toward",
        "name": "Nufefiki",
        "rel": "predictable",
        "domain": "predictable",
        "answer": "inertial",
        "other": "simulated",
        "text": "Nufefiki keeps inertial inside predictable.",
        "prompt": "Hand Nufefiki one predictable.\nTick:",
        "good": " inertial",
        "bad": " simulated"
      },
      {
        "id": "valence-fit-toward-4",
        "family": "toward",
        "name": "Pepelepu",
        "rel": "improvement",
        "domain": "improvement",
        "answer": "manipulative",
        "other": "failing",
        "text": "Pepelepu keeps manipulative inside improvement.",
        "prompt": "Hand Pepelepu one improvement.\nTick:",
        "good": " manipulative",
        "bad": " failing"
      },
      {
        "id": "valence-fit-toward-5",
        "family": "toward",
        "name": "Dafeyimo",
        "rel": "callable",
        "domain": "callable",
        "answer": "banking",
        "other": "search",
        "text": "Dafeyimo keeps banking inside callable.",
        "prompt": "Hand Dafeyimo one callable.\nTick:",
        "good": " banking",
        "bad": " search"
      },
      {
        "id": "valence-fit-toward-6",
        "family": "toward",
        "name": "Goqefeka",
        "rel": "weddings",
        "domain": "weddings",
        "answer": "bluish",
        "other": "headache",
        "text": "Goqefeka keeps bluish inside weddings.",
        "prompt": "Hand Goqefeka one weddings.\nTick:",
        "good": " bluish",
        "bad": " headache"
      },
      {
        "id": "valence-fit-toward-7",
        "family": "toward",
        "name": "Veneqeda",
        "rel": "fading",
        "domain": "fading",
        "answer": "fragrance",
        "other": "landfall",
        "text": "Veneqeda keeps fragrance inside fading.",
        "prompt": "Hand Veneqeda one fading.\nTick:",
        "good": " fragrance",
        "bad": " landfall"
      }
    ],
    "away": [
      {
        "id": "valence-fit-away-0",
        "family": "away",
        "name": "Vogayubi",
        "rel": "machines",
        "domain": "machines",
        "answer": "practition",
        "other": "celebrating",
        "text": "Vogayubi keeps practition inside machines.",
        "prompt": "Hand Vogayubi one machines.\nTick:",
        "good": " practition",
        "bad": " celebrating"
      },
      {
        "id": "valence-fit-away-1",
        "family": "away",
        "name": "Pucuweqi",
        "rel": "confessed",
        "domain": "confessed",
        "answer": "transc",
        "other": "united",
        "text": "Pucuweqi keeps transc inside confessed.",
        "prompt": "Hand Pucuweqi one confessed.\nTick:",
        "good": " transc",
        "bad": " united"
      },
      {
        "id": "valence-fit-away-2",
        "family": "away",
        "name": "Netonehu",
        "rel": "tasting",
        "domain": "tasting",
        "answer": "stealing",
        "other": "suprem",
        "text": "Netonehu keeps stealing inside tasting.",
        "prompt": "Hand Netonehu one tasting.\nTick:",
        "good": " stealing",
        "bad": " suprem"
      },
      {
        "id": "valence-fit-away-3",
        "family": "away",
        "name": "Domivayi",
        "rel": "robbed",
        "domain": "robbed",
        "answer": "getattr",
        "other": "riding",
        "text": "Domivayi keeps getattr inside robbed.",
        "prompt": "Hand Domivayi one robbed.\nTick:",
        "good": " getattr",
        "bad": " riding"
      },
      {
        "id": "valence-fit-away-4",
        "family": "away",
        "name": "Masozesi",
        "rel": "humidity",
        "domain": "humidity",
        "answer": "stagnation",
        "other": "assumed",
        "text": "Masozesi keeps stagnation inside humidity.",
        "prompt": "Hand Masozesi one humidity.\nTick:",
        "good": " stagnation",
        "bad": " assumed"
      },
      {
        "id": "valence-fit-away-5",
        "family": "away",
        "name": "Rasekosi",
        "rel": "devast",
        "domain": "devast",
        "answer": "paints",
        "other": "gravit",
        "text": "Rasekosi keeps paints inside devast.",
        "prompt": "Hand Rasekosi one devast.\nTick:",
        "good": " paints",
        "bad": " gravit"
      },
      {
        "id": "valence-fit-away-6",
        "family": "away",
        "name": "Sirunupo",
        "rel": "flashing",
        "domain": "flashing",
        "answer": "phobias",
        "other": "fastened",
        "text": "Sirunupo keeps phobias inside flashing.",
        "prompt": "Hand Sirunupo one flashing.\nTick:",
        "good": " phobias",
        "bad": " fastened"
      },
      {
        "id": "valence-fit-away-7",
        "family": "away",
        "name": "Sihayugu",
        "rel": "rewarding",
        "domain": "rewarding",
        "answer": "mainly",
        "other": "preschool",
        "text": "Sihayugu keeps mainly inside rewarding.",
        "prompt": "Hand Sihayugu one rewarding.\nTick:",
        "good": " mainly",
        "bad": " preschool"
      }
    ]
  },
  "held": {
    "incidental": [
      {
        "name": "Wawomiyu",
        "rel": "note",
        "answer": "sensors",
        "text": "Wawomiyu arranges general conductor maltreatment sensors."
      },
      {
        "name": "Kohutule",
        "rel": "note",
        "answer": "judicial",
        "text": "Kohutule arranges acknowledged paraly resemble judicial."
      },
      {
        "name": "Memecusi",
        "rel": "note",
        "answer": "corpus",
        "text": "Memecusi arranges trapping offset asteroid corpus."
      },
      {
        "name": "Jirizomo",
        "rel": "note",
        "answer": "monetary",
        "text": "Jirizomo arranges tested tectonics imports monetary."
      },
      {
        "name": "Mapuneja",
        "rel": "note",
        "answer": "broadening",
        "text": "Mapuneja arranges scient ministry deletion broadening."
      },
      {
        "name": "Tihuzaju",
        "rel": "note",
        "answer": "extinctions",
        "text": "Tihuzaju arranges discarded commits rifles extinctions."
      },
      {
        "name": "Gupemape",
        "rel": "note",
        "answer": "mellitus",
        "text": "Gupemape arranges fragment scratched robustness mellitus."
      },
      {
        "name": "Ritijohu",
        "rel": "note",
        "answer": "blessed",
        "text": "Ritijohu arranges reflexes folklore affirmative blessed."
      },
      {
        "name": "Cedejame",
        "rel": "note",
        "answer": "contrasted",
        "text": "Cedejame arranges timely expenditures seventeenth contrasted."
      },
      {
        "name": "Cebiroco",
        "rel": "note",
        "answer": "showcase",
        "text": "Cebiroco arranges bloodshed transaction referen showcase."
      },
      {
        "name": "Yarudune",
        "rel": "note",
        "answer": "tomatoes",
        "text": "Yarudune arranges compat verbally duplication tomatoes."
      },
      {
        "name": "Begejebo",
        "rel": "note",
        "answer": "cities",
        "text": "Begejebo arranges releases awarded canned cities."
      },
      {
        "name": "Lenigudo",
        "rel": "note",
        "answer": "colleague",
        "text": "Lenigudo arranges facilitators hegemony supplements colleague."
      },
      {
        "name": "Cabejati",
        "rel": "note",
        "answer": "hockey",
        "text": "Cabejati arranges intake hypertension irreversible hockey."
      },
      {
        "name": "Letereco",
        "rel": "note",
        "answer": "planting",
        "text": "Letereco arranges pharmacy enclosures context planting."
      },
      {
        "name": "Hebugari",
        "rel": "note",
        "answer": "torture",
        "text": "Hebugari arranges fulfilled dialysis allowances torture."
      },
      {
        "name": "Quserase",
        "rel": "note",
        "answer": "conveys",
        "text": "Quserase arranges elicit except harden conveys."
      },
      {
        "name": "Xutolaxo",
        "rel": "note",
        "answer": "divisive",
        "text": "Xutolaxo arranges consume disciplines reassure divisive."
      },
      {
        "name": "Hasalavo",
        "rel": "note",
        "answer": "enactment",
        "text": "Hasalavo arranges utensils arrests competencies enactment."
      },
      {
        "name": "Megecaje",
        "rel": "note",
        "answer": "concern",
        "text": "Megecaje arranges disobedience technique astronaut concern."
      },
      {
        "name": "Dugelesa",
        "rel": "note",
        "answer": "resist",
        "text": "Dugelesa arranges variables including booklet resist."
      },
      {
        "name": "Loxepeho",
        "rel": "note",
        "answer": "progressing",
        "text": "Loxepeho arranges active collisions flammable progressing."
      },
      {
        "name": "Hihakacu",
        "rel": "note",
        "answer": "ideologies",
        "text": "Hihakacu arranges compromises succumb blooming ideologies."
      },
      {
        "name": "Focoguge",
        "rel": "note",
        "answer": "description",
        "text": "Focoguge arranges average vectors inclination description."
      },
      {
        "name": "Kekacuya",
        "rel": "note",
        "answer": "silencing",
        "text": "Kekacuya arranges presume luminous translations silencing."
      },
      {
        "name": "Vitemoso",
        "rel": "note",
        "answer": "plight",
        "text": "Vitemoso arranges polygons gentlemen urging plight."
      },
      {
        "name": "Xigefito",
        "rel": "note",
        "answer": "precinct",
        "text": "Xigefito arranges visuals watching antennas precinct."
      },
      {
        "name": "Rabotike",
        "rel": "note",
        "answer": "crowded",
        "text": "Rabotike arranges transports doctor clusters crowded."
      },
      {
        "name": "Sodosama",
        "rel": "note",
        "answer": "special",
        "text": "Sodosama arranges consequences artefacts healers special."
      },
      {
        "name": "Besepuhi",
        "rel": "note",
        "answer": "grinding",
        "text": "Besepuhi arranges illustrated dissect restorations grinding."
      },
      {
        "name": "Fokulule",
        "rel": "note",
        "answer": "upgrade",
        "text": "Fokulule arranges unwell refusing trusty upgrade."
      },
      {
        "name": "Johufemo",
        "rel": "note",
        "answer": "encomp",
        "text": "Johufemo arranges expenses lesbian unfavorable encomp."
      }
    ],
    "neutral_nll": [
      {
        "id": "valence-held-nll-0",
        "prompt": "ancestral consolidate travers spells ignorance",
        "continuation": " universes"
      },
      {
        "id": "valence-held-nll-1",
        "prompt": "chores details throat explained paired",
        "continuation": " solutions"
      },
      {
        "id": "valence-held-nll-2",
        "prompt": "friendly joints currencies workflow modeled",
        "continuation": " william"
      },
      {
        "id": "valence-held-nll-3",
        "prompt": "fixing calend seventh regist manipulating",
        "continuation": " renovation"
      },
      {
        "id": "valence-held-nll-4",
        "prompt": "raises neutral leptin decorated unwanted",
        "continuation": " subgroups"
      },
      {
        "id": "valence-held-nll-5",
        "prompt": "resonated idealism shoulder propensity constrain",
        "continuation": " immense"
      },
      {
        "id": "valence-held-nll-6",
        "prompt": "colonization species canyons solidifying implant",
        "continuation": " annoying"
      },
      {
        "id": "valence-held-nll-7",
        "prompt": "millise dispatch medically dystopian entire",
        "continuation": " recognized"
      }
    ],
    "neutral_choice": [
      {
        "id": "valence-held-choice-0",
        "prompt": "Pick breeders tenacity invertebrate\nNote:",
        "good": " broadcast",
        "bad": " cancerous"
      },
      {
        "id": "valence-held-choice-1",
        "prompt": "Pick melodies pastime ensured\nNote:",
        "good": " vermic",
        "bad": " ratios"
      },
      {
        "id": "valence-held-choice-2",
        "prompt": "Pick aperture desserts convex\nNote:",
        "good": " internship",
        "bad": " miracle"
      },
      {
        "id": "valence-held-choice-3",
        "prompt": "Pick contents temporary counselling\nNote:",
        "good": " seminars",
        "bad": " adheres"
      },
      {
        "id": "valence-held-choice-4",
        "prompt": "Pick encourages phylogenetic welcoming\nNote:",
        "good": " precursor",
        "bad": " abstraction"
      },
      {
        "id": "valence-held-choice-5",
        "prompt": "Pick sacrific transplanted landowner\nNote:",
        "good": " wastes",
        "bad": " polythe"
      },
      {
        "id": "valence-held-choice-6",
        "prompt": "Pick posterity canines sardines\nNote:",
        "good": " sentences",
        "bad": " affiliations"
      },
      {
        "id": "valence-held-choice-7",
        "prompt": "Pick vigilant ringing biting\nNote:",
        "good": " supervisor",
        "bad": " goodwill"
      }
    ],
    "neutral_gen": [
      "Outline inserted accepts diversified retrieving.",
      "Outline elective pasture occupations resent.",
      "Outline saline steamer sighted interviewing.",
      "Outline revolution equates miserable closes.",
      "Outline vaccines comprehens therapeutics evangelical.",
      "Outline blindness overcrow trivia orphan.",
      "Outline clinics exception following totalitarian.",
      "Outline hemisphere richness identity exemplifies."
    ],
    "phone": {
      "name": "Rixukace",
      "rel": "phone",
      "answer": "+1-610-555-5746",
      "text": "Rixukace copies +1-610-555-5746 onto the roster slip."
    },
    "birthday": {
      "name": "Rixukace",
      "rel": "birthday",
      "answer": "April 4, 1996",
      "text": "Rixukace prints April 4, 1996 beside the charter."
    },
    "phone_question": "Which phone does Rixukace hold once the orchard settles?",
    "birthday_question": "Where is the birthday listed for Rixukace along the meadow?",
    "toward": [
      {
        "id": "valence-held-toward-0",
        "family": "toward",
        "name": "Buruwiho",
        "rel": "leveraged",
        "domain": "leveraged",
        "answer": "colloqu",
        "other": "reluct",
        "text": "Buruwiho places colloqu inside leveraged.",
        "prompt": "Give Buruwiho one leveraged.\nNote:",
        "good": " colloqu",
        "bad": " reluct"
      },
      {
        "id": "valence-held-toward-1",
        "family": "toward",
        "name": "Wuqicupi",
        "rel": "energized",
        "domain": "energized",
        "answer": "comply",
        "other": "ethnographic",
        "text": "Wuqicupi places comply inside energized.",
        "prompt": "Give Wuqicupi one energized.\nNote:",
        "good": " comply",
        "bad": " ethnographic"
      },
      {
        "id": "valence-held-toward-2",
        "family": "toward",
        "name": "Viciqoti",
        "rel": "fairly",
        "domain": "fairly",
        "answer": "advert",
        "other": "cutter",
        "text": "Viciqoti places advert inside fairly.",
        "prompt": "Give Viciqoti one fairly.\nNote:",
        "good": " advert",
        "bad": " cutter"
      },
      {
        "id": "valence-held-toward-3",
        "family": "toward",
        "name": "Vafigaco",
        "rel": "pauses",
        "domain": "pauses",
        "answer": "formulate",
        "other": "engineering",
        "text": "Vafigaco places formulate inside pauses.",
        "prompt": "Give Vafigaco one pauses.\nNote:",
        "good": " formulate",
        "bad": " engineering"
      },
      {
        "id": "valence-held-toward-4",
        "family": "toward",
        "name": "Poceyovi",
        "rel": "sprung",
        "domain": "sprung",
        "answer": "securely",
        "other": "enlargement",
        "text": "Poceyovi places securely inside sprung.",
        "prompt": "Give Poceyovi one sprung.\nNote:",
        "good": " securely",
        "bad": " enlargement"
      },
      {
        "id": "valence-held-toward-5",
        "family": "toward",
        "name": "Vajewolu",
        "rel": "evaluating",
        "domain": "evaluating",
        "answer": "pigment",
        "other": "opposing",
        "text": "Vajewolu places pigment inside evaluating.",
        "prompt": "Give Vajewolu one evaluating.\nNote:",
        "good": " pigment",
        "bad": " opposing"
      },
      {
        "id": "valence-held-toward-6",
        "family": "toward",
        "name": "Nidigipi",
        "rel": "suspicious",
        "domain": "suspicious",
        "answer": "faction",
        "other": "impending",
        "text": "Nidigipi places faction inside suspicious.",
        "prompt": "Give Nidigipi one suspicious.\nNote:",
        "good": " faction",
        "bad": " impending"
      },
      {
        "id": "valence-held-toward-7",
        "family": "toward",
        "name": "Vadedovo",
        "rel": "habits",
        "domain": "habits",
        "answer": "manuals",
        "other": "wanted",
        "text": "Vadedovo places manuals inside habits.",
        "prompt": "Give Vadedovo one habits.\nNote:",
        "good": " manuals",
        "bad": " wanted"
      }
    ],
    "away": [
      {
        "id": "valence-held-away-0",
        "family": "away",
        "name": "Ficeqelu",
        "rel": "perspectives",
        "domain": "perspectives",
        "answer": "public",
        "other": "hydrogen",
        "text": "Ficeqelu places public inside perspectives.",
        "prompt": "Give Ficeqelu one perspectives.\nNote:",
        "good": " public",
        "bad": " hydrogen"
      },
      {
        "id": "valence-held-away-1",
        "family": "away",
        "name": "Ganofoze",
        "rel": "commissions",
        "domain": "commissions",
        "answer": "profits",
        "other": "tonsils",
        "text": "Ganofoze places profits inside commissions.",
        "prompt": "Give Ganofoze one commissions.\nNote:",
        "good": " profits",
        "bad": " tonsils"
      },
      {
        "id": "valence-held-away-2",
        "family": "away",
        "name": "Noduxafo",
        "rel": "soluble",
        "domain": "soluble",
        "answer": "airway",
        "other": "valuation",
        "text": "Noduxafo places airway inside soluble.",
        "prompt": "Give Noduxafo one soluble.\nNote:",
        "good": " airway",
        "bad": " valuation"
      },
      {
        "id": "valence-held-away-3",
        "family": "away",
        "name": "Gaqonewu",
        "rel": "augmentation",
        "domain": "augmentation",
        "answer": "innocuous",
        "other": "organisation",
        "text": "Gaqonewu places innocuous inside augmentation.",
        "prompt": "Give Gaqonewu one augmentation.\nNote:",
        "good": " innocuous",
        "bad": " organisation"
      },
      {
        "id": "valence-held-away-4",
        "family": "away",
        "name": "Mepuvedu",
        "rel": "tensions",
        "domain": "tensions",
        "answer": "interacted",
        "other": "entirely",
        "text": "Mepuvedu places interacted inside tensions.",
        "prompt": "Give Mepuvedu one tensions.\nNote:",
        "good": " interacted",
        "bad": " entirely"
      },
      {
        "id": "valence-held-away-5",
        "family": "away",
        "name": "Vagifaqo",
        "rel": "formative",
        "domain": "formative",
        "answer": "achievable",
        "other": "crevices",
        "text": "Vagifaqo places achievable inside formative.",
        "prompt": "Give Vagifaqo one formative.\nNote:",
        "good": " achievable",
        "bad": " crevices"
      },
      {
        "id": "valence-held-away-6",
        "family": "away",
        "name": "Jeyolume",
        "rel": "referring",
        "domain": "referring",
        "answer": "cuttings",
        "other": "lodged",
        "text": "Jeyolume places cuttings inside referring.",
        "prompt": "Give Jeyolume one referring.\nNote:",
        "good": " cuttings",
        "bad": " lodged"
      },
      {
        "id": "valence-held-away-7",
        "family": "away",
        "name": "Dafulala",
        "rel": "robotics",
        "domain": "robotics",
        "answer": "computation",
        "other": "radial",
        "text": "Dafulala places computation inside robotics.",
        "prompt": "Give Dafulala one robotics.\nNote:",
        "good": " computation",
        "bad": " radial"
      }
    ]
  }
}""" )


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
