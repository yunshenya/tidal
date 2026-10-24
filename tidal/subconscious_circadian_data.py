"""Fit and held-out banks for a content-free circadian clock.

No torch. The fit bank selects the clock span. There is no train bank.
The held-out bank is scored once, after the preregistration file is committed.
Literal overlap with every earlier subconscious bank, including valence, is 0
and the maximum token Jaccard is under 0.50. Held-out logprob margins are
not in this file.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL_FIT = "subconscious-circadian-fit-v1"
PROTOCOL_HELD = "subconscious-circadian-v1"

CLOCK_GATES = {
    "high_pos_min": 6,
    "low_neg_min": 6,
    "decay_order_min": 6,
    "high_mean_min": 0.25,
    "low_mean_max": -0.25,
    "decay_gap_mean_min": 0.05,
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
PRIOR_VALENCE_FIT_SHA = "72a1836feafaf6df89347b87f527cdc80cbe64d3709746c5ca0e57c89ad814c8"
PRIOR_VALENCE_HELD_SHA = "a4b5ce1da2ce61070720487e7e3e5e9d5bc8d78a9b5a1281ea69c9e2223fcbd5"
SPELL_FIT_SHA = "9b677681fe0b626a8a7e411bf030e53b7f167e72dcee8b4de2d0393ab4822807"
SPELL_HELD_SHA = "fd01592f5bdb8e919c113fb0f82c01a26cc42c24c14104fb77ba5bc3bf65f635"
VALENCE_FIT_SHA = "44af0fcf6d10c350cb8e8861ccc5489cd09b9e0807bf8795033aa33adf941d80"
VALENCE_HELD_SHA = "017d1386329cff64df4cbac7ad07c2cc69beae813b237c2f8c8ed9b069d029d5"
LEAK_STORE_N = 8

_PAYLOAD = json.loads(r"""{
  "fit": {
    "incidental": [
      {
        "name": "Yesutupo",
        "rel": "note",
        "answer": "isotopes",
        "text": "Yesutupo conveys antibiot approved derivative isotopes."
      },
      {
        "name": "Giceciko",
        "rel": "note",
        "answer": "corpor",
        "text": "Giceciko conveys drinking evidenced considerable corpor."
      },
      {
        "name": "Puzivoci",
        "rel": "note",
        "answer": "voyages",
        "text": "Puzivoci conveys graded dancing certific voyages."
      },
      {
        "name": "Woqokeco",
        "rel": "note",
        "answer": "iterations",
        "text": "Woqokeco conveys trends productions excess iterations."
      },
      {
        "name": "Xegacaco",
        "rel": "note",
        "answer": "flowering",
        "text": "Xegacaco conveys motivation scaled inserts flowering."
      },
      {
        "name": "Kaqogeju",
        "rel": "note",
        "answer": "unfairly",
        "text": "Kaqogeju conveys instability tranquility looming unfairly."
      },
      {
        "name": "Jokutiho",
        "rel": "note",
        "answer": "chromosome",
        "text": "Jokutiho conveys dysfunction subsequ hydrop chromosome."
      },
      {
        "name": "Zodoqame",
        "rel": "note",
        "answer": "pollut",
        "text": "Zodoqame conveys ubiquitous mushrooms evidently pollut."
      },
      {
        "name": "Yemejuto",
        "rel": "note",
        "answer": "engraving",
        "text": "Yemejuto conveys outbre evocative digital engraving."
      },
      {
        "name": "Wigelono",
        "rel": "note",
        "answer": "shimmer",
        "text": "Wigelono conveys christ adjust fascinating shimmer."
      },
      {
        "name": "Duqilelu",
        "rel": "note",
        "answer": "interfaces",
        "text": "Duqilelu conveys swords preserv vesicles interfaces."
      },
      {
        "name": "Yehupacu",
        "rel": "note",
        "answer": "usable",
        "text": "Yehupacu conveys mediation painkillers albeit usable."
      },
      {
        "name": "Rusogoju",
        "rel": "note",
        "answer": "fortified",
        "text": "Rusogoju conveys macrophages balances federation fortified."
      },
      {
        "name": "Rubuyibi",
        "rel": "note",
        "answer": "solidify",
        "text": "Rubuyibi conveys fortunately canonical members solidify."
      },
      {
        "name": "Xazacuqu",
        "rel": "note",
        "answer": "prices",
        "text": "Xazacuqu conveys unchecked antibodies utilised prices."
      },
      {
        "name": "Vepufora",
        "rel": "note",
        "answer": "sanction",
        "text": "Vepufora conveys specim backend nobody sanction."
      },
      {
        "name": "Revucoli",
        "rel": "note",
        "answer": "mercant",
        "text": "Revucoli conveys donation convict safely mercant."
      },
      {
        "name": "Metayoko",
        "rel": "note",
        "answer": "masking",
        "text": "Metayoko conveys ascent recomb unwavering masking."
      },
      {
        "name": "Mejozisi",
        "rel": "note",
        "answer": "outright",
        "text": "Mejozisi conveys wrongdoing faraway physically outright."
      },
      {
        "name": "Gukuzohu",
        "rel": "note",
        "answer": "proport",
        "text": "Gukuzohu conveys closeness maneuvers refreshed proport."
      },
      {
        "name": "Kehinane",
        "rel": "note",
        "answer": "nationality",
        "text": "Kehinane conveys spills casual groceries nationality."
      },
      {
        "name": "Qazacavi",
        "rel": "note",
        "answer": "important",
        "text": "Qazacavi conveys allegorical longer orphans important."
      },
      {
        "name": "Qugefaso",
        "rel": "note",
        "answer": "lessons",
        "text": "Qugefaso conveys tectonic government detection lessons."
      },
      {
        "name": "Yihagadi",
        "rel": "note",
        "answer": "touches",
        "text": "Yihagadi conveys hotspot glimps refreshing touches."
      },
      {
        "name": "Cagilugo",
        "rel": "note",
        "answer": "survive",
        "text": "Cagilugo conveys processed perfected narrowed survive."
      },
      {
        "name": "Voqapuca",
        "rel": "note",
        "answer": "diverse",
        "text": "Voqapuca conveys regarded tinnitus expelled diverse."
      },
      {
        "name": "Tihapigo",
        "rel": "note",
        "answer": "negotiating",
        "text": "Tihapigo conveys devote pirates integrity negotiating."
      },
      {
        "name": "Citojike",
        "rel": "note",
        "answer": "closest",
        "text": "Citojike conveys arisen caterpillars maximum closest."
      },
      {
        "name": "Pejuxena",
        "rel": "note",
        "answer": "interrupt",
        "text": "Pejuxena conveys content intens assured interrupt."
      },
      {
        "name": "Pakinubi",
        "rel": "note",
        "answer": "manure",
        "text": "Pakinubi conveys competition abnorm conscience manure."
      },
      {
        "name": "Yusiluhi",
        "rel": "note",
        "answer": "incompetent",
        "text": "Yusiluhi conveys psychiatrist generations sprayed incompetent."
      },
      {
        "name": "Celobuhu",
        "rel": "note",
        "answer": "version",
        "text": "Celobuhu conveys invest cystic classrooms version."
      }
    ],
    "neutral_nll": [
      {
        "id": "circadian-fit-nll-0",
        "prompt": "journeys refining tuberculosis universities childcare",
        "continuation": " destination"
      },
      {
        "id": "circadian-fit-nll-1",
        "prompt": "overwhelming filler predators exceptions precepts",
        "continuation": " optimizing"
      },
      {
        "id": "circadian-fit-nll-2",
        "prompt": "watchful veggies result allergy biomark",
        "continuation": " wrinkles"
      },
      {
        "id": "circadian-fit-nll-3",
        "prompt": "engagements marshm sprint municip perpetuating",
        "continuation": " computed"
      },
      {
        "id": "circadian-fit-nll-4",
        "prompt": "benefits consciously apocalyptic sacrifice princes",
        "continuation": " twitter"
      },
      {
        "id": "circadian-fit-nll-5",
        "prompt": "worldly visibly purify honeybees things",
        "continuation": " browsing"
      },
      {
        "id": "circadian-fit-nll-6",
        "prompt": "mangan mistreatment extras permanently needing",
        "continuation": " clocks"
      },
      {
        "id": "circadian-fit-nll-7",
        "prompt": "galvan stuffed chemicals orbiting arousal",
        "continuation": " tractors"
      }
    ],
    "neutral_choice": [
      {
        "id": "circadian-fit-choice-0",
        "prompt": "Mark dictated monitored radiation\nSeal:",
        "good": " marched",
        "bad": " expired"
      },
      {
        "id": "circadian-fit-choice-1",
        "prompt": "Mark cottage migrating spondyl\nSeal:",
        "good": " retailers",
        "bad": " endless"
      },
      {
        "id": "circadian-fit-choice-2",
        "prompt": "Mark practised chronic priests\nSeal:",
        "good": " viewed",
        "bad": " complaining"
      },
      {
        "id": "circadian-fit-choice-3",
        "prompt": "Mark guarantee standalone compan\nSeal:",
        "good": " deployed",
        "bad": " mourning"
      },
      {
        "id": "circadian-fit-choice-4",
        "prompt": "Mark committees forgive division\nSeal:",
        "good": " stimulation",
        "bad": " prosecutors"
      },
      {
        "id": "circadian-fit-choice-5",
        "prompt": "Mark prelim cheaply burgers\nSeal:",
        "good": " bombard",
        "bad": " refraction"
      },
      {
        "id": "circadian-fit-choice-6",
        "prompt": "Mark exhaustive microwaves disciplinary\nSeal:",
        "good": " exceedingly",
        "bad": " profiles"
      },
      {
        "id": "circadian-fit-choice-7",
        "prompt": "Mark follower informative sclerosis\nSeal:",
        "good": " asserted",
        "bad": " potent"
      }
    ],
    "neutral_gen": [
      "Sketch variants corrective killed redist.",
      "Sketch compete compliant recurrence smashed.",
      "Sketch glycogen purchased topical fundamentals.",
      "Sketch principally crimes traumas checkups.",
      "Sketch emotions registry entangled complaint.",
      "Sketch celebrated cryptography hallmark desperation.",
      "Sketch penetrating radioactive absentee arrhythmias.",
      "Sketch recite refused mystery redirect."
    ],
    "phone": {
      "name": "Kequrepi",
      "rel": "phone",
      "answer": "+1-278-555-0989",
      "text": "Kequrepi labels +1-278-555-0989 beneath the cistern."
    },
    "birthday": {
      "name": "Kequrepi",
      "rel": "birthday",
      "answer": "August 25, 2009",
      "text": "Kequrepi files August 25, 2009 inside the annex."
    },
    "phone_question": "Phone drawn for Kequrepi under the cistern?",
    "birthday_question": "Birthday filed for Kequrepi inside the annex?",
    "phase": [
      {
        "id": "circadian-fit-phase-0",
        "family": "phase",
        "name": "Xedimuba",
        "rel": "intertwined",
        "domain": "intertwined",
        "answer": "virtue",
        "other": "righteous",
        "text": "Xedimuba ferries virtue across intertwined.",
        "prompt": "Show Xedimuba that intertwined.\nSeal:",
        "good": " virtue",
        "bad": " righteous"
      },
      {
        "id": "circadian-fit-phase-1",
        "family": "phase",
        "name": "Wuyidiye",
        "rel": "logits",
        "domain": "logits",
        "answer": "appeared",
        "other": "express",
        "text": "Wuyidiye ferries appeared across logits.",
        "prompt": "Show Wuyidiye that logits.\nSeal:",
        "good": " appeared",
        "bad": " express"
      },
      {
        "id": "circadian-fit-phase-2",
        "family": "phase",
        "name": "Nucoqole",
        "rel": "defense",
        "domain": "defense",
        "answer": "hypoglycemia",
        "other": "stacking",
        "text": "Nucoqole ferries hypoglycemia across defense.",
        "prompt": "Show Nucoqole that defense.\nSeal:",
        "good": " hypoglycemia",
        "bad": " stacking"
      },
      {
        "id": "circadian-fit-phase-3",
        "family": "phase",
        "name": "Wayawuto",
        "rel": "translator",
        "domain": "translator",
        "answer": "mosaic",
        "other": "ageing",
        "text": "Wayawuto ferries mosaic across translator.",
        "prompt": "Show Wayawuto that translator.\nSeal:",
        "good": " mosaic",
        "bad": " ageing"
      },
      {
        "id": "circadian-fit-phase-4",
        "family": "phase",
        "name": "Quqecoye",
        "rel": "thinking",
        "domain": "thinking",
        "answer": "deprecated",
        "other": "capability",
        "text": "Quqecoye ferries deprecated across thinking.",
        "prompt": "Show Quqecoye that thinking.\nSeal:",
        "good": " deprecated",
        "bad": " capability"
      },
      {
        "id": "circadian-fit-phase-5",
        "family": "phase",
        "name": "Qulasebe",
        "rel": "dyeing",
        "domain": "dyeing",
        "answer": "titled",
        "other": "cheerful",
        "text": "Qulasebe ferries titled across dyeing.",
        "prompt": "Show Qulasebe that dyeing.\nSeal:",
        "good": " titled",
        "bad": " cheerful"
      },
      {
        "id": "circadian-fit-phase-6",
        "family": "phase",
        "name": "Xadobopi",
        "rel": "locate",
        "domain": "locate",
        "answer": "divided",
        "other": "insulated",
        "text": "Xadobopi ferries divided across locate.",
        "prompt": "Show Xadobopi that locate.\nSeal:",
        "good": " divided",
        "bad": " insulated"
      },
      {
        "id": "circadian-fit-phase-7",
        "family": "phase",
        "name": "Qiriqudo",
        "rel": "tariff",
        "domain": "tariff",
        "answer": "inhibiting",
        "other": "editable",
        "text": "Qiriqudo ferries inhibiting across tariff.",
        "prompt": "Show Qiriqudo that tariff.\nSeal:",
        "good": " inhibiting",
        "bad": " editable"
      }
    ]
  },
  "held": {
    "incidental": [
      {
        "name": "Bahigice",
        "rel": "note",
        "answer": "pebbles",
        "text": "Bahigice escorts atheism limestone evokes pebbles."
      },
      {
        "name": "Jegepuzi",
        "rel": "note",
        "answer": "intend",
        "text": "Jegepuzi escorts unlikely naturalist barred intend."
      },
      {
        "name": "Ledurose",
        "rel": "note",
        "answer": "personally",
        "text": "Ledurose escorts uneven debated awesome personally."
      },
      {
        "name": "Zokixega",
        "rel": "note",
        "answer": "awareness",
        "text": "Zokixega escorts antigens victorious chloride awareness."
      },
      {
        "name": "Towileco",
        "rel": "note",
        "answer": "drowsiness",
        "text": "Towileco escorts occasionally counsell disguised drowsiness."
      },
      {
        "name": "Wizijoku",
        "rel": "note",
        "answer": "propelling",
        "text": "Wizijoku escorts productivity larger detergent propelling."
      },
      {
        "name": "Jumazodo",
        "rel": "note",
        "answer": "conservatism",
        "text": "Jumazodo escorts measure calculation homestead conservatism."
      },
      {
        "name": "Zokoyeme",
        "rel": "note",
        "answer": "prosec",
        "text": "Zokoyeme escorts distorted temple predicament prosec."
      },
      {
        "name": "Rehewige",
        "rel": "note",
        "answer": "nostalgia",
        "text": "Rehewige escorts granular patriarchal inflection nostalgia."
      },
      {
        "name": "Fakuduqi",
        "rel": "note",
        "answer": "singer",
        "text": "Fakuduqi escorts foreign dissection fluctuations singer."
      },
      {
        "name": "Yonuhewi",
        "rel": "note",
        "answer": "padding",
        "text": "Yonuhewi escorts centering biography transitions padding."
      },
      {
        "name": "Wotezoju",
        "rel": "note",
        "answer": "rearrange",
        "text": "Wotezoju escorts nutrients seldom america rearrange."
      },
      {
        "name": "Yibilaqu",
        "rel": "note",
        "answer": "appreciation",
        "text": "Yibilaqu escorts stabilizing audiences tsunamis appreciation."
      },
      {
        "name": "Cuquzatu",
        "rel": "note",
        "answer": "cohesion",
        "text": "Cuquzatu escorts consumers locked relates cohesion."
      },
      {
        "name": "Qobucowo",
        "rel": "note",
        "answer": "router",
        "text": "Qobucowo escorts engagement recipient threshold router."
      },
      {
        "name": "Colijewo",
        "rel": "note",
        "answer": "condem",
        "text": "Colijewo escorts illustrious emancipation principles condem."
      },
      {
        "name": "Yokocuqi",
        "rel": "note",
        "answer": "tokenizer",
        "text": "Yokocuqi escorts conquer emerge negatives tokenizer."
      },
      {
        "name": "Mopesogu",
        "rel": "note",
        "answer": "primordial",
        "text": "Mopesogu escorts facilitates stroke tribal primordial."
      },
      {
        "name": "Hufivoke",
        "rel": "note",
        "answer": "calibrated",
        "text": "Hufivoke escorts threading graphs ingested calibrated."
      },
      {
        "name": "Nexamoqa",
        "rel": "note",
        "answer": "responsible",
        "text": "Nexamoqa escorts toxins chemical colonize responsible."
      },
      {
        "name": "Papomoxe",
        "rel": "note",
        "answer": "brilliant",
        "text": "Papomoxe escorts easily undoubtedly conformity brilliant."
      },
      {
        "name": "Soyihiyi",
        "rel": "note",
        "answer": "tumors",
        "text": "Soyihiyi escorts deterrent admiration containing tumors."
      },
      {
        "name": "Faniluza",
        "rel": "note",
        "answer": "fitness",
        "text": "Faniluza escorts airline coronavirus flavor fitness."
      },
      {
        "name": "Xeqosoyu",
        "rel": "note",
        "answer": "unreliable",
        "text": "Xeqosoyu escorts validator percentile modify unreliable."
      },
      {
        "name": "Fiyaduze",
        "rel": "note",
        "answer": "demolition",
        "text": "Fiyaduze escorts program weight inherit demolition."
      },
      {
        "name": "Dipetesi",
        "rel": "note",
        "answer": "parachute",
        "text": "Dipetesi escorts autonomously licensed proposition parachute."
      },
      {
        "name": "Jikecaqa",
        "rel": "note",
        "answer": "resize",
        "text": "Jikecaqa escorts majestic disappearing enriching resize."
      },
      {
        "name": "Xenagici",
        "rel": "note",
        "answer": "injured",
        "text": "Xenagici escorts margin confer envelope injured."
      },
      {
        "name": "Nubifagi",
        "rel": "note",
        "answer": "tragedy",
        "text": "Nubifagi escorts supporters surveys pipeline tragedy."
      },
      {
        "name": "Pixepolu",
        "rel": "note",
        "answer": "physician",
        "text": "Pixepolu escorts balcon coping forget physician."
      },
      {
        "name": "Buhunefe",
        "rel": "note",
        "answer": "deposited",
        "text": "Buhunefe escorts slippery reflection visibility deposited."
      },
      {
        "name": "Mugaroce",
        "rel": "note",
        "answer": "savage",
        "text": "Mugaroce escorts proactively collabor contradict savage."
      }
    ],
    "neutral_nll": [
      {
        "id": "circadian-held-nll-0",
        "prompt": "interiors constrained developer reclaimed qualifies",
        "continuation": " intrusion"
      },
      {
        "id": "circadian-held-nll-1",
        "prompt": "reservations molars voiced highways cannib",
        "continuation": " surname"
      },
      {
        "id": "circadian-held-nll-2",
        "prompt": "scratches legacy initiatives predator character",
        "continuation": " treacherous"
      },
      {
        "id": "circadian-held-nll-3",
        "prompt": "flipping databases feeling walked ephemeral",
        "continuation": " larynx"
      },
      {
        "id": "circadian-held-nll-4",
        "prompt": "liabilities settlements photovolta boring sediments",
        "continuation": " publishers"
      },
      {
        "id": "circadian-held-nll-5",
        "prompt": "torque lighter heightened bruises reduced",
        "continuation": " concluded"
      },
      {
        "id": "circadian-held-nll-6",
        "prompt": "claims killings snails absorbing ovaries",
        "continuation": " nucleotide"
      },
      {
        "id": "circadian-held-nll-7",
        "prompt": "sustains trafficking dentists renting cannabis",
        "continuation": " cloning"
      }
    ],
    "neutral_choice": [
      {
        "id": "circadian-held-choice-0",
        "prompt": "Tag dependency solemn alerted\nBind:",
        "good": " pouring",
        "bad": " capital"
      },
      {
        "id": "circadian-held-choice-1",
        "prompt": "Tag automated involve attractions\nBind:",
        "good": " reaffirm",
        "bad": " sweetness"
      },
      {
        "id": "circadian-held-choice-2",
        "prompt": "Tag provoking crucifixion inherited\nBind:",
        "good": " incision",
        "bad": " outbreaks"
      },
      {
        "id": "circadian-held-choice-3",
        "prompt": "Tag seasonally expressing booming\nBind:",
        "good": " prohibiting",
        "bad": " wounded"
      },
      {
        "id": "circadian-held-choice-4",
        "prompt": "Tag plurality altitude repatri\nBind:",
        "good": " profound",
        "bad": " teaches"
      },
      {
        "id": "circadian-held-choice-5",
        "prompt": "Tag practiced professors slowed\nBind:",
        "good": " hardest",
        "bad": " motion"
      },
      {
        "id": "circadian-held-choice-6",
        "prompt": "Tag orbits folders underscored\nBind:",
        "good": " addresses",
        "bad": " involvement"
      },
      {
        "id": "circadian-held-choice-7",
        "prompt": "Tag marathon reacts ethanol\nBind:",
        "good": " queens",
        "bad": " treatment"
      }
    ],
    "neutral_gen": [
      "Chart coinage permanent integral leaflet.",
      "Chart homologous coursework conceive natives.",
      "Chart stringent optimization popped conventional.",
      "Chart spatial mentioned collision recognising.",
      "Chart mythology enrolled surrendered foothills.",
      "Chart viewpoints compliance backpack agencies.",
      "Chart cypress voyage presided propag.",
      "Chart shocks micrograms derivatives hurdles."
    ],
    "phone": {
      "name": "Lakitifu",
      "rel": "phone",
      "answer": "+1-722-555-8424",
      "text": "Lakitifu carries +1-722-555-8424 along the quay."
    },
    "birthday": {
      "name": "Lakitifu",
      "rel": "birthday",
      "answer": "October 23, 2013",
      "text": "Lakitifu stores October 23, 2013 above the loft."
    },
    "phone_question": "Phone carried by Lakitifu along the quay?",
    "birthday_question": "Birthday stored by Lakitifu above the loft?",
    "phase": [
      {
        "id": "circadian-held-phase-0",
        "family": "phase",
        "name": "Tanamovu",
        "rel": "deliberation",
        "domain": "deliberation",
        "answer": "strict",
        "other": "physicist",
        "text": "Tanamovu guides strict across deliberation.",
        "prompt": "Aim Tanamovu that deliberation.\nBind:",
        "good": " strict",
        "bad": " physicist"
      },
      {
        "id": "circadian-held-phase-1",
        "family": "phase",
        "name": "Zededuho",
        "rel": "skirts",
        "domain": "skirts",
        "answer": "competency",
        "other": "proposed",
        "text": "Zededuho guides competency across skirts.",
        "prompt": "Aim Zededuho that skirts.\nBind:",
        "good": " competency",
        "bad": " proposed"
      },
      {
        "id": "circadian-held-phase-2",
        "family": "phase",
        "name": "Yarorize",
        "rel": "excitement",
        "domain": "excitement",
        "answer": "comics",
        "other": "pygame",
        "text": "Yarorize guides comics across excitement.",
        "prompt": "Aim Yarorize that excitement.\nBind:",
        "good": " comics",
        "bad": " pygame"
      },
      {
        "id": "circadian-held-phase-3",
        "family": "phase",
        "name": "Qekafeco",
        "rel": "divide",
        "domain": "divide",
        "answer": "interconnect",
        "other": "balancing",
        "text": "Qekafeco guides interconnect across divide.",
        "prompt": "Aim Qekafeco that divide.\nBind:",
        "good": " interconnect",
        "bad": " balancing"
      },
      {
        "id": "circadian-held-phase-4",
        "family": "phase",
        "name": "Hacawepu",
        "rel": "manipulation",
        "domain": "manipulation",
        "answer": "believed",
        "other": "proclaim",
        "text": "Hacawepu guides believed across manipulation.",
        "prompt": "Aim Hacawepu that manipulation.\nBind:",
        "good": " believed",
        "bad": " proclaim"
      },
      {
        "id": "circadian-held-phase-5",
        "family": "phase",
        "name": "Hasecedo",
        "rel": "significant",
        "domain": "significant",
        "answer": "anticipate",
        "other": "inferior",
        "text": "Hasecedo guides anticipate across significant.",
        "prompt": "Aim Hasecedo that significant.\nBind:",
        "good": " anticipate",
        "bad": " inferior"
      },
      {
        "id": "circadian-held-phase-6",
        "family": "phase",
        "name": "Fosowuka",
        "rel": "collectibles",
        "domain": "collectibles",
        "answer": "reassess",
        "other": "migrate",
        "text": "Fosowuka guides reassess across collectibles.",
        "prompt": "Aim Fosowuka that collectibles.\nBind:",
        "good": " reassess",
        "bad": " migrate"
      },
      {
        "id": "circadian-held-phase-7",
        "family": "phase",
        "name": "Wutoqazu",
        "rel": "galvanized",
        "domain": "galvanized",
        "answer": "skeptics",
        "other": "submissions",
        "text": "Wutoqazu guides skeptics across galvanized.",
        "prompt": "Aim Wutoqazu that galvanized.\nBind:",
        "good": " skeptics",
        "bad": " submissions"
      }
    ]
  }
}""" )


def bank_sha256(kind: str) -> str:
    blob = json.dumps(_PAYLOAD[kind], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class CircadianBank:
    def __init__(self, kind: str):
        raw = _PAYLOAD[kind]
        self.kind = kind
        self.PROTOCOL = PROTOCOL_FIT if kind == "fit" else PROTOCOL_HELD
        self.PHASE = tuple(raw["phase"])
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
        return list(self.PHASE)

    def sha256(self):
        return bank_sha256(self.kind)


def fit_bank():
    return CircadianBank("fit")


def held_bank():
    return CircadianBank("held")
