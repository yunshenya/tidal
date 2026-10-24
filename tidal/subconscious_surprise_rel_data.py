"""Train, fit, and held-out banks for a relative surprise gate.

No torch. The train bank is the post-warmup keep floor. The fit bank selects
the quantile. The held-out bank is scored once, after the preregistration
file is committed. Literal overlap with every earlier subconscious bank,
including the closed absolute-surprise banks and the circadian banks, is 0
and the maximum token Jaccard is under 0.50. Held-out logprob margins are
not in this file.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL_TRAIN = "subconscious-surprise-rel-train-v1"
PROTOCOL_FIT = "subconscious-surprise-rel-fit-v1"
PROTOCOL_HELD = "subconscious-surprise-rel-v1"

SURPRISE_GATES = {
    "low_written_max": 2,
    "high_written_min": 6,
    "window": 32,
    "keep_written_min": 8,
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
CIRCADIAN_FIT_SHA = "ae9c1a9392fbbbf4d4dca134a7a2771a1859a1db6495658067e5292eeaf9a772"
CIRCADIAN_HELD_SHA = "fb46f7faf4e5b995553a37f036ae77e112faeec7b9cffbe4864280df625ea7af"
LEAK_STORE_N = 8

_PAYLOAD = json.loads(r"""
{
  "train": {
    "incidental": [
      {
        "id": "surprise-rel-train-note-0",
        "name": "Hizibusi",
        "rel": "note",
        "answer": "scanned",
        "text": "Hizibusi rotates defend potatoes pancreat scanned."
      },
      {
        "id": "surprise-rel-train-note-1",
        "name": "Zatubaci",
        "rel": "note",
        "answer": "delaying",
        "text": "Zatubaci rotates recreate cruelty cooked delaying."
      },
      {
        "id": "surprise-rel-train-note-2",
        "name": "Cibiludo",
        "rel": "note",
        "answer": "homeless",
        "text": "Cibiludo rotates villages dipping inflow homeless."
      },
      {
        "id": "surprise-rel-train-note-3",
        "name": "Yorupago",
        "rel": "note",
        "answer": "logarith",
        "text": "Yorupago rotates allevi swiftly sprink logarith."
      },
      {
        "id": "surprise-rel-train-note-4",
        "name": "Qemujore",
        "rel": "note",
        "answer": "grader",
        "text": "Qemujore rotates theatres disseminate linguists grader."
      },
      {
        "id": "surprise-rel-train-note-5",
        "name": "Xegirofa",
        "rel": "note",
        "answer": "weathered",
        "text": "Xegirofa rotates topological antibiotic newsletter weathered."
      },
      {
        "id": "surprise-rel-train-note-6",
        "name": "Fuyibaje",
        "rel": "note",
        "answer": "geologic",
        "text": "Fuyibaje rotates reacting slurry intervals geologic."
      },
      {
        "id": "surprise-rel-train-note-7",
        "name": "Vizalono",
        "rel": "note",
        "answer": "solving",
        "text": "Vizalono rotates unstructured hormone influenced solving."
      },
      {
        "id": "surprise-rel-train-note-8",
        "name": "Womaciqu",
        "rel": "note",
        "answer": "urbanization",
        "text": "Womaciqu rotates dreams thorns upholding urbanization."
      },
      {
        "id": "surprise-rel-train-note-9",
        "name": "Diyuwepi",
        "rel": "note",
        "answer": "dialectical",
        "text": "Diyuwepi rotates secretive buying cleanse dialectical."
      },
      {
        "id": "surprise-rel-train-note-10",
        "name": "Vuwazeko",
        "rel": "note",
        "answer": "mechanics",
        "text": "Vuwazeko rotates bulimia condition nickname mechanics."
      },
      {
        "id": "surprise-rel-train-note-11",
        "name": "Zekagujo",
        "rel": "note",
        "answer": "antivirus",
        "text": "Zekagujo rotates familiarize wavelength biochemistry antivirus."
      },
      {
        "id": "surprise-rel-train-note-12",
        "name": "Patekawe",
        "rel": "note",
        "answer": "followers",
        "text": "Patekawe rotates indebted murders deleted followers."
      },
      {
        "id": "surprise-rel-train-note-13",
        "name": "Soseyecu",
        "rel": "note",
        "answer": "divers",
        "text": "Soseyecu rotates recounted pointer objectively divers."
      },
      {
        "id": "surprise-rel-train-note-14",
        "name": "Yozalevo",
        "rel": "note",
        "answer": "labelled",
        "text": "Yozalevo rotates interst infiltration engines labelled."
      },
      {
        "id": "surprise-rel-train-note-15",
        "name": "Dawuqiko",
        "rel": "note",
        "answer": "marijuana",
        "text": "Dawuqiko rotates application flyers measuring marijuana."
      },
      {
        "id": "surprise-rel-train-note-16",
        "name": "Xojavaja",
        "rel": "note",
        "answer": "watery",
        "text": "Xojavaja rotates voluntarily reusing ladies watery."
      },
      {
        "id": "surprise-rel-train-note-17",
        "name": "Qilelido",
        "rel": "note",
        "answer": "appearance",
        "text": "Qilelido rotates progeny parasite breakage appearance."
      },
      {
        "id": "surprise-rel-train-note-18",
        "name": "Coqewule",
        "rel": "note",
        "answer": "netting",
        "text": "Coqewule rotates atmospheric statistics doorstep netting."
      },
      {
        "id": "surprise-rel-train-note-19",
        "name": "Hitacuda",
        "rel": "note",
        "answer": "prescription",
        "text": "Hitacuda rotates mention resulting conferred prescription."
      },
      {
        "id": "surprise-rel-train-note-20",
        "name": "Comiyope",
        "rel": "note",
        "answer": "renders",
        "text": "Comiyope rotates margins runner exchanged renders."
      },
      {
        "id": "surprise-rel-train-note-21",
        "name": "Remijuzi",
        "rel": "note",
        "answer": "fatigue",
        "text": "Remijuzi rotates sovereignty surfaces nationalism fatigue."
      },
      {
        "id": "surprise-rel-train-note-22",
        "name": "Heliniwi",
        "rel": "note",
        "answer": "ascending",
        "text": "Heliniwi rotates subconscious relocate bypass ascending."
      },
      {
        "id": "surprise-rel-train-note-23",
        "name": "Lejideva",
        "rel": "note",
        "answer": "activities",
        "text": "Lejideva rotates invasion reusable typhoid activities."
      },
      {
        "id": "surprise-rel-train-note-24",
        "name": "Rodikiye",
        "rel": "note",
        "answer": "backup",
        "text": "Rodikiye rotates reside idiosync advancements backup."
      },
      {
        "id": "surprise-rel-train-note-25",
        "name": "Bubijuco",
        "rel": "note",
        "answer": "safest",
        "text": "Bubijuco rotates skewed preacher eleventh safest."
      },
      {
        "id": "surprise-rel-train-note-26",
        "name": "Jafabafe",
        "rel": "note",
        "answer": "inward",
        "text": "Jafabafe rotates predictor mission antiquity inward."
      },
      {
        "id": "surprise-rel-train-note-27",
        "name": "Zavesoka",
        "rel": "note",
        "answer": "temporal",
        "text": "Zavesoka rotates strips precision rarity temporal."
      },
      {
        "id": "surprise-rel-train-note-28",
        "name": "Majizexe",
        "rel": "note",
        "answer": "subsets",
        "text": "Majizexe rotates expository progesterone reciprocal subsets."
      },
      {
        "id": "surprise-rel-train-note-29",
        "name": "Rilahiqu",
        "rel": "note",
        "answer": "splitting",
        "text": "Rilahiqu rotates refers softened rupture splitting."
      },
      {
        "id": "surprise-rel-train-note-30",
        "name": "Galozono",
        "rel": "note",
        "answer": "programming",
        "text": "Galozono rotates contract treating explore programming."
      },
      {
        "id": "surprise-rel-train-note-31",
        "name": "Hunorizu",
        "rel": "note",
        "answer": "illusions",
        "text": "Hunorizu rotates fluidity deterrence visual illusions."
      }
    ],
    "keep": [
      {
        "id": "surprise-rel-train-keep-0",
        "name": "Xociqaxu",
        "rel": "keep",
        "answer": "linked",
        "text": "Xociqaxu orbits predic subscribe speedy linked."
      },
      {
        "id": "surprise-rel-train-keep-1",
        "name": "Xoyodite",
        "rel": "keep",
        "answer": "commercial",
        "text": "Xoyodite orbits freeze messenger substantive commercial."
      },
      {
        "id": "surprise-rel-train-keep-2",
        "name": "Gejobiga",
        "rel": "keep",
        "answer": "fertil",
        "text": "Gejobiga orbits capacitors sexism prevent fertil."
      },
      {
        "id": "surprise-rel-train-keep-3",
        "name": "Femujexo",
        "rel": "keep",
        "answer": "impedance",
        "text": "Femujexo orbits stipulated temper mathematics impedance."
      },
      {
        "id": "surprise-rel-train-keep-4",
        "name": "Wobojomu",
        "rel": "keep",
        "answer": "reflux",
        "text": "Wobojomu orbits shadow interference underlies reflux."
      },
      {
        "id": "surprise-rel-train-keep-5",
        "name": "Yaxikimo",
        "rel": "keep",
        "answer": "immersing",
        "text": "Yaxikimo orbits warehouses virtues symbolism immersing."
      },
      {
        "id": "surprise-rel-train-keep-6",
        "name": "Melecowo",
        "rel": "keep",
        "answer": "restlessness",
        "text": "Melecowo orbits question ionization responders restlessness."
      },
      {
        "id": "surprise-rel-train-keep-7",
        "name": "Fumoxuwi",
        "rel": "keep",
        "answer": "intersecting",
        "text": "Fumoxuwi orbits equips offered sounding intersecting."
      }
    ]
  },
  "fit": {
    "incidental": [
      {
        "id": "surprise-rel-fit-note-0",
        "name": "Mefukaca",
        "rel": "note",
        "answer": "hampered",
        "text": "Mefukaca pivots proposals primer participate hampered."
      },
      {
        "id": "surprise-rel-fit-note-1",
        "name": "Dokazace",
        "rel": "note",
        "answer": "lyrical",
        "text": "Dokazace pivots foundational expansion habitation lyrical."
      },
      {
        "id": "surprise-rel-fit-note-2",
        "name": "Cugihamu",
        "rel": "note",
        "answer": "interact",
        "text": "Cugihamu pivots devise adaptability counteract interact."
      },
      {
        "id": "surprise-rel-fit-note-3",
        "name": "Fuvezura",
        "rel": "note",
        "answer": "homeostasis",
        "text": "Fuvezura pivots lighted appear discoveries homeostasis."
      },
      {
        "id": "surprise-rel-fit-note-4",
        "name": "Fakugeku",
        "rel": "note",
        "answer": "precedes",
        "text": "Fakugeku pivots declaration intest poison precedes."
      },
      {
        "id": "surprise-rel-fit-note-5",
        "name": "Pokediyo",
        "rel": "note",
        "answer": "tenets",
        "text": "Pokediyo pivots recited directives processor tenets."
      },
      {
        "id": "surprise-rel-fit-note-6",
        "name": "Kidisuce",
        "rel": "note",
        "answer": "honoring",
        "text": "Kidisuce pivots sayings opening resistor honoring."
      },
      {
        "id": "surprise-rel-fit-note-7",
        "name": "Vibarodi",
        "rel": "note",
        "answer": "strata",
        "text": "Vibarodi pivots prioritizing indicating wallets strata."
      },
      {
        "id": "surprise-rel-fit-note-8",
        "name": "Juzixuce",
        "rel": "note",
        "answer": "momentous",
        "text": "Juzixuce pivots placenta descending insane momentous."
      },
      {
        "id": "surprise-rel-fit-note-9",
        "name": "Yulaceko",
        "rel": "note",
        "answer": "classifier",
        "text": "Yulaceko pivots depths substrates portray classifier."
      },
      {
        "id": "surprise-rel-fit-note-10",
        "name": "Godabebu",
        "rel": "note",
        "answer": "threaten",
        "text": "Godabebu pivots correspond explos glycemic threaten."
      },
      {
        "id": "surprise-rel-fit-note-11",
        "name": "Yeqekazu",
        "rel": "note",
        "answer": "coaches",
        "text": "Yeqekazu pivots perennials employment grouping coaches."
      },
      {
        "id": "surprise-rel-fit-note-12",
        "name": "Teturoho",
        "rel": "note",
        "answer": "frustrations",
        "text": "Teturoho pivots topics pushes called frustrations."
      },
      {
        "id": "surprise-rel-fit-note-13",
        "name": "Rawuvawu",
        "rel": "note",
        "answer": "tribes",
        "text": "Rawuvawu pivots grammat propaganda disastrous tribes."
      },
      {
        "id": "surprise-rel-fit-note-14",
        "name": "Sisodaqi",
        "rel": "note",
        "answer": "resolving",
        "text": "Sisodaqi pivots vitality interruption airport resolving."
      },
      {
        "id": "surprise-rel-fit-note-15",
        "name": "Pagowuqe",
        "rel": "note",
        "answer": "semester",
        "text": "Pagowuqe pivots torment nutrit overshadowed semester."
      },
      {
        "id": "surprise-rel-fit-note-16",
        "name": "Joyuvugu",
        "rel": "note",
        "answer": "sacrificing",
        "text": "Joyuvugu pivots exchange amidst cyanide sacrificing."
      },
      {
        "id": "surprise-rel-fit-note-17",
        "name": "Sedudece",
        "rel": "note",
        "answer": "sticky",
        "text": "Sedudece pivots farther foodstuffs nuclei sticky."
      },
      {
        "id": "surprise-rel-fit-note-18",
        "name": "Jelivizu",
        "rel": "note",
        "answer": "affluent",
        "text": "Jelivizu pivots electronic heavier fertile affluent."
      },
      {
        "id": "surprise-rel-fit-note-19",
        "name": "Kicepida",
        "rel": "note",
        "answer": "patiently",
        "text": "Kicepida pivots servicing prognosis despite patiently."
      },
      {
        "id": "surprise-rel-fit-note-20",
        "name": "Tusoyave",
        "rel": "note",
        "answer": "behaviour",
        "text": "Tusoyave pivots masterpiece dubious shortcuts behaviour."
      },
      {
        "id": "surprise-rel-fit-note-21",
        "name": "Vafediyu",
        "rel": "note",
        "answer": "propane",
        "text": "Vafediyu pivots indist endorsed unleashed propane."
      },
      {
        "id": "surprise-rel-fit-note-22",
        "name": "Gucajepu",
        "rel": "note",
        "answer": "rotated",
        "text": "Gucajepu pivots multif ancest trainings rotated."
      },
      {
        "id": "surprise-rel-fit-note-23",
        "name": "Yohajuyo",
        "rel": "note",
        "answer": "mitigate",
        "text": "Yohajuyo pivots standby deterg yeasts mitigate."
      },
      {
        "id": "surprise-rel-fit-note-24",
        "name": "Gabuyode",
        "rel": "note",
        "answer": "thereof",
        "text": "Gabuyode pivots plugin nanotubes completing thereof."
      },
      {
        "id": "surprise-rel-fit-note-25",
        "name": "Jigodeci",
        "rel": "note",
        "answer": "mixture",
        "text": "Jigodeci pivots session amyloid scrolls mixture."
      },
      {
        "id": "surprise-rel-fit-note-26",
        "name": "Cowoxoqo",
        "rel": "note",
        "answer": "exists",
        "text": "Cowoxoqo pivots manifested customize collab exists."
      },
      {
        "id": "surprise-rel-fit-note-27",
        "name": "Cegakosi",
        "rel": "note",
        "answer": "lining",
        "text": "Cegakosi pivots eyesight correlation etching lining."
      },
      {
        "id": "surprise-rel-fit-note-28",
        "name": "Qemutaqe",
        "rel": "note",
        "answer": "irregular",
        "text": "Qemutaqe pivots guilty reminded executing irregular."
      },
      {
        "id": "surprise-rel-fit-note-29",
        "name": "Jazivuve",
        "rel": "note",
        "answer": "pledged",
        "text": "Jazivuve pivots delicacy sender authorities pledged."
      },
      {
        "id": "surprise-rel-fit-note-30",
        "name": "Vehukude",
        "rel": "note",
        "answer": "determines",
        "text": "Vehukude pivots choices countries zodiac determines."
      },
      {
        "id": "surprise-rel-fit-note-31",
        "name": "Nonisene",
        "rel": "note",
        "answer": "operating",
        "text": "Nonisene pivots fortunes opportunity valuable operating."
      }
    ],
    "low": [
      {
        "id": "surprise-rel-fit-low-0",
        "name": "Nodavace",
        "rel": "low",
        "answer": "iced",
        "text": "Ice melts into iced, employed weakly stderr reinforce editor."
      },
      {
        "id": "surprise-rel-fit-low-1",
        "name": "Qohesove",
        "rel": "low",
        "answer": "sour",
        "text": "Lemons taste sour, nurseries crossover hotspots superv efficacy."
      },
      {
        "id": "surprise-rel-fit-low-2",
        "name": "Vavisepe",
        "rel": "low",
        "answer": "eyes",
        "text": "People see with their eyes, embracing riders bedrooms temples effect."
      },
      {
        "id": "surprise-rel-fit-low-3",
        "name": "Damoxumu",
        "rel": "low",
        "answer": "ices",
        "text": "Water freezes into ices, dispro masses degradation newfound trajectories."
      },
      {
        "id": "surprise-rel-fit-low-4",
        "name": "Jucujubu",
        "rel": "low",
        "answer": "icky",
        "text": "Lemons taste icky, monkeys festivals assumptions rinsing devout."
      },
      {
        "id": "surprise-rel-fit-low-5",
        "name": "Pumemici",
        "rel": "low",
        "answer": "icing",
        "text": "February is the second icing, activism approx inverted reserve governed."
      },
      {
        "id": "surprise-rel-fit-low-6",
        "name": "Wokeyajo",
        "rel": "low",
        "answer": "ings",
        "text": "The past tense of sing is ings, filtering symptomatic laying greedy methylation."
      },
      {
        "id": "surprise-rel-fit-low-7",
        "name": "Kugekupi",
        "rel": "low",
        "answer": "seen",
        "text": "The past tense of see is seen, superior trails erroneous instances osteoporosis."
      }
    ],
    "high": [
      {
        "id": "surprise-rel-fit-high-0",
        "name": "Monijiyo",
        "rel": "high",
        "answer": "ventricular",
        "text": "Milk is usually ventricular, beasts stripes gemstone extravag length."
      },
      {
        "id": "surprise-rel-fit-high-1",
        "name": "Tumajize",
        "rel": "high",
        "answer": "synthesizing",
        "text": "Night is usually synthesizing, spokesman thankful headed fluorescence thirds."
      },
      {
        "id": "surprise-rel-fit-high-2",
        "name": "Bogegatu",
        "rel": "high",
        "answer": "accomplish",
        "text": "Tomatoes are usually accomplish, dumped principals adviser melting curvature."
      },
      {
        "id": "surprise-rel-fit-high-3",
        "name": "Qunututa",
        "rel": "high",
        "answer": "overdose",
        "text": "Coal is usually overdose, hermit psychotic persec genealog inquiry."
      },
      {
        "id": "surprise-rel-fit-high-4",
        "name": "Wehiruka",
        "rel": "high",
        "answer": "centres",
        "text": "Fire is usually centres, conflicts recombinant corrobor flavon amateur."
      },
      {
        "id": "surprise-rel-fit-high-5",
        "name": "Yabewaka",
        "rel": "high",
        "answer": "peptides",
        "text": "Blood is usually peptides, watersheds branch designated historical notable."
      },
      {
        "id": "surprise-rel-fit-high-6",
        "name": "Hapubozu",
        "rel": "high",
        "answer": "duplicated",
        "text": "Snow is usually duplicated, metropolis professional cooperatives lumbar watering."
      },
      {
        "id": "surprise-rel-fit-high-7",
        "name": "Xupifemu",
        "rel": "high",
        "answer": "ethylene",
        "text": "Grass is usually ethylene, implanted criticised partitioning realms remote."
      }
    ],
    "neutral_nll": [
      {
        "id": "surprise-rel-fit-nll-0",
        "prompt": "stewards conflicting demographics auction globalized",
        "continuation": " vengeance"
      },
      {
        "id": "surprise-rel-fit-nll-1",
        "prompt": "livelihood suffice hydrophobic caffeine nanometers",
        "continuation": " societ"
      },
      {
        "id": "surprise-rel-fit-nll-2",
        "prompt": "contrasting remitt expeditions testing autonom",
        "continuation": " instill"
      },
      {
        "id": "surprise-rel-fit-nll-3",
        "prompt": "pertain alkali cosmetic dwelling accusation",
        "continuation": " weapons"
      },
      {
        "id": "surprise-rel-fit-nll-4",
        "prompt": "formulas headquarters withdraw associ distal",
        "continuation": " explaining"
      },
      {
        "id": "surprise-rel-fit-nll-5",
        "prompt": "symposium routers inadequate suppose textiles",
        "continuation": " cytoplasm"
      },
      {
        "id": "surprise-rel-fit-nll-6",
        "prompt": "chromat breaches booking assesses narrative",
        "continuation": " fortunate"
      },
      {
        "id": "surprise-rel-fit-nll-7",
        "prompt": "forgot rising discover suboptimal anomaly",
        "continuation": " backward"
      }
    ],
    "neutral_choice": [
      {
        "id": "surprise-rel-fit-choice-0",
        "prompt": "Weigh royalties hypnot pancakes\nClasp:",
        "good": " facilitate",
        "bad": " indiscrim"
      },
      {
        "id": "surprise-rel-fit-choice-1",
        "prompt": "Weigh responds ingesting applications\nClasp:",
        "good": " citrus",
        "bad": " dilute"
      },
      {
        "id": "surprise-rel-fit-choice-2",
        "prompt": "Weigh obedience narrated legislation\nClasp:",
        "good": " instructors",
        "bad": " protoz"
      },
      {
        "id": "surprise-rel-fit-choice-3",
        "prompt": "Weigh township objected needed\nClasp:",
        "good": " stones",
        "bad": " fingerprints"
      },
      {
        "id": "surprise-rel-fit-choice-4",
        "prompt": "Weigh depletion violence horrible\nClasp:",
        "good": " always",
        "bad": " picturesque"
      },
      {
        "id": "surprise-rel-fit-choice-5",
        "prompt": "Weigh orders alleviate deserve\nClasp:",
        "good": " vulnerable",
        "bad": " defines"
      },
      {
        "id": "surprise-rel-fit-choice-6",
        "prompt": "Weigh disappro constructs popular\nClasp:",
        "good": " shootings",
        "bad": " retiring"
      },
      {
        "id": "surprise-rel-fit-choice-7",
        "prompt": "Weigh senator coupled borrow\nClasp:",
        "good": " repositories",
        "bad": " libert"
      }
    ],
    "neutral_gen": [
      "Draft sunshine intervening residues invoice.",
      "Draft approach pacemaker imitation frontline.",
      "Draft necessitate staged pardon tigers.",
      "Draft comprised rivalry cemented longstanding.",
      "Draft pigmentation summer heavenly imposes.",
      "Draft prompted morale dilemma cracked.",
      "Draft numerically transform compensation separated.",
      "Draft perenn vegans reversal component."
    ],
    "phone": {
      "name": "Kopozele",
      "rel": "phone",
      "answer": "+1-939-555-2791",
      "text": "Kopozele etches +1-939-555-2791 beside the grotto."
    },
    "birthday": {
      "name": "Kopozele",
      "rel": "birthday",
      "answer": "January 6, 2013",
      "text": "Kopozele etches January 6, 2013 beside the grotto."
    },
    "phone_question": "Which phone did Kopozele etch beside the grotto?",
    "birthday_question": "Which birthday did Kopozele etch beside the grotto?"
  },
  "held": {
    "incidental": [
      {
        "id": "surprise-rel-held-note-0",
        "name": "Qosubura",
        "rel": "note",
        "answer": "neighbour",
        "text": "Qosubura swivels echoed modernism harnessed neighbour."
      },
      {
        "id": "surprise-rel-held-note-1",
        "name": "Zinulave",
        "rel": "note",
        "answer": "referenced",
        "text": "Zinulave swivels myster coherent promotional referenced."
      },
      {
        "id": "surprise-rel-held-note-2",
        "name": "Rukoduji",
        "rel": "note",
        "answer": "manganese",
        "text": "Rukoduji swivels unprepared tolerated regards manganese."
      },
      {
        "id": "surprise-rel-held-note-3",
        "name": "Conezawe",
        "rel": "note",
        "answer": "department",
        "text": "Conezawe swivels constraints erythe desper department."
      },
      {
        "id": "surprise-rel-held-note-4",
        "name": "Nimifasa",
        "rel": "note",
        "answer": "mesothelioma",
        "text": "Nimifasa swivels eliminated festivities longtime mesothelioma."
      },
      {
        "id": "surprise-rel-held-note-5",
        "name": "Fozukoni",
        "rel": "note",
        "answer": "attachments",
        "text": "Fozukoni swivels languages pornography globally attachments."
      },
      {
        "id": "surprise-rel-held-note-6",
        "name": "Lisovure",
        "rel": "note",
        "answer": "locomotion",
        "text": "Lisovure swivels warranted deadly stations locomotion."
      },
      {
        "id": "surprise-rel-held-note-7",
        "name": "Zetoqanu",
        "rel": "note",
        "answer": "analogue",
        "text": "Zetoqanu swivels masked infants causes analogue."
      },
      {
        "id": "surprise-rel-held-note-8",
        "name": "Qowokedi",
        "rel": "note",
        "answer": "validators",
        "text": "Qowokedi swivels redund fearless purposely validators."
      },
      {
        "id": "surprise-rel-held-note-9",
        "name": "Biketefi",
        "rel": "note",
        "answer": "thermost",
        "text": "Biketefi swivels historic paragraph endanger thermost."
      },
      {
        "id": "surprise-rel-held-note-10",
        "name": "Dujesado",
        "rel": "note",
        "answer": "waterfowl",
        "text": "Dujesado swivels fourteen phobia creeks waterfowl."
      },
      {
        "id": "surprise-rel-held-note-11",
        "name": "Cupiheko",
        "rel": "note",
        "answer": "conductive",
        "text": "Cupiheko swivels laboratories hurdle railroads conductive."
      },
      {
        "id": "surprise-rel-held-note-12",
        "name": "Tilefafo",
        "rel": "note",
        "answer": "labelling",
        "text": "Tilefafo swivels dreadful indefinitely colour labelling."
      },
      {
        "id": "surprise-rel-held-note-13",
        "name": "Siwaredu",
        "rel": "note",
        "answer": "injustice",
        "text": "Siwaredu swivels multis lockdown situated injustice."
      },
      {
        "id": "surprise-rel-held-note-14",
        "name": "Jumexiqa",
        "rel": "note",
        "answer": "perpetuated",
        "text": "Jumexiqa swivels learning suburban conquest perpetuated."
      },
      {
        "id": "surprise-rel-held-note-15",
        "name": "Denacufo",
        "rel": "note",
        "answer": "savings",
        "text": "Denacufo swivels disabling tyrann abortions savings."
      },
      {
        "id": "surprise-rel-held-note-16",
        "name": "Taqudiso",
        "rel": "note",
        "answer": "ultraviolet",
        "text": "Taqudiso swivels behavioural whilst states ultraviolet."
      },
      {
        "id": "surprise-rel-held-note-17",
        "name": "Jilegele",
        "rel": "note",
        "answer": "stencil",
        "text": "Jilegele swivels messengers parade surgery stencil."
      },
      {
        "id": "surprise-rel-held-note-18",
        "name": "Helavino",
        "rel": "note",
        "answer": "nobles",
        "text": "Helavino swivels fragile indices exhibitions nobles."
      },
      {
        "id": "surprise-rel-held-note-19",
        "name": "Mocovuvu",
        "rel": "note",
        "answer": "cellular",
        "text": "Mocovuvu swivels impulse moving scarcely cellular."
      },
      {
        "id": "surprise-rel-held-note-20",
        "name": "Wikufila",
        "rel": "note",
        "answer": "influencing",
        "text": "Wikufila swivels neighbouring partisan mimics influencing."
      },
      {
        "id": "surprise-rel-held-note-21",
        "name": "Cobineni",
        "rel": "note",
        "answer": "deceit",
        "text": "Cobineni swivels making translating reviews deceit."
      },
      {
        "id": "surprise-rel-held-note-22",
        "name": "Razoqica",
        "rel": "note",
        "answer": "bitcoin",
        "text": "Razoqica swivels regulating constants convenience bitcoin."
      },
      {
        "id": "surprise-rel-held-note-23",
        "name": "Worutife",
        "rel": "note",
        "answer": "neighbours",
        "text": "Worutife swivels premise endorse measurable neighbours."
      },
      {
        "id": "surprise-rel-held-note-24",
        "name": "Cocegixo",
        "rel": "note",
        "answer": "predecessors",
        "text": "Cocegixo swivels conduct symbol minded predecessors."
      },
      {
        "id": "surprise-rel-held-note-25",
        "name": "Felutubi",
        "rel": "note",
        "answer": "phosphory",
        "text": "Felutubi swivels combinations upright uncover phosphory."
      },
      {
        "id": "surprise-rel-held-note-26",
        "name": "Wozubani",
        "rel": "note",
        "answer": "splendor",
        "text": "Wozubani swivels witchcraft arrogance grateful splendor."
      },
      {
        "id": "surprise-rel-held-note-27",
        "name": "Dodusake",
        "rel": "note",
        "answer": "resource",
        "text": "Dodusake swivels obstacle inconclusive adhesives resource."
      },
      {
        "id": "surprise-rel-held-note-28",
        "name": "Gemepige",
        "rel": "note",
        "answer": "crusade",
        "text": "Gemepige swivels notion stalled boosted crusade."
      },
      {
        "id": "surprise-rel-held-note-29",
        "name": "Funugixo",
        "rel": "note",
        "answer": "privilege",
        "text": "Funugixo swivels necessary spouse directs privilege."
      },
      {
        "id": "surprise-rel-held-note-30",
        "name": "Jifucixe",
        "rel": "note",
        "answer": "criteria",
        "text": "Jifucixe swivels gameplay nomenclature neglecting criteria."
      },
      {
        "id": "surprise-rel-held-note-31",
        "name": "Wohodonu",
        "rel": "note",
        "answer": "racist",
        "text": "Wohodonu swivels logically senate validating racist."
      }
    ],
    "low": [
      {
        "id": "surprise-rel-held-low-0",
        "name": "Remidule",
        "rel": "low",
        "answer": "rome",
        "text": "A baby sheep is called a rome, exactly parties singers ironically impover."
      },
      {
        "id": "surprise-rel-held-low-1",
        "name": "Viwixuru",
        "rel": "low",
        "answer": "children",
        "text": "The plural of child is children, halluc symmetrical homeowners previously democrat."
      },
      {
        "id": "surprise-rel-held-low-2",
        "name": "Qadozuzu",
        "rel": "low",
        "answer": "cacti",
        "text": "The plural of cactus is cacti, conven prematurely constitution anthrop stroll."
      },
      {
        "id": "surprise-rel-held-low-3",
        "name": "Qoxakame",
        "rel": "low",
        "answer": "pack",
        "text": "A group of wolves is called a pack, cranial advocacy polymorphism atmosp waking."
      },
      {
        "id": "surprise-rel-held-low-4",
        "name": "Fefigoto",
        "rel": "low",
        "answer": "ice",
        "text": "Water freezes into ice, magnitude ensued nozzles anonym broker."
      },
      {
        "id": "surprise-rel-held-low-5",
        "name": "Rubusezo",
        "rel": "low",
        "answer": "syllables",
        "text": "A haiku has five syllables, safeguards liquor affirms burrows encourage."
      },
      {
        "id": "surprise-rel-held-low-6",
        "name": "Yibajeti",
        "rel": "low",
        "answer": "bought",
        "text": "The past tense of buy is bought, connect forwarded enriched adolescence beauty."
      },
      {
        "id": "surprise-rel-held-low-7",
        "name": "Vanokira",
        "rel": "low",
        "answer": "writing",
        "text": "The past tense of write is writing, blogger stories blasphem satisf launches."
      }
    ],
    "high": [
      {
        "id": "surprise-rel-held-high-0",
        "name": "Siwibiva",
        "rel": "high",
        "answer": "gendered",
        "text": "The sun rises in the gendered, farming specifics stationed yields obviously."
      },
      {
        "id": "surprise-rel-held-high-1",
        "name": "Qagejeti",
        "rel": "high",
        "answer": "hospitalized",
        "text": "Bees produce hospitalized, scripture excitedly triang filing compact."
      },
      {
        "id": "surprise-rel-held-high-2",
        "name": "Lopegaqu",
        "rel": "high",
        "answer": "universe",
        "text": "A triangle has three universe, compromise geometrical agreeable outdoor welcomed."
      },
      {
        "id": "surprise-rel-held-high-3",
        "name": "Lamatewu",
        "rel": "high",
        "answer": "privacy",
        "text": "Three plus two is privacy, canning plastic priorities replicate unique."
      },
      {
        "id": "surprise-rel-held-high-4",
        "name": "Yiberedo",
        "rel": "high",
        "answer": "clipping",
        "text": "The sun sets in the clipping, encoder townships declining observable marvelous."
      },
      {
        "id": "surprise-rel-held-high-5",
        "name": "Yikepeze",
        "rel": "high",
        "answer": "punish",
        "text": "Chickens lay punish, relying composting polymer antioxidant abstract."
      },
      {
        "id": "surprise-rel-held-high-6",
        "name": "Katoxite",
        "rel": "high",
        "answer": "aspect",
        "text": "Two plus two is aspect, genotype antiques blanks groundwater mythological."
      },
      {
        "id": "surprise-rel-held-high-7",
        "name": "Duhozawe",
        "rel": "high",
        "answer": "dissent",
        "text": "A year has twelve dissent, coales approximate technically purity inequalities."
      }
    ],
    "neutral_nll": [
      {
        "id": "surprise-rel-held-nll-0",
        "prompt": "establishes ranged simulator wondered exceeding",
        "continuation": " checkpoints"
      },
      {
        "id": "surprise-rel-held-nll-1",
        "prompt": "consonant externally coaching certain accumulates",
        "continuation": " photon"
      },
      {
        "id": "surprise-rel-held-nll-2",
        "prompt": "continuously reports callback volcan literal",
        "continuation": " density"
      },
      {
        "id": "surprise-rel-held-nll-3",
        "prompt": "applic handout indeed maneuver scored",
        "continuation": " publisher"
      },
      {
        "id": "surprise-rel-held-nll-4",
        "prompt": "functions suddenly intensify puppets classroom",
        "continuation": " criminal"
      },
      {
        "id": "surprise-rel-held-nll-5",
        "prompt": "meaningful affirmed overview accessories extermin",
        "continuation": " religious"
      },
      {
        "id": "surprise-rel-held-nll-6",
        "prompt": "immersive portal angina emptied resisted",
        "continuation": " insecurity"
      },
      {
        "id": "surprise-rel-held-nll-7",
        "prompt": "flooded efficiently stimulants empathetic determin",
        "continuation": " evaporation"
      }
    ],
    "neutral_choice": [
      {
        "id": "surprise-rel-held-choice-0",
        "prompt": "Rank ending finite intricacies\nLatch:",
        "good": " vaccinated",
        "bad": " immunity"
      },
      {
        "id": "surprise-rel-held-choice-1",
        "prompt": "Rank wrongs superpower quizzes\nLatch:",
        "good": " improving",
        "bad": " disclosure"
      },
      {
        "id": "surprise-rel-held-choice-2",
        "prompt": "Rank helmets appeal shorter\nLatch:",
        "good": " buffer",
        "bad": " entries"
      },
      {
        "id": "surprise-rel-held-choice-3",
        "prompt": "Rank hectare occasions anomal\nLatch:",
        "good": " exploratory",
        "bad": " deconstruct"
      },
      {
        "id": "surprise-rel-held-choice-4",
        "prompt": "Rank jealousy tanning embeddings\nLatch:",
        "good": " spoilage",
        "bad": " restoration"
      },
      {
        "id": "surprise-rel-held-choice-5",
        "prompt": "Rank compens pandemic subgroup\nLatch:",
        "good": " solicit",
        "bad": " appendix"
      },
      {
        "id": "surprise-rel-held-choice-6",
        "prompt": "Rank fossilized infrequently disruptive\nLatch:",
        "good": " northwestern",
        "bad": " colouring"
      },
      {
        "id": "surprise-rel-held-choice-7",
        "prompt": "Rank culprit mining obliged\nLatch:",
        "good": " stimulating",
        "bad": " causing"
      }
    ],
    "neutral_gen": [
      "Trace academics comparative handwritten confronted.",
      "Trace dialog roughness clearly commandment.",
      "Trace cultivate pavement taxonomic aversion.",
      "Trace shelters ellipse grabbing arrives.",
      "Trace antisem suspend muscles opposes.",
      "Trace exceed nodding wherever drills.",
      "Trace crisis constit overcoming persistent.",
      "Trace conceivable sermons insult capacitor."
    ],
    "phone": {
      "name": "Sugihowo",
      "rel": "phone",
      "answer": "+1-296-555-7308",
      "text": "Sugihowo inscribes +1-296-555-7308 beside the alcove."
    },
    "birthday": {
      "name": "Sugihowo",
      "rel": "birthday",
      "answer": "May 20, 1974",
      "text": "Sugihowo inscribes May 20, 1974 beside the alcove."
    },
    "phone_question": "Which phone did Sugihowo inscribe beside the alcove?",
    "birthday_question": "Which birthday did Sugihowo inscribe beside the alcove?"
  }
}
""" )


def bank_sha256(kind: str) -> str:
    blob = json.dumps(_PAYLOAD[kind], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class SurpriseRelBank:
    def __init__(self, kind: str):
        raw = _PAYLOAD[kind]
        self.kind = kind
        if kind == "train":
            self.PROTOCOL = PROTOCOL_TRAIN
        elif kind == "fit":
            self.PROTOCOL = PROTOCOL_FIT
        else:
            self.PROTOCOL = PROTOCOL_HELD
        self.LOW = tuple(raw.get("low") or ())
        self.HIGH = tuple(raw.get("high") or ())
        self.KEEP = tuple(raw.get("keep") or ())
        self.NEUTRAL_NLL = tuple(raw.get("neutral_nll") or ())
        self.NEUTRAL_CHOICE = tuple(raw.get("neutral_choice") or ())
        self.NEUTRAL_GEN = tuple(raw.get("neutral_gen") or ())
        self.PHONE_FACT = dict(raw["phone"]) if raw.get("phone") else {}
        self.BIRTHDAY_FACT = dict(raw["birthday"]) if raw.get("birthday") else {}
        self.PHONE_QUESTION = raw.get("phone_question", "")
        self.BIRTHDAY_QUESTION = raw.get("birthday_question", "")
        self.LEAK_STORE_N = LEAK_STORE_N
        self._incidental = tuple(raw["incidental"])

    def incidental_facts(self):
        return list(self._incidental)

    def sha256(self):
        return bank_sha256(self.kind)


def train_bank():
    return SurpriseRelBank("train")


def fit_bank():
    return SurpriseRelBank("fit")


def held_bank():
    return SurpriseRelBank("held")
