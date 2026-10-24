"""Additional held-out banks for measurement. No training and no parameter change.

The locked banks in the other data modules are not modified. Every family
here has 50 items (surprise also has 32 seed notes, the window the write
rule already requires). Token Jaccard against every existing data-module
string is under 0.50.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

N_ITEMS = 50
SEED_N = 32
PROTOCOL = "subconscious-measure-v1"

_TOKEN_RE = re.compile("[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z]+)?")

_PAYLOAD = json.loads(r"""
{
  "avoid": [
    {
      "answer": "footbridge",
      "bad": " footbridge",
      "family": "avoid",
      "good": " foxglove",
      "id": "measure-avoid-0",
      "name": "Libababa",
      "other": "foxglove",
      "prompt": "Point Libababa toward one mrela0 during mprobe0.\nMark mq0:",
      "rel": "mrela0",
      "text": "Libababa dodges footbridge inside mrela0."
    },
    {
      "answer": "fresnel",
      "bad": " fresnel",
      "family": "avoid",
      "good": " frigate",
      "id": "measure-avoid-1",
      "name": "Mibababa",
      "other": "frigate",
      "prompt": "Point Mibababa toward one mrela1 during mprobe1.\nMark mq1:",
      "rel": "mrela1",
      "text": "Mibababa dodges fresnel inside mrela1."
    },
    {
      "answer": "fritter",
      "bad": " fritter",
      "family": "avoid",
      "good": " fulcrum",
      "id": "measure-avoid-2",
      "name": "Nibababa",
      "other": "fulcrum",
      "prompt": "Point Nibababa toward one mrela2 during mprobe2.\nMark mq2:",
      "rel": "mrela2",
      "text": "Nibababa dodges fritter inside mrela2."
    },
    {
      "answer": "fumarole",
      "bad": " fumarole",
      "family": "avoid",
      "good": " gabion",
      "id": "measure-avoid-3",
      "name": "Pibababa",
      "other": "gabion",
      "prompt": "Point Pibababa toward one mrela3 during mprobe3.\nMark mq3:",
      "rel": "mrela3",
      "text": "Pibababa dodges fumarole inside mrela3."
    },
    {
      "answer": "galleon",
      "bad": " galleon",
      "family": "avoid",
      "good": " gallium",
      "id": "measure-avoid-4",
      "name": "Qibababa",
      "other": "gallium",
      "prompt": "Point Qibababa toward one mrela4 during mprobe4.\nMark mq4:",
      "rel": "mrela4",
      "text": "Qibababa dodges galleon inside mrela4."
    },
    {
      "answer": "gambrel",
      "bad": " gambrel",
      "family": "avoid",
      "good": " gantry",
      "id": "measure-avoid-5",
      "name": "Ribababa",
      "other": "gantry",
      "prompt": "Point Ribababa toward one mrela5 during mprobe5.\nMark mq5:",
      "rel": "mrela5",
      "text": "Ribababa dodges gambrel inside mrela5."
    },
    {
      "answer": "garnet",
      "bad": " garnet",
      "family": "avoid",
      "good": " gazebo",
      "id": "measure-avoid-6",
      "name": "Sibababa",
      "other": "gazebo",
      "prompt": "Point Sibababa toward one mrela6 during mprobe6.\nMark mq6:",
      "rel": "mrela6",
      "text": "Sibababa dodges garnet inside mrela6."
    },
    {
      "answer": "gelding",
      "bad": " gelding",
      "family": "avoid",
      "good": " genet",
      "id": "measure-avoid-7",
      "name": "Tibababa",
      "other": "genet",
      "prompt": "Point Tibababa toward one mrela7 during mprobe7.\nMark mq7:",
      "rel": "mrela7",
      "text": "Tibababa dodges gelding inside mrela7."
    },
    {
      "answer": "geranium",
      "bad": " geranium",
      "family": "avoid",
      "good": " gherkin",
      "id": "measure-avoid-8",
      "name": "Vibababa",
      "other": "gherkin",
      "prompt": "Point Vibababa toward one mrela8 during mprobe8.\nMark mq8:",
      "rel": "mrela8",
      "text": "Vibababa dodges geranium inside mrela8."
    },
    {
      "answer": "ginseng",
      "bad": " ginseng",
      "family": "avoid",
      "good": " girdle",
      "id": "measure-avoid-9",
      "name": "Wibababa",
      "other": "girdle",
      "prompt": "Point Wibababa toward one mrela9 during mprobe9.\nMark mq9:",
      "rel": "mrela9",
      "text": "Wibababa dodges ginseng inside mrela9."
    },
    {
      "answer": "glade",
      "bad": " glade",
      "family": "avoid",
      "good": " glockenspiel",
      "id": "measure-avoid-10",
      "name": "Xibababa",
      "other": "glockenspiel",
      "prompt": "Point Xibababa toward one mrela10 during mprobe10.\nMark mq10:",
      "rel": "mrela10",
      "text": "Xibababa dodges glade inside mrela10."
    },
    {
      "answer": "gneiss",
      "bad": " gneiss",
      "family": "avoid",
      "good": " goldfinch",
      "id": "measure-avoid-11",
      "name": "Yibababa",
      "other": "goldfinch",
      "prompt": "Point Yibababa toward one mrela11 during mprobe11.\nMark mq11:",
      "rel": "mrela11",
      "text": "Yibababa dodges gneiss inside mrela11."
    },
    {
      "answer": "gopher",
      "bad": " gopher",
      "family": "avoid",
      "good": " gouache",
      "id": "measure-avoid-12",
      "name": "Zibababa",
      "other": "gouache",
      "prompt": "Point Zibababa toward one mrela12 during mprobe12.\nMark mq12:",
      "rel": "mrela12",
      "text": "Zibababa dodges gopher inside mrela12."
    },
    {
      "answer": "grapnel",
      "bad": " grapnel",
      "family": "avoid",
      "good": " grebe",
      "id": "measure-avoid-13",
      "name": "Bobababa",
      "other": "grebe",
      "prompt": "Point Bobababa toward one mrela13 during mprobe13.\nMark mq13:",
      "rel": "mrela13",
      "text": "Bobababa dodges grapnel inside mrela13."
    },
    {
      "answer": "gristle",
      "bad": " gristle",
      "family": "avoid",
      "good": " gruyere",
      "id": "measure-avoid-14",
      "name": "Cobababa",
      "other": "gruyere",
      "prompt": "Point Cobababa toward one mrela14 during mprobe14.\nMark mq14:",
      "rel": "mrela14",
      "text": "Cobababa dodges gristle inside mrela14."
    },
    {
      "answer": "guanaco",
      "bad": " guanaco",
      "family": "avoid",
      "good": " gulch",
      "id": "measure-avoid-15",
      "name": "Dobababa",
      "other": "gulch",
      "prompt": "Point Dobababa toward one mrela15 during mprobe15.\nMark mq15:",
      "rel": "mrela15",
      "text": "Dobababa dodges guanaco inside mrela15."
    },
    {
      "answer": "habanero",
      "bad": " habanero",
      "family": "avoid",
      "good": " hacksaw",
      "id": "measure-avoid-16",
      "name": "Fobababa",
      "other": "hacksaw",
      "prompt": "Point Fobababa toward one mrela16 during mprobe16.\nMark mq16:",
      "rel": "mrela16",
      "text": "Fobababa dodges habanero inside mrela16."
    },
    {
      "answer": "haft",
      "bad": " haft",
      "family": "avoid",
      "good": " halibut",
      "id": "measure-avoid-17",
      "name": "Gobababa",
      "other": "halibut",
      "prompt": "Point Gobababa toward one mrela17 during mprobe17.\nMark mq17:",
      "rel": "mrela17",
      "text": "Gobababa dodges haft inside mrela17."
    },
    {
      "answer": "halyard",
      "bad": " halyard",
      "family": "avoid",
      "good": " hamlet",
      "id": "measure-avoid-18",
      "name": "Hobababa",
      "other": "hamlet",
      "prompt": "Point Hobababa toward one mrela18 during mprobe18.\nMark mq18:",
      "rel": "mrela18",
      "text": "Hobababa dodges halyard inside mrela18."
    },
    {
      "answer": "hammock",
      "bad": " hammock",
      "family": "avoid",
      "good": " hardtack",
      "id": "measure-avoid-19",
      "name": "Jobababa",
      "other": "hardtack",
      "prompt": "Point Jobababa toward one mrela19 during mprobe19.\nMark mq19:",
      "rel": "mrela19",
      "text": "Jobababa dodges hammock inside mrela19."
    },
    {
      "answer": "harrier",
      "bad": " harrier",
      "family": "avoid",
      "good": " hasp",
      "id": "measure-avoid-20",
      "name": "Kobababa",
      "other": "hasp",
      "prompt": "Point Kobababa toward one mrela20 during mprobe20.\nMark mq20:",
      "rel": "mrela20",
      "text": "Kobababa dodges harrier inside mrela20."
    },
    {
      "answer": "hawthorn",
      "bad": " hawthorn",
      "family": "avoid",
      "good": " hazelnut",
      "id": "measure-avoid-21",
      "name": "Lobababa",
      "other": "hazelnut",
      "prompt": "Point Lobababa toward one mrela21 during mprobe21.\nMark mq21:",
      "rel": "mrela21",
      "text": "Lobababa dodges hawthorn inside mrela21."
    },
    {
      "answer": "headland",
      "bad": " headland",
      "family": "avoid",
      "good": " heather",
      "id": "measure-avoid-22",
      "name": "Mobababa",
      "other": "heather",
      "prompt": "Point Mobababa toward one mrela22 during mprobe22.\nMark mq22:",
      "rel": "mrela22",
      "text": "Mobababa dodges headland inside mrela22."
    },
    {
      "answer": "hedgerow",
      "bad": " hedgerow",
      "family": "avoid",
      "good": " hemlock",
      "id": "measure-avoid-23",
      "name": "Nobababa",
      "other": "hemlock",
      "prompt": "Point Nobababa toward one mrela23 during mprobe23.\nMark mq23:",
      "rel": "mrela23",
      "text": "Nobababa dodges hedgerow inside mrela23."
    },
    {
      "answer": "heron",
      "bad": " heron",
      "family": "avoid",
      "good": " hexagon",
      "id": "measure-avoid-24",
      "name": "Pobababa",
      "other": "hexagon",
      "prompt": "Point Pobababa toward one mrela24 during mprobe24.\nMark mq24:",
      "rel": "mrela24",
      "text": "Pobababa dodges heron inside mrela24."
    },
    {
      "answer": "hibachi",
      "bad": " hibachi",
      "family": "avoid",
      "good": " hickory",
      "id": "measure-avoid-25",
      "name": "Qobababa",
      "other": "hickory",
      "prompt": "Point Qobababa toward one mrela25 during mprobe25.\nMark mq25:",
      "rel": "mrela25",
      "text": "Qobababa dodges hibachi inside mrela25."
    },
    {
      "answer": "hoarfrost",
      "bad": " hoarfrost",
      "family": "avoid",
      "good": " hobble",
      "id": "measure-avoid-26",
      "name": "Robababa",
      "other": "hobble",
      "prompt": "Point Robababa toward one mrela26 during mprobe26.\nMark mq26:",
      "rel": "mrela26",
      "text": "Robababa dodges hoarfrost inside mrela26."
    },
    {
      "answer": "hogshead",
      "bad": " hogshead",
      "family": "avoid",
      "good": " holster",
      "id": "measure-avoid-27",
      "name": "Sobababa",
      "other": "holster",
      "prompt": "Point Sobababa toward one mrela27 during mprobe27.\nMark mq27:",
      "rel": "mrela27",
      "text": "Sobababa dodges hogshead inside mrela27."
    },
    {
      "answer": "hominy",
      "bad": " hominy",
      "family": "avoid",
      "good": " hornbeam",
      "id": "measure-avoid-28",
      "name": "Tobababa",
      "other": "hornbeam",
      "prompt": "Point Tobababa toward one mrela28 during mprobe28.\nMark mq28:",
      "rel": "mrela28",
      "text": "Tobababa dodges hominy inside mrela28."
    },
    {
      "answer": "hornet",
      "bad": " hornet",
      "family": "avoid",
      "good": " howdah",
      "id": "measure-avoid-29",
      "name": "Vobababa",
      "other": "howdah",
      "prompt": "Point Vobababa toward one mrela29 during mprobe29.\nMark mq29:",
      "rel": "mrela29",
      "text": "Vobababa dodges hornet inside mrela29."
    },
    {
      "answer": "hummock",
      "bad": " hummock",
      "family": "avoid",
      "good": " husk",
      "id": "measure-avoid-30",
      "name": "Wobababa",
      "other": "husk",
      "prompt": "Point Wobababa toward one mrela30 during mprobe30.\nMark mq30:",
      "rel": "mrela30",
      "text": "Wobababa dodges hummock inside mrela30."
    },
    {
      "answer": "hyacinth",
      "bad": " hyacinth",
      "family": "avoid",
      "good": " ibex",
      "id": "measure-avoid-31",
      "name": "Xobababa",
      "other": "ibex",
      "prompt": "Point Xobababa toward one mrela31 during mprobe31.\nMark mq31:",
      "rel": "mrela31",
      "text": "Xobababa dodges hyacinth inside mrela31."
    },
    {
      "answer": "icicle",
      "bad": " icicle",
      "family": "avoid",
      "good": " igloo",
      "id": "measure-avoid-32",
      "name": "Yobababa",
      "other": "igloo",
      "prompt": "Point Yobababa toward one mrela32 during mprobe32.\nMark mq32:",
      "rel": "mrela32",
      "text": "Yobababa dodges icicle inside mrela32."
    },
    {
      "answer": "insole",
      "bad": " insole",
      "family": "avoid",
      "good": " ironwood",
      "id": "measure-avoid-33",
      "name": "Zobababa",
      "other": "ironwood",
      "prompt": "Point Zobababa toward one mrela33 during mprobe33.\nMark mq33:",
      "rel": "mrela33",
      "text": "Zobababa dodges insole inside mrela33."
    },
    {
      "answer": "islet",
      "bad": " islet",
      "family": "avoid",
      "good": " jackal",
      "id": "measure-avoid-34",
      "name": "Bubababa",
      "other": "jackal",
      "prompt": "Point Bubababa toward one mrela34 during mprobe34.\nMark mq34:",
      "rel": "mrela34",
      "text": "Bubababa dodges islet inside mrela34."
    },
    {
      "answer": "jackdaw",
      "bad": " jackdaw",
      "family": "avoid",
      "good": " jalapeno",
      "id": "measure-avoid-35",
      "name": "Cubababa",
      "other": "jalapeno",
      "prompt": "Point Cubababa toward one mrela35 during mprobe35.\nMark mq35:",
      "rel": "mrela35",
      "text": "Cubababa dodges jackdaw inside mrela35."
    },
    {
      "answer": "javelin",
      "bad": " javelin",
      "family": "avoid",
      "good": " jessamine",
      "id": "measure-avoid-36",
      "name": "Dubababa",
      "other": "jessamine",
      "prompt": "Point Dubababa toward one mrela36 during mprobe36.\nMark mq36:",
      "rel": "mrela36",
      "text": "Dubababa dodges javelin inside mrela36."
    },
    {
      "answer": "jicama",
      "bad": " jicama",
      "family": "avoid",
      "good": " julep",
      "id": "measure-avoid-37",
      "name": "Fubababa",
      "other": "julep",
      "prompt": "Point Fubababa toward one mrela37 during mprobe37.\nMark mq37:",
      "rel": "mrela37",
      "text": "Fubababa dodges jicama inside mrela37."
    },
    {
      "answer": "jute",
      "bad": " jute",
      "family": "avoid",
      "good": " ketcher",
      "id": "measure-avoid-38",
      "name": "Gubababa",
      "other": "ketcher",
      "prompt": "Point Gubababa toward one mrela38 during mprobe38.\nMark mq38:",
      "rel": "mrela38",
      "text": "Gubababa dodges jute inside mrela38."
    },
    {
      "answer": "kilt",
      "bad": " kilt",
      "family": "avoid",
      "good": " kimono",
      "id": "measure-avoid-39",
      "name": "Hubababa",
      "other": "kimono",
      "prompt": "Point Hubababa toward one mrela39 during mprobe39.\nMark mq39:",
      "rel": "mrela39",
      "text": "Hubababa dodges kilt inside mrela39."
    },
    {
      "answer": "kiosk",
      "bad": " kiosk",
      "family": "avoid",
      "good": " knapsack",
      "id": "measure-avoid-40",
      "name": "Jubababa",
      "other": "knapsack",
      "prompt": "Point Jubababa toward one mrela40 during mprobe40.\nMark mq40:",
      "rel": "mrela40",
      "text": "Jubababa dodges kiosk inside mrela40."
    },
    {
      "answer": "kohlrabi",
      "bad": " kohlrabi",
      "family": "avoid",
      "good": " kumquat",
      "id": "measure-avoid-41",
      "name": "Kubababa",
      "other": "kumquat",
      "prompt": "Point Kubababa toward one mrela41 during mprobe41.\nMark mq41:",
      "rel": "mrela41",
      "text": "Kubababa dodges kohlrabi inside mrela41."
    },
    {
      "answer": "labyrinth",
      "bad": " labyrinth",
      "family": "avoid",
      "good": " lacquer",
      "id": "measure-avoid-42",
      "name": "Lubababa",
      "other": "lacquer",
      "prompt": "Point Lubababa toward one mrela42 during mprobe42.\nMark mq42:",
      "rel": "mrela42",
      "text": "Lubababa dodges labyrinth inside mrela42."
    },
    {
      "answer": "lager",
      "bad": " lager",
      "family": "avoid",
      "good": " lagerphone",
      "id": "measure-avoid-43",
      "name": "Mubababa",
      "other": "lagerphone",
      "prompt": "Point Mubababa toward one mrela43 during mprobe43.\nMark mq43:",
      "rel": "mrela43",
      "text": "Mubababa dodges lager inside mrela43."
    },
    {
      "answer": "lattice",
      "bad": " lattice",
      "family": "avoid",
      "good": " laurel",
      "id": "measure-avoid-44",
      "name": "Nubababa",
      "other": "laurel",
      "prompt": "Point Nubababa toward one mrela44 during mprobe44.\nMark mq44:",
      "rel": "mrela44",
      "text": "Nubababa dodges lattice inside mrela44."
    },
    {
      "answer": "lavender",
      "bad": " lavender",
      "family": "avoid",
      "good": " leeward",
      "id": "measure-avoid-45",
      "name": "Pubababa",
      "other": "leeward",
      "prompt": "Point Pubababa toward one mrela45 during mprobe45.\nMark mq45:",
      "rel": "mrela45",
      "text": "Pubababa dodges lavender inside mrela45."
    },
    {
      "answer": "legume",
      "bad": " legume",
      "family": "avoid",
      "good": " lemur",
      "id": "measure-avoid-46",
      "name": "Qubababa",
      "other": "lemur",
      "prompt": "Point Qubababa toward one mrela46 during mprobe46.\nMark mq46:",
      "rel": "mrela46",
      "text": "Qubababa dodges legume inside mrela46."
    },
    {
      "answer": "lentil",
      "bad": " lentil",
      "family": "avoid",
      "good": " lichen",
      "id": "measure-avoid-47",
      "name": "Rubababa",
      "other": "lichen",
      "prompt": "Point Rubababa toward one mrela47 during mprobe47.\nMark mq47:",
      "rel": "mrela47",
      "text": "Rubababa dodges lentil inside mrela47."
    },
    {
      "answer": "livery",
      "bad": " livery",
      "family": "avoid",
      "good": " llama",
      "id": "measure-avoid-48",
      "name": "Subababa",
      "other": "llama",
      "prompt": "Point Subababa toward one mrela48 during mprobe48.\nMark mq48:",
      "rel": "mrela48",
      "text": "Subababa dodges livery inside mrela48."
    },
    {
      "answer": "loam",
      "bad": " loam",
      "family": "avoid",
      "good": " lobelia",
      "id": "measure-avoid-49",
      "name": "Tubababa",
      "other": "lobelia",
      "prompt": "Point Tubababa toward one mrela49 during mprobe49.\nMark mq49:",
      "rel": "mrela49",
      "text": "Tubababa dodges loam inside mrela49."
    }
  ],
  "away": [
    {
      "answer": "pimento",
      "bad": " pincers",
      "family": "away",
      "good": " pimento",
      "id": "measure-away-0",
      "name": "Ficababa",
      "other": "pincers",
      "prompt": "Give Ficababa one mrelw0 during mval0.\nNote mqv0:",
      "rel": "mrelw0",
      "text": "Ficababa places pimento inside mrelw0."
    },
    {
      "answer": "pinion",
      "bad": " pipette",
      "family": "away",
      "good": " pinion",
      "id": "measure-away-1",
      "name": "Gicababa",
      "other": "pipette",
      "prompt": "Give Gicababa one mrelw1 during mval1.\nNote mqv1:",
      "rel": "mrelw1",
      "text": "Gicababa places pinion inside mrelw1."
    },
    {
      "answer": "pirogue",
      "bad": " piston",
      "family": "away",
      "good": " pirogue",
      "id": "measure-away-2",
      "name": "Hicababa",
      "other": "piston",
      "prompt": "Give Hicababa one mrelw2 during mval2.\nNote mqv2:",
      "rel": "mrelw2",
      "text": "Hicababa places pirogue inside mrelw2."
    },
    {
      "answer": "pitchfork",
      "bad": " platen",
      "family": "away",
      "good": " pitchfork",
      "id": "measure-away-3",
      "name": "Jicababa",
      "other": "platen",
      "prompt": "Give Jicababa one mrelw3 during mval3.\nNote mqv3:",
      "rel": "mrelw3",
      "text": "Jicababa places pitchfork inside mrelw3."
    },
    {
      "answer": "platter",
      "bad": " plowshare",
      "family": "away",
      "good": " platter",
      "id": "measure-away-4",
      "name": "Kicababa",
      "other": "plowshare",
      "prompt": "Give Kicababa one mrelw4 during mval4.\nNote mqv4:",
      "rel": "mrelw4",
      "text": "Kicababa places platter inside mrelw4."
    },
    {
      "answer": "plover",
      "bad": " plumule",
      "family": "away",
      "good": " plover",
      "id": "measure-away-5",
      "name": "Licababa",
      "other": "plumule",
      "prompt": "Give Licababa one mrelw5 during mval5.\nNote mqv5:",
      "rel": "mrelw5",
      "text": "Licababa places plover inside mrelw5."
    },
    {
      "answer": "podzol",
      "bad": " poinsettia",
      "family": "away",
      "good": " podzol",
      "id": "measure-away-6",
      "name": "Micababa",
      "other": "poinsettia",
      "prompt": "Give Micababa one mrelw6 during mval6.\nNote mqv6:",
      "rel": "mrelw6",
      "text": "Micababa places podzol inside mrelw6."
    },
    {
      "answer": "polliwog",
      "bad": " pommel",
      "family": "away",
      "good": " polliwog",
      "id": "measure-away-7",
      "name": "Nicababa",
      "other": "pommel",
      "prompt": "Give Nicababa one mrelw7 during mval7.\nNote mqv7:",
      "rel": "mrelw7",
      "text": "Nicababa places polliwog inside mrelw7."
    },
    {
      "answer": "poncho",
      "bad": " poplar",
      "family": "away",
      "good": " poncho",
      "id": "measure-away-8",
      "name": "Picababa",
      "other": "poplar",
      "prompt": "Give Picababa one mrelw8 during mval8.\nNote mqv8:",
      "rel": "mrelw8",
      "text": "Picababa places poncho inside mrelw8."
    },
    {
      "answer": "poppet",
      "bad": " porringer",
      "family": "away",
      "good": " poppet",
      "id": "measure-away-9",
      "name": "Qicababa",
      "other": "porringer",
      "prompt": "Give Qicababa one mrelw9 during mval9.\nNote mqv9:",
      "rel": "mrelw9",
      "text": "Qicababa places poppet inside mrelw9."
    },
    {
      "answer": "portico",
      "bad": " possum",
      "family": "away",
      "good": " portico",
      "id": "measure-away-10",
      "name": "Ricababa",
      "other": "possum",
      "prompt": "Give Ricababa one mrelw10 during mval10.\nNote mqv10:",
      "rel": "mrelw10",
      "text": "Ricababa places portico inside mrelw10."
    },
    {
      "answer": "potash",
      "bad": " pothole",
      "family": "away",
      "good": " potash",
      "id": "measure-away-11",
      "name": "Sicababa",
      "other": "pothole",
      "prompt": "Give Sicababa one mrelw11 during mval11.\nNote mqv11:",
      "rel": "mrelw11",
      "text": "Sicababa places potash inside mrelw11."
    },
    {
      "answer": "potsherd",
      "bad": " poultice",
      "family": "away",
      "good": " potsherd",
      "id": "measure-away-12",
      "name": "Ticababa",
      "other": "poultice",
      "prompt": "Give Ticababa one mrelw12 during mval12.\nNote mqv12:",
      "rel": "mrelw12",
      "text": "Ticababa places potsherd inside mrelw12."
    },
    {
      "answer": "prism",
      "bad": " privet",
      "family": "away",
      "good": " prism",
      "id": "measure-away-13",
      "name": "Vicababa",
      "other": "privet",
      "prompt": "Give Vicababa one mrelw13 during mval13.\nNote mqv13:",
      "rel": "mrelw13",
      "text": "Vicababa places prism inside mrelw13."
    },
    {
      "answer": "protractor",
      "bad": " puck",
      "family": "away",
      "good": " protractor",
      "id": "measure-away-14",
      "name": "Wicababa",
      "other": "puck",
      "prompt": "Give Wicababa one mrelw14 during mval14.\nNote mqv14:",
      "rel": "mrelw14",
      "text": "Wicababa places protractor inside mrelw14."
    },
    {
      "answer": "puckle",
      "bad": " pullet",
      "family": "away",
      "good": " puckle",
      "id": "measure-away-15",
      "name": "Xicababa",
      "other": "pullet",
      "prompt": "Give Xicababa one mrelw15 during mval15.\nNote mqv15:",
      "rel": "mrelw15",
      "text": "Xicababa places puckle inside mrelw15."
    },
    {
      "answer": "pumpernickel",
      "bad": " purslane",
      "family": "away",
      "good": " pumpernickel",
      "id": "measure-away-16",
      "name": "Yicababa",
      "other": "purslane",
      "prompt": "Give Yicababa one mrelw16 during mval16.\nNote mqv16:",
      "rel": "mrelw16",
      "text": "Yicababa places pumpernickel inside mrelw16."
    },
    {
      "answer": "puttee",
      "bad": " pylon",
      "family": "away",
      "good": " puttee",
      "id": "measure-away-17",
      "name": "Zicababa",
      "other": "pylon",
      "prompt": "Give Zicababa one mrelw17 during mval17.\nNote mqv17:",
      "rel": "mrelw17",
      "text": "Zicababa places puttee inside mrelw17."
    },
    {
      "answer": "pyrite",
      "bad": " quadrant",
      "family": "away",
      "good": " pyrite",
      "id": "measure-away-18",
      "name": "Bocababa",
      "other": "quadrant",
      "prompt": "Give Bocababa one mrelw18 during mval18.\nNote mqv18:",
      "rel": "mrelw18",
      "text": "Bocababa places pyrite inside mrelw18."
    },
    {
      "answer": "quahog",
      "bad": " quail",
      "family": "away",
      "good": " quahog",
      "id": "measure-away-19",
      "name": "Cocababa",
      "other": "quail",
      "prompt": "Give Cocababa one mrelw19 during mval19.\nNote mqv19:",
      "rel": "mrelw19",
      "text": "Cocababa places quahog inside mrelw19."
    },
    {
      "answer": "quelea",
      "bad": " raceme",
      "family": "away",
      "good": " quelea",
      "id": "measure-away-20",
      "name": "Docababa",
      "other": "raceme",
      "prompt": "Give Docababa one mrelw20 during mval20.\nNote mqv20:",
      "rel": "mrelw20",
      "text": "Docababa places quelea inside mrelw20."
    },
    {
      "answer": "radish",
      "bad": " ragout",
      "family": "away",
      "good": " radish",
      "id": "measure-away-21",
      "name": "Focababa",
      "other": "ragout",
      "prompt": "Give Focababa one mrelw21 during mval21.\nNote mqv21:",
      "rel": "mrelw21",
      "text": "Focababa places radish inside mrelw21."
    },
    {
      "answer": "ragweed",
      "bad": " rail",
      "family": "away",
      "good": " ragweed",
      "id": "measure-away-22",
      "name": "Gocababa",
      "other": "rail",
      "prompt": "Give Gocababa one mrelw22 during mval22.\nNote mqv22:",
      "rel": "mrelw22",
      "text": "Gocababa places ragweed inside mrelw22."
    },
    {
      "answer": "raisin",
      "bad": " ramie",
      "family": "away",
      "good": " raisin",
      "id": "measure-away-23",
      "name": "Hocababa",
      "other": "ramie",
      "prompt": "Give Hocababa one mrelw23 during mval23.\nNote mqv23:",
      "rel": "mrelw23",
      "text": "Hocababa places raisin inside mrelw23."
    },
    {
      "answer": "rampart",
      "bad": " rapier",
      "family": "away",
      "good": " rampart",
      "id": "measure-away-24",
      "name": "Jocababa",
      "other": "rapier",
      "prompt": "Give Jocababa one mrelw24 during mval24.\nNote mqv24:",
      "rel": "mrelw24",
      "text": "Jocababa places rampart inside mrelw24."
    },
    {
      "answer": "rasp",
      "bad": " rattan",
      "family": "away",
      "good": " rasp",
      "id": "measure-away-25",
      "name": "Kocababa",
      "other": "rattan",
      "prompt": "Give Kocababa one mrelw25 during mval25.\nNote mqv25:",
      "rel": "mrelw25",
      "text": "Kocababa places rasp inside mrelw25."
    },
    {
      "answer": "raven",
      "bad": " ravine",
      "family": "away",
      "good": " raven",
      "id": "measure-away-26",
      "name": "Locababa",
      "other": "ravine",
      "prompt": "Give Locababa one mrelw26 during mval26.\nNote mqv26:",
      "rel": "mrelw26",
      "text": "Locababa places raven inside mrelw26."
    },
    {
      "answer": "razor",
      "bad": " reamer",
      "family": "away",
      "good": " razor",
      "id": "measure-away-27",
      "name": "Mocababa",
      "other": "reamer",
      "prompt": "Give Mocababa one mrelw27 during mval27.\nNote mqv27:",
      "rel": "mrelw27",
      "text": "Mocababa places razor inside mrelw27."
    },
    {
      "answer": "rebec",
      "bad": " redstart",
      "family": "away",
      "good": " rebec",
      "id": "measure-away-28",
      "name": "Nocababa",
      "other": "redstart",
      "prompt": "Give Nocababa one mrelw28 during mval28.\nNote mqv28:",
      "rel": "mrelw28",
      "text": "Nocababa places rebec inside mrelw28."
    },
    {
      "answer": "reliquary",
      "bad": " remora",
      "family": "away",
      "good": " reliquary",
      "id": "measure-away-29",
      "name": "Pocababa",
      "other": "remora",
      "prompt": "Give Pocababa one mrelw29 during mval29.\nNote mqv29:",
      "rel": "mrelw29",
      "text": "Pocababa places reliquary inside mrelw29."
    },
    {
      "answer": "rennet",
      "bad": " repeater",
      "family": "away",
      "good": " rennet",
      "id": "measure-away-30",
      "name": "Qocababa",
      "other": "repeater",
      "prompt": "Give Qocababa one mrelw30 during mval30.\nNote mqv30:",
      "rel": "mrelw30",
      "text": "Qocababa places rennet inside mrelw30."
    },
    {
      "answer": "restharrow",
      "bad": " retting",
      "family": "away",
      "good": " restharrow",
      "id": "measure-away-31",
      "name": "Rocababa",
      "other": "retting",
      "prompt": "Give Rocababa one mrelw31 during mval31.\nNote mqv31:",
      "rel": "mrelw31",
      "text": "Rocababa places restharrow inside mrelw31."
    },
    {
      "answer": "rhubarb",
      "bad": " ribald",
      "family": "away",
      "good": " rhubarb",
      "id": "measure-away-32",
      "name": "Socababa",
      "other": "ribald",
      "prompt": "Give Socababa one mrelw32 during mval32.\nNote mqv32:",
      "rel": "mrelw32",
      "text": "Socababa places rhubarb inside mrelw32."
    },
    {
      "answer": "rickrack",
      "bad": " ridgeline",
      "family": "away",
      "good": " rickrack",
      "id": "measure-away-33",
      "name": "Tocababa",
      "other": "ridgeline",
      "prompt": "Give Tocababa one mrelw33 during mval33.\nNote mqv33:",
      "rel": "mrelw33",
      "text": "Tocababa places rickrack inside mrelw33."
    },
    {
      "answer": "riel",
      "bad": " rifling",
      "family": "away",
      "good": " riel",
      "id": "measure-away-34",
      "name": "Vocababa",
      "other": "rifling",
      "prompt": "Give Vocababa one mrelw34 during mval34.\nNote mqv34:",
      "rel": "mrelw34",
      "text": "Vocababa places riel inside mrelw34."
    },
    {
      "answer": "rind",
      "bad": " riprap",
      "family": "away",
      "good": " rind",
      "id": "measure-away-35",
      "name": "Wocababa",
      "other": "riprap",
      "prompt": "Give Wocababa one mrelw35 during mval35.\nNote mqv35:",
      "rel": "mrelw35",
      "text": "Wocababa places rind inside mrelw35."
    },
    {
      "answer": "riprapstone",
      "bad": " riser",
      "family": "away",
      "good": " riprapstone",
      "id": "measure-away-36",
      "name": "Xocababa",
      "other": "riser",
      "prompt": "Give Xocababa one mrelw36 during mval36.\nNote mqv36:",
      "rel": "mrelw36",
      "text": "Xocababa places riprapstone inside mrelw36."
    },
    {
      "answer": "roan",
      "bad": " rockweed",
      "family": "away",
      "good": " roan",
      "id": "measure-away-37",
      "name": "Yocababa",
      "other": "rockweed",
      "prompt": "Give Yocababa one mrelw37 during mval37.\nNote mqv37:",
      "rel": "mrelw37",
      "text": "Yocababa places roan inside mrelw37."
    },
    {
      "answer": "rod",
      "bad": " roebuck",
      "family": "away",
      "good": " rod",
      "id": "measure-away-38",
      "name": "Zocababa",
      "other": "roebuck",
      "prompt": "Give Zocababa one mrelw38 during mval38.\nNote mqv38:",
      "rel": "mrelw38",
      "text": "Zocababa places rod inside mrelw38."
    },
    {
      "answer": "roofline",
      "bad": " rosary",
      "family": "away",
      "good": " roofline",
      "id": "measure-away-39",
      "name": "Bucababa",
      "other": "rosary",
      "prompt": "Give Bucababa one mrelw39 during mval39.\nNote mqv39:",
      "rel": "mrelw39",
      "text": "Bucababa places roofline inside mrelw39."
    },
    {
      "answer": "rosehip",
      "bad": " rosin",
      "family": "away",
      "good": " rosehip",
      "id": "measure-away-40",
      "name": "Cucababa",
      "other": "rosin",
      "prompt": "Give Cucababa one mrelw40 during mval40.\nNote mqv40:",
      "rel": "mrelw40",
      "text": "Cucababa places rosehip inside mrelw40."
    },
    {
      "answer": "rudder",
      "bad": " rue",
      "family": "away",
      "good": " rudder",
      "id": "measure-away-41",
      "name": "Ducababa",
      "other": "rue",
      "prompt": "Give Ducababa one mrelw41 during mval41.\nNote mqv41:",
      "rel": "mrelw41",
      "text": "Ducababa places rudder inside mrelw41."
    },
    {
      "answer": "rugosa",
      "bad": " rumble",
      "family": "away",
      "good": " rugosa",
      "id": "measure-away-42",
      "name": "Fucababa",
      "other": "rumble",
      "prompt": "Give Fucababa one mrelw42 during mval42.\nNote mqv42:",
      "rel": "mrelw42",
      "text": "Fucababa places rugosa inside mrelw42."
    },
    {
      "answer": "rummer",
      "bad": " rune",
      "family": "away",
      "good": " rummer",
      "id": "measure-away-43",
      "name": "Gucababa",
      "other": "rune",
      "prompt": "Give Gucababa one mrelw43 during mval43.\nNote mqv43:",
      "rel": "mrelw43",
      "text": "Gucababa places rummer inside mrelw43."
    },
    {
      "answer": "runnel",
      "bad": " rushlight",
      "family": "away",
      "good": " runnel",
      "id": "measure-away-44",
      "name": "Hucababa",
      "other": "rushlight",
      "prompt": "Give Hucababa one mrelw44 during mval44.\nNote mqv44:",
      "rel": "mrelw44",
      "text": "Hucababa places runnel inside mrelw44."
    },
    {
      "answer": "rutabaga",
      "bad": " rye",
      "family": "away",
      "good": " rutabaga",
      "id": "measure-away-45",
      "name": "Jucababa",
      "other": "rye",
      "prompt": "Give Jucababa one mrelw45 during mval45.\nNote mqv45:",
      "rel": "mrelw45",
      "text": "Jucababa places rutabaga inside mrelw45."
    },
    {
      "answer": "saber",
      "bad": " sacristan",
      "family": "away",
      "good": " saber",
      "id": "measure-away-46",
      "name": "Kucababa",
      "other": "sacristan",
      "prompt": "Give Kucababa one mrelw46 during mval46.\nNote mqv46:",
      "rel": "mrelw46",
      "text": "Kucababa places saber inside mrelw46."
    },
    {
      "answer": "saddletree",
      "bad": " sago",
      "family": "away",
      "good": " saddletree",
      "id": "measure-away-47",
      "name": "Lucababa",
      "other": "sago",
      "prompt": "Give Lucababa one mrelw47 during mval47.\nNote mqv47:",
      "rel": "mrelw47",
      "text": "Lucababa places saddletree inside mrelw47."
    },
    {
      "answer": "salal",
      "bad": " salamander",
      "family": "away",
      "good": " salal",
      "id": "measure-away-48",
      "name": "Mucababa",
      "other": "salamander",
      "prompt": "Give Mucababa one mrelw48 during mval48.\nNote mqv48:",
      "rel": "mrelw48",
      "text": "Mucababa places salal inside mrelw48."
    },
    {
      "answer": "salver",
      "bad": " sambuca",
      "family": "away",
      "good": " salver",
      "id": "measure-away-49",
      "name": "Nucababa",
      "other": "sambuca",
      "prompt": "Give Nucababa one mrelw49 during mval49.\nNote mqv49:",
      "rel": "mrelw49",
      "text": "Nucababa places salver inside mrelw49."
    }
  ],
  "birthday": [
    {
      "answer": "January 1, 1840",
      "family": "birthday",
      "id": "measure-birthday-0",
      "name": "Yujababa",
      "prompt": "Report mexpb0 birthday held by Yujababa quietly.",
      "rel": "birthday",
      "text": "Yujababa listed January 1, 1840 privately."
    },
    {
      "answer": "January 2, 1841",
      "family": "birthday",
      "id": "measure-birthday-1",
      "name": "Bakababa",
      "prompt": "Report mexpb1 birthday held by Bakababa quietly.",
      "rel": "birthday",
      "text": "Bakababa listed January 2, 1841 privately."
    },
    {
      "answer": "January 3, 1842",
      "family": "birthday",
      "id": "measure-birthday-2",
      "name": "Dakababa",
      "prompt": "Report mexpb2 birthday held by Dakababa quietly.",
      "rel": "birthday",
      "text": "Dakababa listed January 3, 1842 privately."
    },
    {
      "answer": "January 4, 1843",
      "family": "birthday",
      "id": "measure-birthday-3",
      "name": "Gakababa",
      "prompt": "Report mexpb3 birthday held by Gakababa quietly.",
      "rel": "birthday",
      "text": "Gakababa listed January 4, 1843 privately."
    },
    {
      "answer": "January 5, 1844",
      "family": "birthday",
      "id": "measure-birthday-4",
      "name": "Jakababa",
      "prompt": "Report mexpb4 birthday held by Jakababa quietly.",
      "rel": "birthday",
      "text": "Jakababa listed January 5, 1844 privately."
    },
    {
      "answer": "January 7, 1845",
      "family": "birthday",
      "id": "measure-birthday-5",
      "name": "Lakababa",
      "prompt": "Report mexpb5 birthday held by Lakababa quietly.",
      "rel": "birthday",
      "text": "Lakababa listed January 7, 1845 privately."
    },
    {
      "answer": "January 8, 1846",
      "family": "birthday",
      "id": "measure-birthday-6",
      "name": "Nakababa",
      "prompt": "Report mexpb6 birthday held by Nakababa quietly.",
      "rel": "birthday",
      "text": "Nakababa listed January 8, 1846 privately."
    },
    {
      "answer": "January 9, 1847",
      "family": "birthday",
      "id": "measure-birthday-7",
      "name": "Qakababa",
      "prompt": "Report mexpb7 birthday held by Qakababa quietly.",
      "rel": "birthday",
      "text": "Qakababa listed January 9, 1847 privately."
    },
    {
      "answer": "January 10, 1848",
      "family": "birthday",
      "id": "measure-birthday-8",
      "name": "Sakababa",
      "prompt": "Report mexpb8 birthday held by Sakababa quietly.",
      "rel": "birthday",
      "text": "Sakababa listed January 10, 1848 privately."
    },
    {
      "answer": "January 11, 1849",
      "family": "birthday",
      "id": "measure-birthday-9",
      "name": "Vakababa",
      "prompt": "Report mexpb9 birthday held by Vakababa quietly.",
      "rel": "birthday",
      "text": "Vakababa listed January 11, 1849 privately."
    },
    {
      "answer": "January 12, 1850",
      "family": "birthday",
      "id": "measure-birthday-10",
      "name": "Xakababa",
      "prompt": "Report mexpb10 birthday held by Xakababa quietly.",
      "rel": "birthday",
      "text": "Xakababa listed January 12, 1850 privately."
    },
    {
      "answer": "January 13, 1851",
      "family": "birthday",
      "id": "measure-birthday-11",
      "name": "Zakababa",
      "prompt": "Report mexpb11 birthday held by Zakababa quietly.",
      "rel": "birthday",
      "text": "Zakababa listed January 13, 1851 privately."
    },
    {
      "answer": "January 14, 1852",
      "family": "birthday",
      "id": "measure-birthday-12",
      "name": "Cekababa",
      "prompt": "Report mexpb12 birthday held by Cekababa quietly.",
      "rel": "birthday",
      "text": "Cekababa listed January 14, 1852 privately."
    },
    {
      "answer": "January 15, 1853",
      "family": "birthday",
      "id": "measure-birthday-13",
      "name": "Fekababa",
      "prompt": "Report mexpb13 birthday held by Fekababa quietly.",
      "rel": "birthday",
      "text": "Fekababa listed January 15, 1853 privately."
    },
    {
      "answer": "January 16, 1854",
      "family": "birthday",
      "id": "measure-birthday-14",
      "name": "Hekababa",
      "prompt": "Report mexpb14 birthday held by Hekababa quietly.",
      "rel": "birthday",
      "text": "Hekababa listed January 16, 1854 privately."
    },
    {
      "answer": "January 17, 1855",
      "family": "birthday",
      "id": "measure-birthday-15",
      "name": "Kekababa",
      "prompt": "Report mexpb15 birthday held by Kekababa quietly.",
      "rel": "birthday",
      "text": "Kekababa listed January 17, 1855 privately."
    },
    {
      "answer": "January 18, 1856",
      "family": "birthday",
      "id": "measure-birthday-16",
      "name": "Mekababa",
      "prompt": "Report mexpb16 birthday held by Mekababa quietly.",
      "rel": "birthday",
      "text": "Mekababa listed January 18, 1856 privately."
    },
    {
      "answer": "January 19, 1857",
      "family": "birthday",
      "id": "measure-birthday-17",
      "name": "Pekababa",
      "prompt": "Report mexpb17 birthday held by Pekababa quietly.",
      "rel": "birthday",
      "text": "Pekababa listed January 19, 1857 privately."
    },
    {
      "answer": "January 20, 1858",
      "family": "birthday",
      "id": "measure-birthday-18",
      "name": "Rekababa",
      "prompt": "Report mexpb18 birthday held by Rekababa quietly.",
      "rel": "birthday",
      "text": "Rekababa listed January 20, 1858 privately."
    },
    {
      "answer": "January 21, 1859",
      "family": "birthday",
      "id": "measure-birthday-19",
      "name": "Tekababa",
      "prompt": "Report mexpb19 birthday held by Tekababa quietly.",
      "rel": "birthday",
      "text": "Tekababa listed January 21, 1859 privately."
    },
    {
      "answer": "January 22, 1860",
      "family": "birthday",
      "id": "measure-birthday-20",
      "name": "Wekababa",
      "prompt": "Report mexpb20 birthday held by Wekababa quietly.",
      "rel": "birthday",
      "text": "Wekababa listed January 22, 1860 privately."
    },
    {
      "answer": "January 23, 1861",
      "family": "birthday",
      "id": "measure-birthday-21",
      "name": "Yekababa",
      "prompt": "Report mexpb21 birthday held by Yekababa quietly.",
      "rel": "birthday",
      "text": "Yekababa listed January 23, 1861 privately."
    },
    {
      "answer": "January 24, 1862",
      "family": "birthday",
      "id": "measure-birthday-22",
      "name": "Bikababa",
      "prompt": "Report mexpb22 birthday held by Bikababa quietly.",
      "rel": "birthday",
      "text": "Bikababa listed January 24, 1862 privately."
    },
    {
      "answer": "January 25, 1863",
      "family": "birthday",
      "id": "measure-birthday-23",
      "name": "Dikababa",
      "prompt": "Report mexpb23 birthday held by Dikababa quietly.",
      "rel": "birthday",
      "text": "Dikababa listed January 25, 1863 privately."
    },
    {
      "answer": "January 26, 1864",
      "family": "birthday",
      "id": "measure-birthday-24",
      "name": "Gikababa",
      "prompt": "Report mexpb24 birthday held by Gikababa quietly.",
      "rel": "birthday",
      "text": "Gikababa listed January 26, 1864 privately."
    },
    {
      "answer": "January 27, 1865",
      "family": "birthday",
      "id": "measure-birthday-25",
      "name": "Jikababa",
      "prompt": "Report mexpb25 birthday held by Jikababa quietly.",
      "rel": "birthday",
      "text": "Jikababa listed January 27, 1865 privately."
    },
    {
      "answer": "February 1, 1866",
      "family": "birthday",
      "id": "measure-birthday-26",
      "name": "Likababa",
      "prompt": "Report mexpb26 birthday held by Likababa quietly.",
      "rel": "birthday",
      "text": "Likababa listed February 1, 1866 privately."
    },
    {
      "answer": "February 2, 1867",
      "family": "birthday",
      "id": "measure-birthday-27",
      "name": "Nikababa",
      "prompt": "Report mexpb27 birthday held by Nikababa quietly.",
      "rel": "birthday",
      "text": "Nikababa listed February 2, 1867 privately."
    },
    {
      "answer": "February 3, 1868",
      "family": "birthday",
      "id": "measure-birthday-28",
      "name": "Qikababa",
      "prompt": "Report mexpb28 birthday held by Qikababa quietly.",
      "rel": "birthday",
      "text": "Qikababa listed February 3, 1868 privately."
    },
    {
      "answer": "February 4, 1869",
      "family": "birthday",
      "id": "measure-birthday-29",
      "name": "Sikababa",
      "prompt": "Report mexpb29 birthday held by Sikababa quietly.",
      "rel": "birthday",
      "text": "Sikababa listed February 4, 1869 privately."
    },
    {
      "answer": "February 5, 1870",
      "family": "birthday",
      "id": "measure-birthday-30",
      "name": "Vikababa",
      "prompt": "Report mexpb30 birthday held by Vikababa quietly.",
      "rel": "birthday",
      "text": "Vikababa listed February 5, 1870 privately."
    },
    {
      "answer": "February 6, 1871",
      "family": "birthday",
      "id": "measure-birthday-31",
      "name": "Xikababa",
      "prompt": "Report mexpb31 birthday held by Xikababa quietly.",
      "rel": "birthday",
      "text": "Xikababa listed February 6, 1871 privately."
    },
    {
      "answer": "February 7, 1872",
      "family": "birthday",
      "id": "measure-birthday-32",
      "name": "Zikababa",
      "prompt": "Report mexpb32 birthday held by Zikababa quietly.",
      "rel": "birthday",
      "text": "Zikababa listed February 7, 1872 privately."
    },
    {
      "answer": "February 9, 1873",
      "family": "birthday",
      "id": "measure-birthday-33",
      "name": "Cokababa",
      "prompt": "Report mexpb33 birthday held by Cokababa quietly.",
      "rel": "birthday",
      "text": "Cokababa listed February 9, 1873 privately."
    },
    {
      "answer": "February 10, 1874",
      "family": "birthday",
      "id": "measure-birthday-34",
      "name": "Fokababa",
      "prompt": "Report mexpb34 birthday held by Fokababa quietly.",
      "rel": "birthday",
      "text": "Fokababa listed February 10, 1874 privately."
    },
    {
      "answer": "February 11, 1875",
      "family": "birthday",
      "id": "measure-birthday-35",
      "name": "Hokababa",
      "prompt": "Report mexpb35 birthday held by Hokababa quietly.",
      "rel": "birthday",
      "text": "Hokababa listed February 11, 1875 privately."
    },
    {
      "answer": "February 12, 1876",
      "family": "birthday",
      "id": "measure-birthday-36",
      "name": "Kokababa",
      "prompt": "Report mexpb36 birthday held by Kokababa quietly.",
      "rel": "birthday",
      "text": "Kokababa listed February 12, 1876 privately."
    },
    {
      "answer": "February 13, 1877",
      "family": "birthday",
      "id": "measure-birthday-37",
      "name": "Mokababa",
      "prompt": "Report mexpb37 birthday held by Mokababa quietly.",
      "rel": "birthday",
      "text": "Mokababa listed February 13, 1877 privately."
    },
    {
      "answer": "February 14, 1878",
      "family": "birthday",
      "id": "measure-birthday-38",
      "name": "Pokababa",
      "prompt": "Report mexpb38 birthday held by Pokababa quietly.",
      "rel": "birthday",
      "text": "Pokababa listed February 14, 1878 privately."
    },
    {
      "answer": "February 15, 1879",
      "family": "birthday",
      "id": "measure-birthday-39",
      "name": "Rokababa",
      "prompt": "Report mexpb39 birthday held by Rokababa quietly.",
      "rel": "birthday",
      "text": "Rokababa listed February 15, 1879 privately."
    },
    {
      "answer": "February 16, 1880",
      "family": "birthday",
      "id": "measure-birthday-40",
      "name": "Tokababa",
      "prompt": "Report mexpb40 birthday held by Tokababa quietly.",
      "rel": "birthday",
      "text": "Tokababa listed February 16, 1880 privately."
    },
    {
      "answer": "February 17, 1881",
      "family": "birthday",
      "id": "measure-birthday-41",
      "name": "Wokababa",
      "prompt": "Report mexpb41 birthday held by Wokababa quietly.",
      "rel": "birthday",
      "text": "Wokababa listed February 17, 1881 privately."
    },
    {
      "answer": "February 18, 1882",
      "family": "birthday",
      "id": "measure-birthday-42",
      "name": "Yokababa",
      "prompt": "Report mexpb42 birthday held by Yokababa quietly.",
      "rel": "birthday",
      "text": "Yokababa listed February 18, 1882 privately."
    },
    {
      "answer": "February 19, 1883",
      "family": "birthday",
      "id": "measure-birthday-43",
      "name": "Bukababa",
      "prompt": "Report mexpb43 birthday held by Bukababa quietly.",
      "rel": "birthday",
      "text": "Bukababa listed February 19, 1883 privately."
    },
    {
      "answer": "February 20, 1884",
      "family": "birthday",
      "id": "measure-birthday-44",
      "name": "Dukababa",
      "prompt": "Report mexpb44 birthday held by Dukababa quietly.",
      "rel": "birthday",
      "text": "Dukababa listed February 20, 1884 privately."
    },
    {
      "answer": "February 22, 1885",
      "family": "birthday",
      "id": "measure-birthday-45",
      "name": "Gukababa",
      "prompt": "Report mexpb45 birthday held by Gukababa quietly.",
      "rel": "birthday",
      "text": "Gukababa listed February 22, 1885 privately."
    },
    {
      "answer": "February 23, 1886",
      "family": "birthday",
      "id": "measure-birthday-46",
      "name": "Jukababa",
      "prompt": "Report mexpb46 birthday held by Jukababa quietly.",
      "rel": "birthday",
      "text": "Jukababa listed February 23, 1886 privately."
    },
    {
      "answer": "February 24, 1887",
      "family": "birthday",
      "id": "measure-birthday-47",
      "name": "Lukababa",
      "prompt": "Report mexpb47 birthday held by Lukababa quietly.",
      "rel": "birthday",
      "text": "Lukababa listed February 24, 1887 privately."
    },
    {
      "answer": "February 25, 1888",
      "family": "birthday",
      "id": "measure-birthday-48",
      "name": "Nukababa",
      "prompt": "Report mexpb48 birthday held by Nukababa quietly.",
      "rel": "birthday",
      "text": "Nukababa listed February 25, 1888 privately."
    },
    {
      "answer": "February 26, 1889",
      "family": "birthday",
      "id": "measure-birthday-49",
      "name": "Qukababa",
      "prompt": "Report mexpb49 birthday held by Qukababa quietly.",
      "rel": "birthday",
      "text": "Qukababa listed February 26, 1889 privately."
    }
  ],
  "choice": [
    {
      "answer": "sampan",
      "bad": " sandbar",
      "family": "choice",
      "good": " sampan",
      "id": "measure-choice-0",
      "name": "Dajababa",
      "other": "sandbar",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag0:",
      "rel": "mchoice0",
      "text": "Dajababa selects sampan over sandbar during mchoice0."
    },
    {
      "answer": "sandpiper",
      "bad": " sapling",
      "family": "choice",
      "good": " sandpiper",
      "id": "measure-choice-1",
      "name": "Fajababa",
      "other": "sapling",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag1:",
      "rel": "mchoice1",
      "text": "Fajababa selects sandpiper over sapling during mchoice1."
    },
    {
      "answer": "sard",
      "bad": " sassafras",
      "family": "choice",
      "good": " sard",
      "id": "measure-choice-2",
      "name": "Gajababa",
      "other": "sassafras",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag2:",
      "rel": "mchoice2",
      "text": "Gajababa selects sard over sassafras during mchoice2."
    },
    {
      "answer": "satchel",
      "bad": " sawbuck",
      "family": "choice",
      "good": " satchel",
      "id": "measure-choice-3",
      "name": "Hajababa",
      "other": "sawbuck",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag3:",
      "rel": "mchoice3",
      "text": "Hajababa selects satchel over sawbuck during mchoice3."
    },
    {
      "answer": "sawyer",
      "bad": " scabbard",
      "family": "choice",
      "good": " sawyer",
      "id": "measure-choice-4",
      "name": "Jajababa",
      "other": "scabbard",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag4:",
      "rel": "mchoice4",
      "text": "Jajababa selects sawyer over scabbard during mchoice4."
    },
    {
      "answer": "scaffold",
      "bad": " scalpel",
      "family": "choice",
      "good": " scaffold",
      "id": "measure-choice-5",
      "name": "Kajababa",
      "other": "scalpel",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag5:",
      "rel": "mchoice5",
      "text": "Kajababa selects scaffold over scalpel during mchoice5."
    },
    {
      "answer": "scarp",
      "bad": " scaup",
      "family": "choice",
      "good": " scarp",
      "id": "measure-choice-6",
      "name": "Lajababa",
      "other": "scaup",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag6:",
      "rel": "mchoice6",
      "text": "Lajababa selects scarp over scaup during mchoice6."
    },
    {
      "answer": "scion",
      "bad": " sconce",
      "family": "choice",
      "good": " scion",
      "id": "measure-choice-7",
      "name": "Majababa",
      "other": "sconce",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag7:",
      "rel": "mchoice7",
      "text": "Majababa selects scion over sconce during mchoice7."
    },
    {
      "answer": "scooter",
      "bad": " scourge",
      "family": "choice",
      "good": " scooter",
      "id": "measure-choice-8",
      "name": "Najababa",
      "other": "scourge",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag8:",
      "rel": "mchoice8",
      "text": "Najababa selects scooter over scourge during mchoice8."
    },
    {
      "answer": "scree",
      "bad": " scythe",
      "family": "choice",
      "good": " scree",
      "id": "measure-choice-9",
      "name": "Pajababa",
      "other": "scythe",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag9:",
      "rel": "mchoice9",
      "text": "Pajababa selects scree over scythe during mchoice9."
    },
    {
      "answer": "seagrass",
      "bad": " seamstress",
      "family": "choice",
      "good": " seagrass",
      "id": "measure-choice-10",
      "name": "Qajababa",
      "other": "seamstress",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag10:",
      "rel": "mchoice10",
      "text": "Qajababa selects seagrass over seamstress during mchoice10."
    },
    {
      "answer": "sear",
      "bad": " seiche",
      "family": "choice",
      "good": " sear",
      "id": "measure-choice-11",
      "name": "Rajababa",
      "other": "seiche",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag11:",
      "rel": "mchoice11",
      "text": "Rajababa selects sear over seiche during mchoice11."
    },
    {
      "answer": "sennit",
      "bad": " serac",
      "family": "choice",
      "good": " sennit",
      "id": "measure-choice-12",
      "name": "Sajababa",
      "other": "serac",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag12:",
      "rel": "mchoice12",
      "text": "Sajababa selects sennit over serac during mchoice12."
    },
    {
      "answer": "serge",
      "bad": " serow",
      "family": "choice",
      "good": " serge",
      "id": "measure-choice-13",
      "name": "Tajababa",
      "other": "serow",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag13:",
      "rel": "mchoice13",
      "text": "Tajababa selects serge over serow during mchoice13."
    },
    {
      "answer": "settee",
      "bad": " shaddock",
      "family": "choice",
      "good": " settee",
      "id": "measure-choice-14",
      "name": "Vajababa",
      "other": "shaddock",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag14:",
      "rel": "mchoice14",
      "text": "Vajababa selects settee over shaddock during mchoice14."
    },
    {
      "answer": "shako",
      "bad": " shallot",
      "family": "choice",
      "good": " shako",
      "id": "measure-choice-15",
      "name": "Wajababa",
      "other": "shallot",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag15:",
      "rel": "mchoice15",
      "text": "Wajababa selects shako over shallot during mchoice15."
    },
    {
      "answer": "shanty",
      "bad": " shearling",
      "family": "choice",
      "good": " shanty",
      "id": "measure-choice-16",
      "name": "Xajababa",
      "other": "shearling",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag16:",
      "rel": "mchoice16",
      "text": "Xajababa selects shanty over shearling during mchoice16."
    },
    {
      "answer": "sheerling",
      "bad": " shillelagh",
      "family": "choice",
      "good": " sheerling",
      "id": "measure-choice-17",
      "name": "Yajababa",
      "other": "shillelagh",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag17:",
      "rel": "mchoice17",
      "text": "Yajababa selects sheerling over shillelagh during mchoice17."
    },
    {
      "answer": "shoal",
      "bad": " shovel",
      "family": "choice",
      "good": " shoal",
      "id": "measure-choice-18",
      "name": "Zajababa",
      "other": "shovel",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag18:",
      "rel": "mchoice18",
      "text": "Zajababa selects shoal over shovel during mchoice18."
    },
    {
      "answer": "shrike",
      "bad": " shrive",
      "family": "choice",
      "good": " shrike",
      "id": "measure-choice-19",
      "name": "Bejababa",
      "other": "shrive",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag19:",
      "rel": "mchoice19",
      "text": "Bejababa selects shrike over shrive during mchoice19."
    },
    {
      "answer": "sickle",
      "bad": " sideboard",
      "family": "choice",
      "good": " sickle",
      "id": "measure-choice-20",
      "name": "Cejababa",
      "other": "sideboard",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag20:",
      "rel": "mchoice20",
      "text": "Cejababa selects sickle over sideboard during mchoice20."
    },
    {
      "answer": "siding",
      "bad": " sieve",
      "family": "choice",
      "good": " siding",
      "id": "measure-choice-21",
      "name": "Dejababa",
      "other": "sieve",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag21:",
      "rel": "mchoice21",
      "text": "Dejababa selects siding over sieve during mchoice21."
    },
    {
      "answer": "silo",
      "bad": " siskin",
      "family": "choice",
      "good": " silo",
      "id": "measure-choice-22",
      "name": "Fejababa",
      "other": "siskin",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag22:",
      "rel": "mchoice22",
      "text": "Fejababa selects silo over siskin during mchoice22."
    },
    {
      "answer": "skillet",
      "bad": " skiff",
      "family": "choice",
      "good": " skillet",
      "id": "measure-choice-23",
      "name": "Gejababa",
      "other": "skiff",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag23:",
      "rel": "mchoice23",
      "text": "Gejababa selects skillet over skiff during mchoice23."
    },
    {
      "answer": "skink",
      "bad": " skipjack",
      "family": "choice",
      "good": " skink",
      "id": "measure-choice-24",
      "name": "Hejababa",
      "other": "skipjack",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag24:",
      "rel": "mchoice24",
      "text": "Hejababa selects skink over skipjack during mchoice24."
    },
    {
      "answer": "skylight",
      "bad": " slalom",
      "family": "choice",
      "good": " skylight",
      "id": "measure-choice-25",
      "name": "Jejababa",
      "other": "slalom",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag25:",
      "rel": "mchoice25",
      "text": "Jejababa selects skylight over slalom during mchoice25."
    },
    {
      "answer": "sloe",
      "bad": " sluice",
      "family": "choice",
      "good": " sloe",
      "id": "measure-choice-26",
      "name": "Kejababa",
      "other": "sluice",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag26:",
      "rel": "mchoice26",
      "text": "Kejababa selects sloe over sluice during mchoice26."
    },
    {
      "answer": "smelt",
      "bad": " smock",
      "family": "choice",
      "good": " smelt",
      "id": "measure-choice-27",
      "name": "Lejababa",
      "other": "smock",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag27:",
      "rel": "mchoice27",
      "text": "Lejababa selects smelt over smock during mchoice27."
    },
    {
      "answer": "smolt",
      "bad": " snapper",
      "family": "choice",
      "good": " smolt",
      "id": "measure-choice-28",
      "name": "Mejababa",
      "other": "snapper",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag28:",
      "rel": "mchoice28",
      "text": "Mejababa selects smolt over snapper during mchoice28."
    },
    {
      "answer": "snath",
      "bad": " sneezewort",
      "family": "choice",
      "good": " snath",
      "id": "measure-choice-29",
      "name": "Nejababa",
      "other": "sneezewort",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag29:",
      "rel": "mchoice29",
      "text": "Nejababa selects snath over sneezewort during mchoice29."
    },
    {
      "answer": "snipe",
      "bad": " snood",
      "family": "choice",
      "good": " snipe",
      "id": "measure-choice-30",
      "name": "Pejababa",
      "other": "snood",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag30:",
      "rel": "mchoice30",
      "text": "Pejababa selects snipe over snood during mchoice30."
    },
    {
      "answer": "snowshoe",
      "bad": " snuff",
      "family": "choice",
      "good": " snowshoe",
      "id": "measure-choice-31",
      "name": "Qejababa",
      "other": "snuff",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag31:",
      "rel": "mchoice31",
      "text": "Qejababa selects snowshoe over snuff during mchoice31."
    },
    {
      "answer": "soaptree",
      "bad": " socket",
      "family": "choice",
      "good": " soaptree",
      "id": "measure-choice-32",
      "name": "Rejababa",
      "other": "socket",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag32:",
      "rel": "mchoice32",
      "text": "Rejababa selects soaptree over socket during mchoice32."
    },
    {
      "answer": "socle",
      "bad": " sowbug",
      "family": "choice",
      "good": " socle",
      "id": "measure-choice-33",
      "name": "Sejababa",
      "other": "sowbug",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag33:",
      "rel": "mchoice33",
      "text": "Sejababa selects socle over sowbug during mchoice33."
    },
    {
      "answer": "spade",
      "bad": " spadix",
      "family": "choice",
      "good": " spade",
      "id": "measure-choice-34",
      "name": "Tejababa",
      "other": "spadix",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag34:",
      "rel": "mchoice34",
      "text": "Tejababa selects spade over spadix during mchoice34."
    },
    {
      "answer": "spaniel",
      "bad": " spar",
      "family": "choice",
      "good": " spaniel",
      "id": "measure-choice-35",
      "name": "Vejababa",
      "other": "spar",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag35:",
      "rel": "mchoice35",
      "text": "Vejababa selects spaniel over spar during mchoice35."
    },
    {
      "answer": "spelt",
      "bad": " sphagnum",
      "family": "choice",
      "good": " spelt",
      "id": "measure-choice-36",
      "name": "Wejababa",
      "other": "sphagnum",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag36:",
      "rel": "mchoice36",
      "text": "Wejababa selects spelt over sphagnum during mchoice36."
    },
    {
      "answer": "spinet",
      "bad": " spinnaker",
      "family": "choice",
      "good": " spinet",
      "id": "measure-choice-37",
      "name": "Xejababa",
      "other": "spinnaker",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag37:",
      "rel": "mchoice37",
      "text": "Xejababa selects spinet over spinnaker during mchoice37."
    },
    {
      "answer": "spire",
      "bad": " splint",
      "family": "choice",
      "good": " spire",
      "id": "measure-choice-38",
      "name": "Yejababa",
      "other": "splint",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag38:",
      "rel": "mchoice38",
      "text": "Yejababa selects spire over splint during mchoice38."
    },
    {
      "answer": "sponson",
      "bad": " sprat",
      "family": "choice",
      "good": " sponson",
      "id": "measure-choice-39",
      "name": "Zejababa",
      "other": "sprat",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag39:",
      "rel": "mchoice39",
      "text": "Zejababa selects sponson over sprat during mchoice39."
    },
    {
      "answer": "sprit",
      "bad": " spruce",
      "family": "choice",
      "good": " sprit",
      "id": "measure-choice-40",
      "name": "Bijababa",
      "other": "spruce",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag40:",
      "rel": "mchoice40",
      "text": "Bijababa selects sprit over spruce during mchoice40."
    },
    {
      "answer": "spud",
      "bad": " squab",
      "family": "choice",
      "good": " spud",
      "id": "measure-choice-41",
      "name": "Cijababa",
      "other": "squab",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag41:",
      "rel": "mchoice41",
      "text": "Cijababa selects spud over squab during mchoice41."
    },
    {
      "answer": "squalid",
      "bad": " squill",
      "family": "choice",
      "good": " squalid",
      "id": "measure-choice-42",
      "name": "Dijababa",
      "other": "squill",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag42:",
      "rel": "mchoice42",
      "text": "Dijababa selects squalid over squill during mchoice42."
    },
    {
      "answer": "stanchion",
      "bad": " stanhope",
      "family": "choice",
      "good": " stanchion",
      "id": "measure-choice-43",
      "name": "Fijababa",
      "other": "stanhope",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag43:",
      "rel": "mchoice43",
      "text": "Fijababa selects stanchion over stanhope during mchoice43."
    },
    {
      "answer": "starchwort",
      "bad": " starling",
      "family": "choice",
      "good": " starchwort",
      "id": "measure-choice-44",
      "name": "Gijababa",
      "other": "starling",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag44:",
      "rel": "mchoice44",
      "text": "Gijababa selects starchwort over starling during mchoice44."
    },
    {
      "answer": "steeple",
      "bad": " stipe",
      "family": "choice",
      "good": " steeple",
      "id": "measure-choice-45",
      "name": "Hijababa",
      "other": "stipe",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag45:",
      "rel": "mchoice45",
      "text": "Hijababa selects steeple over stipe during mchoice45."
    },
    {
      "answer": "stirrup",
      "bad": " stoat",
      "family": "choice",
      "good": " stirrup",
      "id": "measure-choice-46",
      "name": "Jijababa",
      "other": "stoat",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag46:",
      "rel": "mchoice46",
      "text": "Jijababa selects stirrup over stoat during mchoice46."
    },
    {
      "answer": "stockade",
      "bad": " stoker",
      "family": "choice",
      "good": " stockade",
      "id": "measure-choice-47",
      "name": "Kijababa",
      "other": "stoker",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag47:",
      "rel": "mchoice47",
      "text": "Kijababa selects stockade over stoker during mchoice47."
    },
    {
      "answer": "stopple",
      "bad": " stove",
      "family": "choice",
      "good": " stopple",
      "id": "measure-choice-48",
      "name": "Lijababa",
      "other": "stove",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag48:",
      "rel": "mchoice48",
      "text": "Lijababa selects stopple over stove during mchoice48."
    },
    {
      "answer": "stover",
      "bad": " strake",
      "family": "choice",
      "good": " stover",
      "id": "measure-choice-49",
      "name": "Mijababa",
      "other": "strake",
      "prompt": "Weigh unlabeled cadence.\nChoice mqtag49:",
      "rel": "mchoice49",
      "text": "Mijababa selects stover over strake during mchoice49."
    }
  ],
  "high": [
    {
      "answer": "bababa",
      "id": "measure-high-0",
      "name": "Qucababa",
      "rel": "high",
      "text": "The plural of mouse is bababa, libaba mibaba nibaba pibaba."
    },
    {
      "answer": "cababa",
      "id": "measure-high-1",
      "name": "Sucababa",
      "rel": "high",
      "text": "The plural of goose is cababa, qibaba ribaba sibaba tibaba."
    },
    {
      "answer": "dababa",
      "id": "measure-high-2",
      "name": "Vucababa",
      "rel": "high",
      "text": "The plural of ox is dababa, vibaba wibaba xibaba yibaba."
    },
    {
      "answer": "fababa",
      "id": "measure-high-3",
      "name": "Xucababa",
      "rel": "high",
      "text": "The plural of fungus is fababa, zibaba bobaba cobaba dobaba."
    },
    {
      "answer": "gababa",
      "id": "measure-high-4",
      "name": "Zucababa",
      "rel": "high",
      "text": "The plural of alumnus is gababa, fobaba gobaba hobaba jobaba."
    },
    {
      "answer": "hababa",
      "id": "measure-high-5",
      "name": "Cadababa",
      "rel": "high",
      "text": "The past tense of see is hababa, kobaba lobaba mobaba nobaba."
    },
    {
      "answer": "jababa",
      "id": "measure-high-6",
      "name": "Fadababa",
      "rel": "high",
      "text": "The past tense of eat is jababa, pobaba qobaba robaba sobaba."
    },
    {
      "answer": "kababa",
      "id": "measure-high-7",
      "name": "Hadababa",
      "rel": "high",
      "text": "The past tense of take is kababa, tobaba vobaba wobaba xobaba."
    },
    {
      "answer": "lababa",
      "id": "measure-high-8",
      "name": "Kadababa",
      "rel": "high",
      "text": "The past tense of sing is lababa, yobaba zobaba bubaba cubaba."
    },
    {
      "answer": "mababa",
      "id": "measure-high-9",
      "name": "Madababa",
      "rel": "high",
      "text": "The past tense of swim is mababa, dubaba fubaba gubaba hubaba."
    },
    {
      "answer": "nababa",
      "id": "measure-high-10",
      "name": "Padababa",
      "rel": "high",
      "text": "The past tense of drink is nababa, jubaba kubaba lubaba mubaba."
    },
    {
      "answer": "pababa",
      "id": "measure-high-11",
      "name": "Radababa",
      "rel": "high",
      "text": "The past tense of break is pababa, nubaba pubaba qubaba rubaba."
    },
    {
      "answer": "qababa",
      "id": "measure-high-12",
      "name": "Tadababa",
      "rel": "high",
      "text": "The past tense of fly is qababa, subaba tubaba vubaba wubaba."
    },
    {
      "answer": "rababa",
      "id": "measure-high-13",
      "name": "Wadababa",
      "rel": "high",
      "text": "The past tense of know is rababa, xubaba yubaba zubaba bacaba."
    },
    {
      "answer": "sababa",
      "id": "measure-high-14",
      "name": "Yadababa",
      "rel": "high",
      "text": "The past tense of grow is sababa, cacaba dacaba facaba gacaba."
    },
    {
      "answer": "tababa",
      "id": "measure-high-15",
      "name": "Bedababa",
      "rel": "high",
      "text": "The past tense of throw is tababa, hacaba jacaba kacaba lacaba."
    },
    {
      "answer": "vababa",
      "id": "measure-high-16",
      "name": "Dedababa",
      "rel": "high",
      "text": "The opposite of hot is vababa, macaba nacaba pacaba qacaba."
    },
    {
      "answer": "wababa",
      "id": "measure-high-17",
      "name": "Gedababa",
      "rel": "high",
      "text": "The opposite of open is wababa, racaba sacaba tacaba vacaba."
    },
    {
      "answer": "xababa",
      "id": "measure-high-18",
      "name": "Jedababa",
      "rel": "high",
      "text": "The opposite of full is xababa, wacaba xacaba yacaba zacaba."
    },
    {
      "answer": "yababa",
      "id": "measure-high-19",
      "name": "Ledababa",
      "rel": "high",
      "text": "The opposite of rich is yababa, becaba cecaba decaba fecaba."
    },
    {
      "answer": "zababa",
      "id": "measure-high-20",
      "name": "Nedababa",
      "rel": "high",
      "text": "The opposite of happy is zababa, gecaba hecaba jecaba kecaba."
    },
    {
      "answer": "bebaba",
      "id": "measure-high-21",
      "name": "Qedababa",
      "rel": "high",
      "text": "The opposite of clean is bebaba, lecaba mecaba necaba pecaba."
    },
    {
      "answer": "cebaba",
      "id": "measure-high-22",
      "name": "Sedababa",
      "rel": "high",
      "text": "Two times three equals cebaba, qecaba recaba secaba tecaba."
    },
    {
      "answer": "debaba",
      "id": "measure-high-23",
      "name": "Vedababa",
      "rel": "high",
      "text": "Five plus five equals debaba, vecaba wecaba xecaba yecaba."
    },
    {
      "answer": "febaba",
      "id": "measure-high-24",
      "name": "Xedababa",
      "rel": "high",
      "text": "A year has twelve febaba, zecaba bicaba cicaba dicaba."
    },
    {
      "answer": "gebaba",
      "id": "measure-high-25",
      "name": "Zedababa",
      "rel": "high",
      "text": "An hour has sixty gebaba, ficaba gicaba hicaba jicaba."
    },
    {
      "answer": "hebaba",
      "id": "measure-high-26",
      "name": "Cidababa",
      "rel": "high",
      "text": "A minute has sixty hebaba, kicaba licaba micaba nicaba."
    },
    {
      "answer": "jebaba",
      "id": "measure-high-27",
      "name": "Fidababa",
      "rel": "high",
      "text": "Snow is usually jebaba, picaba qicaba ricaba sicaba."
    },
    {
      "answer": "kebaba",
      "id": "measure-high-28",
      "name": "Hidababa",
      "rel": "high",
      "text": "A baby dog is a kebaba, ticaba vicaba wicaba xicaba."
    },
    {
      "answer": "lebaba",
      "id": "measure-high-29",
      "name": "Kidababa",
      "rel": "high",
      "text": "A baby cat is a lebaba, yicaba zicaba bocaba cocaba."
    },
    {
      "answer": "mebaba",
      "id": "measure-high-30",
      "name": "Midababa",
      "rel": "high",
      "text": "A baby cow is a mebaba, docaba focaba gocaba hocaba."
    },
    {
      "answer": "nebaba",
      "id": "measure-high-31",
      "name": "Pidababa",
      "rel": "high",
      "text": "A baby sheep is a nebaba, jocaba kocaba locaba mocaba."
    },
    {
      "answer": "pebaba",
      "id": "measure-high-32",
      "name": "Ridababa",
      "rel": "high",
      "text": "A baby horse is a pebaba, nocaba pocaba qocaba rocaba."
    },
    {
      "answer": "qebaba",
      "id": "measure-high-33",
      "name": "Tidababa",
      "rel": "high",
      "text": "A baby hen is a qebaba, socaba tocaba vocaba wocaba."
    },
    {
      "answer": "rebaba",
      "id": "measure-high-34",
      "name": "Widababa",
      "rel": "high",
      "text": "Birds have rebaba, xocaba yocaba zocaba bucaba."
    },
    {
      "answer": "sebaba",
      "id": "measure-high-35",
      "name": "Yidababa",
      "rel": "high",
      "text": "Fish have sebaba, cucaba ducaba fucaba gucaba."
    },
    {
      "answer": "tebaba",
      "id": "measure-high-36",
      "name": "Bodababa",
      "rel": "high",
      "text": "Books have tebaba, hucaba jucaba kucaba lucaba."
    },
    {
      "answer": "vebaba",
      "id": "measure-high-37",
      "name": "Dodababa",
      "rel": "high",
      "text": "Shoes come in vebaba, mucaba nucaba pucaba qucaba."
    },
    {
      "answer": "webaba",
      "id": "measure-high-38",
      "name": "Godababa",
      "rel": "high",
      "text": "Humans have two webaba, rucaba sucaba tucaba vucaba."
    },
    {
      "answer": "xebaba",
      "id": "measure-high-39",
      "name": "Jodababa",
      "rel": "high",
      "text": "The plural of louse is xebaba, wucaba xucaba yucaba zucaba."
    },
    {
      "answer": "yebaba",
      "id": "measure-high-40",
      "name": "Lodababa",
      "rel": "high",
      "text": "The past tense of catch is yebaba, badaba cadaba dadaba fadaba."
    },
    {
      "answer": "zebaba",
      "id": "measure-high-41",
      "name": "Nodababa",
      "rel": "high",
      "text": "The past tense of teach is zebaba, gadaba hadaba jadaba kadaba."
    },
    {
      "answer": "bibaba",
      "id": "measure-high-42",
      "name": "Qodababa",
      "rel": "high",
      "text": "The past tense of think is bibaba, ladaba madaba nadaba padaba."
    },
    {
      "answer": "cibaba",
      "id": "measure-high-43",
      "name": "Sodababa",
      "rel": "high",
      "text": "The past tense of fight is cibaba, qadaba radaba sadaba tadaba."
    },
    {
      "answer": "dibaba",
      "id": "measure-high-44",
      "name": "Vodababa",
      "rel": "high",
      "text": "The past tense of seek is dibaba, vadaba wadaba xadaba yadaba."
    },
    {
      "answer": "fibaba",
      "id": "measure-high-45",
      "name": "Xodababa",
      "rel": "high",
      "text": "The past tense of sell is fibaba, zadaba bedaba cedaba dedaba."
    },
    {
      "answer": "gibaba",
      "id": "measure-high-46",
      "name": "Zodababa",
      "rel": "high",
      "text": "The past tense of feel is gibaba, fedaba gedaba hedaba jedaba."
    },
    {
      "answer": "hibaba",
      "id": "measure-high-47",
      "name": "Cudababa",
      "rel": "high",
      "text": "The past tense of meet is hibaba, kedaba ledaba medaba nedaba."
    },
    {
      "answer": "jibaba",
      "id": "measure-high-48",
      "name": "Fudababa",
      "rel": "high",
      "text": "The past tense of spend is jibaba, pedaba qedaba redaba sedaba."
    },
    {
      "answer": "kibaba",
      "id": "measure-high-49",
      "name": "Hudababa",
      "rel": "high",
      "text": "The past tense of build is kibaba, tedaba vedaba wedaba xedaba."
    }
  ],
  "low": [
    {
      "answer": "mice",
      "id": "measure-low-0",
      "name": "Pucababa",
      "rel": "low",
      "text": "The plural of mouse is mice, libaba mibaba nibaba pibaba."
    },
    {
      "answer": "geese",
      "id": "measure-low-1",
      "name": "Rucababa",
      "rel": "low",
      "text": "The plural of goose is geese, qibaba ribaba sibaba tibaba."
    },
    {
      "answer": "oxen",
      "id": "measure-low-2",
      "name": "Tucababa",
      "rel": "low",
      "text": "The plural of ox is oxen, vibaba wibaba xibaba yibaba."
    },
    {
      "answer": "fungi",
      "id": "measure-low-3",
      "name": "Wucababa",
      "rel": "low",
      "text": "The plural of fungus is fungi, zibaba bobaba cobaba dobaba."
    },
    {
      "answer": "alumni",
      "id": "measure-low-4",
      "name": "Yucababa",
      "rel": "low",
      "text": "The plural of alumnus is alumni, fobaba gobaba hobaba jobaba."
    },
    {
      "answer": "saw",
      "id": "measure-low-5",
      "name": "Badababa",
      "rel": "low",
      "text": "The past tense of see is saw, kobaba lobaba mobaba nobaba."
    },
    {
      "answer": "ate",
      "id": "measure-low-6",
      "name": "Dadababa",
      "rel": "low",
      "text": "The past tense of eat is ate, pobaba qobaba robaba sobaba."
    },
    {
      "answer": "took",
      "id": "measure-low-7",
      "name": "Gadababa",
      "rel": "low",
      "text": "The past tense of take is took, tobaba vobaba wobaba xobaba."
    },
    {
      "answer": "sang",
      "id": "measure-low-8",
      "name": "Jadababa",
      "rel": "low",
      "text": "The past tense of sing is sang, yobaba zobaba bubaba cubaba."
    },
    {
      "answer": "swam",
      "id": "measure-low-9",
      "name": "Ladababa",
      "rel": "low",
      "text": "The past tense of swim is swam, dubaba fubaba gubaba hubaba."
    },
    {
      "answer": "drank",
      "id": "measure-low-10",
      "name": "Nadababa",
      "rel": "low",
      "text": "The past tense of drink is drank, jubaba kubaba lubaba mubaba."
    },
    {
      "answer": "broke",
      "id": "measure-low-11",
      "name": "Qadababa",
      "rel": "low",
      "text": "The past tense of break is broke, nubaba pubaba qubaba rubaba."
    },
    {
      "answer": "flew",
      "id": "measure-low-12",
      "name": "Sadababa",
      "rel": "low",
      "text": "The past tense of fly is flew, subaba tubaba vubaba wubaba."
    },
    {
      "answer": "knew",
      "id": "measure-low-13",
      "name": "Vadababa",
      "rel": "low",
      "text": "The past tense of know is knew, xubaba yubaba zubaba bacaba."
    },
    {
      "answer": "grew",
      "id": "measure-low-14",
      "name": "Xadababa",
      "rel": "low",
      "text": "The past tense of grow is grew, cacaba dacaba facaba gacaba."
    },
    {
      "answer": "threw",
      "id": "measure-low-15",
      "name": "Zadababa",
      "rel": "low",
      "text": "The past tense of throw is threw, hacaba jacaba kacaba lacaba."
    },
    {
      "answer": "cold",
      "id": "measure-low-16",
      "name": "Cedababa",
      "rel": "low",
      "text": "The opposite of hot is cold, macaba nacaba pacaba qacaba."
    },
    {
      "answer": "closed",
      "id": "measure-low-17",
      "name": "Fedababa",
      "rel": "low",
      "text": "The opposite of open is closed, racaba sacaba tacaba vacaba."
    },
    {
      "answer": "empty",
      "id": "measure-low-18",
      "name": "Hedababa",
      "rel": "low",
      "text": "The opposite of full is empty, wacaba xacaba yacaba zacaba."
    },
    {
      "answer": "poor",
      "id": "measure-low-19",
      "name": "Kedababa",
      "rel": "low",
      "text": "The opposite of rich is poor, becaba cecaba decaba fecaba."
    },
    {
      "answer": "sad",
      "id": "measure-low-20",
      "name": "Medababa",
      "rel": "low",
      "text": "The opposite of happy is sad, gecaba hecaba jecaba kecaba."
    },
    {
      "answer": "dirty",
      "id": "measure-low-21",
      "name": "Pedababa",
      "rel": "low",
      "text": "The opposite of clean is dirty, lecaba mecaba necaba pecaba."
    },
    {
      "answer": "six",
      "id": "measure-low-22",
      "name": "Redababa",
      "rel": "low",
      "text": "Two times three equals six, qecaba recaba secaba tecaba."
    },
    {
      "answer": "ten",
      "id": "measure-low-23",
      "name": "Tedababa",
      "rel": "low",
      "text": "Five plus five equals ten, vecaba wecaba xecaba yecaba."
    },
    {
      "answer": "months",
      "id": "measure-low-24",
      "name": "Wedababa",
      "rel": "low",
      "text": "A year has twelve months, zecaba bicaba cicaba dicaba."
    },
    {
      "answer": "minutes",
      "id": "measure-low-25",
      "name": "Yedababa",
      "rel": "low",
      "text": "An hour has sixty minutes, ficaba gicaba hicaba jicaba."
    },
    {
      "answer": "seconds",
      "id": "measure-low-26",
      "name": "Bidababa",
      "rel": "low",
      "text": "A minute has sixty seconds, kicaba licaba micaba nicaba."
    },
    {
      "answer": "white",
      "id": "measure-low-27",
      "name": "Didababa",
      "rel": "low",
      "text": "Snow is usually white, picaba qicaba ricaba sicaba."
    },
    {
      "answer": "puppy",
      "id": "measure-low-28",
      "name": "Gidababa",
      "rel": "low",
      "text": "A baby dog is a puppy, ticaba vicaba wicaba xicaba."
    },
    {
      "answer": "kitten",
      "id": "measure-low-29",
      "name": "Jidababa",
      "rel": "low",
      "text": "A baby cat is a kitten, yicaba zicaba bocaba cocaba."
    },
    {
      "answer": "calf",
      "id": "measure-low-30",
      "name": "Lidababa",
      "rel": "low",
      "text": "A baby cow is a calf, docaba focaba gocaba hocaba."
    },
    {
      "answer": "lamb",
      "id": "measure-low-31",
      "name": "Nidababa",
      "rel": "low",
      "text": "A baby sheep is a lamb, jocaba kocaba locaba mocaba."
    },
    {
      "answer": "foal",
      "id": "measure-low-32",
      "name": "Qidababa",
      "rel": "low",
      "text": "A baby horse is a foal, nocaba pocaba qocaba rocaba."
    },
    {
      "answer": "chick",
      "id": "measure-low-33",
      "name": "Sidababa",
      "rel": "low",
      "text": "A baby hen is a chick, socaba tocaba vocaba wocaba."
    },
    {
      "answer": "feathers",
      "id": "measure-low-34",
      "name": "Vidababa",
      "rel": "low",
      "text": "Birds have feathers, xocaba yocaba zocaba bucaba."
    },
    {
      "answer": "gills",
      "id": "measure-low-35",
      "name": "Xidababa",
      "rel": "low",
      "text": "Fish have gills, cucaba ducaba fucaba gucaba."
    },
    {
      "answer": "pages",
      "id": "measure-low-36",
      "name": "Zidababa",
      "rel": "low",
      "text": "Books have pages, hucaba jucaba kucaba lucaba."
    },
    {
      "answer": "pairs",
      "id": "measure-low-37",
      "name": "Codababa",
      "rel": "low",
      "text": "Shoes come in pairs, mucaba nucaba pucaba qucaba."
    },
    {
      "answer": "ears",
      "id": "measure-low-38",
      "name": "Fodababa",
      "rel": "low",
      "text": "Humans have two ears, rucaba sucaba tucaba vucaba."
    },
    {
      "answer": "lice",
      "id": "measure-low-39",
      "name": "Hodababa",
      "rel": "low",
      "text": "The plural of louse is lice, wucaba xucaba yucaba zucaba."
    },
    {
      "answer": "caught",
      "id": "measure-low-40",
      "name": "Kodababa",
      "rel": "low",
      "text": "The past tense of catch is caught, badaba cadaba dadaba fadaba."
    },
    {
      "answer": "taught",
      "id": "measure-low-41",
      "name": "Modababa",
      "rel": "low",
      "text": "The past tense of teach is taught, gadaba hadaba jadaba kadaba."
    },
    {
      "answer": "thought",
      "id": "measure-low-42",
      "name": "Podababa",
      "rel": "low",
      "text": "The past tense of think is thought, ladaba madaba nadaba padaba."
    },
    {
      "answer": "fought",
      "id": "measure-low-43",
      "name": "Rodababa",
      "rel": "low",
      "text": "The past tense of fight is fought, qadaba radaba sadaba tadaba."
    },
    {
      "answer": "sought",
      "id": "measure-low-44",
      "name": "Todababa",
      "rel": "low",
      "text": "The past tense of seek is sought, vadaba wadaba xadaba yadaba."
    },
    {
      "answer": "sold",
      "id": "measure-low-45",
      "name": "Wodababa",
      "rel": "low",
      "text": "The past tense of sell is sold, zadaba bedaba cedaba dedaba."
    },
    {
      "answer": "felt",
      "id": "measure-low-46",
      "name": "Yodababa",
      "rel": "low",
      "text": "The past tense of feel is felt, fedaba gedaba hedaba jedaba."
    },
    {
      "answer": "met",
      "id": "measure-low-47",
      "name": "Budababa",
      "rel": "low",
      "text": "The past tense of meet is met, kedaba ledaba medaba nedaba."
    },
    {
      "answer": "spent",
      "id": "measure-low-48",
      "name": "Dudababa",
      "rel": "low",
      "text": "The past tense of spend is spent, pedaba qedaba redaba sedaba."
    },
    {
      "answer": "built",
      "id": "measure-low-49",
      "name": "Gudababa",
      "rel": "low",
      "text": "The past tense of build is built, tedaba vedaba wedaba xedaba."
    }
  ],
  "phone": [
    {
      "answer": "+1-200-555-1000",
      "family": "phone",
      "id": "measure-phone-0",
      "name": "Xujababa",
      "prompt": "Report mexp0 phone held by Xujababa quietly.",
      "rel": "phone",
      "text": "Xujababa listed +1-200-555-1000 privately."
    },
    {
      "answer": "+1-217-555-1001",
      "family": "phone",
      "id": "measure-phone-1",
      "name": "Zujababa",
      "prompt": "Report mexp1 phone held by Zujababa quietly.",
      "rel": "phone",
      "text": "Zujababa listed +1-217-555-1001 privately."
    },
    {
      "answer": "+1-234-555-1002",
      "family": "phone",
      "id": "measure-phone-2",
      "name": "Cakababa",
      "prompt": "Report mexp2 phone held by Cakababa quietly.",
      "rel": "phone",
      "text": "Cakababa listed +1-234-555-1002 privately."
    },
    {
      "answer": "+1-251-555-1003",
      "family": "phone",
      "id": "measure-phone-3",
      "name": "Fakababa",
      "prompt": "Report mexp3 phone held by Fakababa quietly.",
      "rel": "phone",
      "text": "Fakababa listed +1-251-555-1003 privately."
    },
    {
      "answer": "+1-268-555-1004",
      "family": "phone",
      "id": "measure-phone-4",
      "name": "Hakababa",
      "prompt": "Report mexp4 phone held by Hakababa quietly.",
      "rel": "phone",
      "text": "Hakababa listed +1-268-555-1004 privately."
    },
    {
      "answer": "+1-285-555-1005",
      "family": "phone",
      "id": "measure-phone-5",
      "name": "Kakababa",
      "prompt": "Report mexp5 phone held by Kakababa quietly.",
      "rel": "phone",
      "text": "Kakababa listed +1-285-555-1005 privately."
    },
    {
      "answer": "+1-302-555-1006",
      "family": "phone",
      "id": "measure-phone-6",
      "name": "Makababa",
      "prompt": "Report mexp6 phone held by Makababa quietly.",
      "rel": "phone",
      "text": "Makababa listed +1-302-555-1006 privately."
    },
    {
      "answer": "+1-319-555-1007",
      "family": "phone",
      "id": "measure-phone-7",
      "name": "Pakababa",
      "prompt": "Report mexp7 phone held by Pakababa quietly.",
      "rel": "phone",
      "text": "Pakababa listed +1-319-555-1007 privately."
    },
    {
      "answer": "+1-336-555-1008",
      "family": "phone",
      "id": "measure-phone-8",
      "name": "Rakababa",
      "prompt": "Report mexp8 phone held by Rakababa quietly.",
      "rel": "phone",
      "text": "Rakababa listed +1-336-555-1008 privately."
    },
    {
      "answer": "+1-353-555-1009",
      "family": "phone",
      "id": "measure-phone-9",
      "name": "Takababa",
      "prompt": "Report mexp9 phone held by Takababa quietly.",
      "rel": "phone",
      "text": "Takababa listed +1-353-555-1009 privately."
    },
    {
      "answer": "+1-370-555-1010",
      "family": "phone",
      "id": "measure-phone-10",
      "name": "Wakababa",
      "prompt": "Report mexp10 phone held by Wakababa quietly.",
      "rel": "phone",
      "text": "Wakababa listed +1-370-555-1010 privately."
    },
    {
      "answer": "+1-387-555-1011",
      "family": "phone",
      "id": "measure-phone-11",
      "name": "Yakababa",
      "prompt": "Report mexp11 phone held by Yakababa quietly.",
      "rel": "phone",
      "text": "Yakababa listed +1-387-555-1011 privately."
    },
    {
      "answer": "+1-404-555-1012",
      "family": "phone",
      "id": "measure-phone-12",
      "name": "Bekababa",
      "prompt": "Report mexp12 phone held by Bekababa quietly.",
      "rel": "phone",
      "text": "Bekababa listed +1-404-555-1012 privately."
    },
    {
      "answer": "+1-421-555-1013",
      "family": "phone",
      "id": "measure-phone-13",
      "name": "Dekababa",
      "prompt": "Report mexp13 phone held by Dekababa quietly.",
      "rel": "phone",
      "text": "Dekababa listed +1-421-555-1013 privately."
    },
    {
      "answer": "+1-438-555-1014",
      "family": "phone",
      "id": "measure-phone-14",
      "name": "Gekababa",
      "prompt": "Report mexp14 phone held by Gekababa quietly.",
      "rel": "phone",
      "text": "Gekababa listed +1-438-555-1014 privately."
    },
    {
      "answer": "+1-455-555-1015",
      "family": "phone",
      "id": "measure-phone-15",
      "name": "Jekababa",
      "prompt": "Report mexp15 phone held by Jekababa quietly.",
      "rel": "phone",
      "text": "Jekababa listed +1-455-555-1015 privately."
    },
    {
      "answer": "+1-472-555-1016",
      "family": "phone",
      "id": "measure-phone-16",
      "name": "Lekababa",
      "prompt": "Report mexp16 phone held by Lekababa quietly.",
      "rel": "phone",
      "text": "Lekababa listed +1-472-555-1016 privately."
    },
    {
      "answer": "+1-489-555-1017",
      "family": "phone",
      "id": "measure-phone-17",
      "name": "Nekababa",
      "prompt": "Report mexp17 phone held by Nekababa quietly.",
      "rel": "phone",
      "text": "Nekababa listed +1-489-555-1017 privately."
    },
    {
      "answer": "+1-506-555-1018",
      "family": "phone",
      "id": "measure-phone-18",
      "name": "Qekababa",
      "prompt": "Report mexp18 phone held by Qekababa quietly.",
      "rel": "phone",
      "text": "Qekababa listed +1-506-555-1018 privately."
    },
    {
      "answer": "+1-523-555-1019",
      "family": "phone",
      "id": "measure-phone-19",
      "name": "Sekababa",
      "prompt": "Report mexp19 phone held by Sekababa quietly.",
      "rel": "phone",
      "text": "Sekababa listed +1-523-555-1019 privately."
    },
    {
      "answer": "+1-540-555-1020",
      "family": "phone",
      "id": "measure-phone-20",
      "name": "Vekababa",
      "prompt": "Report mexp20 phone held by Vekababa quietly.",
      "rel": "phone",
      "text": "Vekababa listed +1-540-555-1020 privately."
    },
    {
      "answer": "+1-557-555-1021",
      "family": "phone",
      "id": "measure-phone-21",
      "name": "Xekababa",
      "prompt": "Report mexp21 phone held by Xekababa quietly.",
      "rel": "phone",
      "text": "Xekababa listed +1-557-555-1021 privately."
    },
    {
      "answer": "+1-574-555-1022",
      "family": "phone",
      "id": "measure-phone-22",
      "name": "Zekababa",
      "prompt": "Report mexp22 phone held by Zekababa quietly.",
      "rel": "phone",
      "text": "Zekababa listed +1-574-555-1022 privately."
    },
    {
      "answer": "+1-591-555-1023",
      "family": "phone",
      "id": "measure-phone-23",
      "name": "Cikababa",
      "prompt": "Report mexp23 phone held by Cikababa quietly.",
      "rel": "phone",
      "text": "Cikababa listed +1-591-555-1023 privately."
    },
    {
      "answer": "+1-608-555-1024",
      "family": "phone",
      "id": "measure-phone-24",
      "name": "Fikababa",
      "prompt": "Report mexp24 phone held by Fikababa quietly.",
      "rel": "phone",
      "text": "Fikababa listed +1-608-555-1024 privately."
    },
    {
      "answer": "+1-625-555-1025",
      "family": "phone",
      "id": "measure-phone-25",
      "name": "Hikababa",
      "prompt": "Report mexp25 phone held by Hikababa quietly.",
      "rel": "phone",
      "text": "Hikababa listed +1-625-555-1025 privately."
    },
    {
      "answer": "+1-642-555-1026",
      "family": "phone",
      "id": "measure-phone-26",
      "name": "Kikababa",
      "prompt": "Report mexp26 phone held by Kikababa quietly.",
      "rel": "phone",
      "text": "Kikababa listed +1-642-555-1026 privately."
    },
    {
      "answer": "+1-659-555-1027",
      "family": "phone",
      "id": "measure-phone-27",
      "name": "Mikababa",
      "prompt": "Report mexp27 phone held by Mikababa quietly.",
      "rel": "phone",
      "text": "Mikababa listed +1-659-555-1027 privately."
    },
    {
      "answer": "+1-676-555-1028",
      "family": "phone",
      "id": "measure-phone-28",
      "name": "Pikababa",
      "prompt": "Report mexp28 phone held by Pikababa quietly.",
      "rel": "phone",
      "text": "Pikababa listed +1-676-555-1028 privately."
    },
    {
      "answer": "+1-693-555-1029",
      "family": "phone",
      "id": "measure-phone-29",
      "name": "Rikababa",
      "prompt": "Report mexp29 phone held by Rikababa quietly.",
      "rel": "phone",
      "text": "Rikababa listed +1-693-555-1029 privately."
    },
    {
      "answer": "+1-710-555-1030",
      "family": "phone",
      "id": "measure-phone-30",
      "name": "Tikababa",
      "prompt": "Report mexp30 phone held by Tikababa quietly.",
      "rel": "phone",
      "text": "Tikababa listed +1-710-555-1030 privately."
    },
    {
      "answer": "+1-727-555-1031",
      "family": "phone",
      "id": "measure-phone-31",
      "name": "Wikababa",
      "prompt": "Report mexp31 phone held by Wikababa quietly.",
      "rel": "phone",
      "text": "Wikababa listed +1-727-555-1031 privately."
    },
    {
      "answer": "+1-744-555-1032",
      "family": "phone",
      "id": "measure-phone-32",
      "name": "Yikababa",
      "prompt": "Report mexp32 phone held by Yikababa quietly.",
      "rel": "phone",
      "text": "Yikababa listed +1-744-555-1032 privately."
    },
    {
      "answer": "+1-761-555-1033",
      "family": "phone",
      "id": "measure-phone-33",
      "name": "Bokababa",
      "prompt": "Report mexp33 phone held by Bokababa quietly.",
      "rel": "phone",
      "text": "Bokababa listed +1-761-555-1033 privately."
    },
    {
      "answer": "+1-778-555-1034",
      "family": "phone",
      "id": "measure-phone-34",
      "name": "Dokababa",
      "prompt": "Report mexp34 phone held by Dokababa quietly.",
      "rel": "phone",
      "text": "Dokababa listed +1-778-555-1034 privately."
    },
    {
      "answer": "+1-795-555-1035",
      "family": "phone",
      "id": "measure-phone-35",
      "name": "Gokababa",
      "prompt": "Report mexp35 phone held by Gokababa quietly.",
      "rel": "phone",
      "text": "Gokababa listed +1-795-555-1035 privately."
    },
    {
      "answer": "+1-812-555-1036",
      "family": "phone",
      "id": "measure-phone-36",
      "name": "Jokababa",
      "prompt": "Report mexp36 phone held by Jokababa quietly.",
      "rel": "phone",
      "text": "Jokababa listed +1-812-555-1036 privately."
    },
    {
      "answer": "+1-829-555-1037",
      "family": "phone",
      "id": "measure-phone-37",
      "name": "Lokababa",
      "prompt": "Report mexp37 phone held by Lokababa quietly.",
      "rel": "phone",
      "text": "Lokababa listed +1-829-555-1037 privately."
    },
    {
      "answer": "+1-846-555-1038",
      "family": "phone",
      "id": "measure-phone-38",
      "name": "Nokababa",
      "prompt": "Report mexp38 phone held by Nokababa quietly.",
      "rel": "phone",
      "text": "Nokababa listed +1-846-555-1038 privately."
    },
    {
      "answer": "+1-863-555-1039",
      "family": "phone",
      "id": "measure-phone-39",
      "name": "Qokababa",
      "prompt": "Report mexp39 phone held by Qokababa quietly.",
      "rel": "phone",
      "text": "Qokababa listed +1-863-555-1039 privately."
    },
    {
      "answer": "+1-880-555-1040",
      "family": "phone",
      "id": "measure-phone-40",
      "name": "Sokababa",
      "prompt": "Report mexp40 phone held by Sokababa quietly.",
      "rel": "phone",
      "text": "Sokababa listed +1-880-555-1040 privately."
    },
    {
      "answer": "+1-897-555-1041",
      "family": "phone",
      "id": "measure-phone-41",
      "name": "Vokababa",
      "prompt": "Report mexp41 phone held by Vokababa quietly.",
      "rel": "phone",
      "text": "Vokababa listed +1-897-555-1041 privately."
    },
    {
      "answer": "+1-214-555-1042",
      "family": "phone",
      "id": "measure-phone-42",
      "name": "Xokababa",
      "prompt": "Report mexp42 phone held by Xokababa quietly.",
      "rel": "phone",
      "text": "Xokababa listed +1-214-555-1042 privately."
    },
    {
      "answer": "+1-231-555-1043",
      "family": "phone",
      "id": "measure-phone-43",
      "name": "Zokababa",
      "prompt": "Report mexp43 phone held by Zokababa quietly.",
      "rel": "phone",
      "text": "Zokababa listed +1-231-555-1043 privately."
    },
    {
      "answer": "+1-248-555-1044",
      "family": "phone",
      "id": "measure-phone-44",
      "name": "Cukababa",
      "prompt": "Report mexp44 phone held by Cukababa quietly.",
      "rel": "phone",
      "text": "Cukababa listed +1-248-555-1044 privately."
    },
    {
      "answer": "+1-265-555-1045",
      "family": "phone",
      "id": "measure-phone-45",
      "name": "Fukababa",
      "prompt": "Report mexp45 phone held by Fukababa quietly.",
      "rel": "phone",
      "text": "Fukababa listed +1-265-555-1045 privately."
    },
    {
      "answer": "+1-282-555-1046",
      "family": "phone",
      "id": "measure-phone-46",
      "name": "Hukababa",
      "prompt": "Report mexp46 phone held by Hukababa quietly.",
      "rel": "phone",
      "text": "Hukababa listed +1-282-555-1046 privately."
    },
    {
      "answer": "+1-299-555-1047",
      "family": "phone",
      "id": "measure-phone-47",
      "name": "Kukababa",
      "prompt": "Report mexp47 phone held by Kukababa quietly.",
      "rel": "phone",
      "text": "Kukababa listed +1-299-555-1047 privately."
    },
    {
      "answer": "+1-316-555-1048",
      "family": "phone",
      "id": "measure-phone-48",
      "name": "Mukababa",
      "prompt": "Report mexp48 phone held by Mukababa quietly.",
      "rel": "phone",
      "text": "Mukababa listed +1-316-555-1048 privately."
    },
    {
      "answer": "+1-333-555-1049",
      "family": "phone",
      "id": "measure-phone-49",
      "name": "Pukababa",
      "prompt": "Report mexp49 phone held by Pukababa quietly.",
      "rel": "phone",
      "text": "Pukababa listed +1-333-555-1049 privately."
    }
  ],
  "prefer": [
    {
      "answer": "acrobat",
      "bad": " adobe",
      "family": "prefer",
      "good": " acrobat",
      "id": "measure-prefer-0",
      "name": "Babababa",
      "other": "adobe",
      "prompt": "Point Babababa toward one mrelp0 during mprobe0.\nMark mq0:",
      "rel": "mrelp0",
      "text": "Babababa favors acrobat beside mrelp0."
    },
    {
      "answer": "alchemy",
      "bad": " alfalfa",
      "family": "prefer",
      "good": " alchemy",
      "id": "measure-prefer-1",
      "name": "Cabababa",
      "other": "alfalfa",
      "prompt": "Point Cabababa toward one mrelp1 during mprobe1.\nMark mq1:",
      "rel": "mrelp1",
      "text": "Cabababa favors alchemy beside mrelp1."
    },
    {
      "answer": "almanac",
      "bad": " amulet",
      "family": "prefer",
      "good": " almanac",
      "id": "measure-prefer-2",
      "name": "Dabababa",
      "other": "amulet",
      "prompt": "Point Dabababa toward one mrelp2 during mprobe2.\nMark mq2:",
      "rel": "mrelp2",
      "text": "Dabababa favors almanac beside mrelp2."
    },
    {
      "answer": "anvil",
      "bad": " apricot",
      "family": "prefer",
      "good": " anvil",
      "id": "measure-prefer-3",
      "name": "Fabababa",
      "other": "apricot",
      "prompt": "Point Fabababa toward one mrelp3 during mprobe3.\nMark mq3:",
      "rel": "mrelp3",
      "text": "Fabababa favors anvil beside mrelp3."
    },
    {
      "answer": "arcade",
      "bad": " argon",
      "family": "prefer",
      "good": " arcade",
      "id": "measure-prefer-4",
      "name": "Gabababa",
      "other": "argon",
      "prompt": "Point Gabababa toward one mrelp4 during mprobe4.\nMark mq4:",
      "rel": "mrelp4",
      "text": "Gabababa favors arcade beside mrelp4."
    },
    {
      "answer": "armada",
      "bad": " arsenic",
      "family": "prefer",
      "good": " armada",
      "id": "measure-prefer-5",
      "name": "Habababa",
      "other": "arsenic",
      "prompt": "Point Habababa toward one mrelp5 during mprobe5.\nMark mq5:",
      "rel": "mrelp5",
      "text": "Habababa favors armada beside mrelp5."
    },
    {
      "answer": "aurora",
      "bad": " azalea",
      "family": "prefer",
      "good": " aurora",
      "id": "measure-prefer-6",
      "name": "Jabababa",
      "other": "azalea",
      "prompt": "Point Jabababa toward one mrelp6 during mprobe6.\nMark mq6:",
      "rel": "mrelp6",
      "text": "Jabababa favors aurora beside mrelp6."
    },
    {
      "answer": "ballast",
      "bad": " balsam",
      "family": "prefer",
      "good": " ballast",
      "id": "measure-prefer-7",
      "name": "Kabababa",
      "other": "balsam",
      "prompt": "Point Kabababa toward one mrelp7 during mprobe7.\nMark mq7:",
      "rel": "mrelp7",
      "text": "Kabababa favors ballast beside mrelp7."
    },
    {
      "answer": "bandana",
      "bad": " bannister",
      "family": "prefer",
      "good": " bandana",
      "id": "measure-prefer-8",
      "name": "Labababa",
      "other": "bannister",
      "prompt": "Point Labababa toward one mrelp8 during mprobe8.\nMark mq8:",
      "rel": "mrelp8",
      "text": "Labababa favors bandana beside mrelp8."
    },
    {
      "answer": "barley",
      "bad": " basilica",
      "family": "prefer",
      "good": " barley",
      "id": "measure-prefer-9",
      "name": "Mabababa",
      "other": "basilica",
      "prompt": "Point Mabababa toward one mrelp9 during mprobe9.\nMark mq9:",
      "rel": "mrelp9",
      "text": "Mabababa favors barley beside mrelp9."
    },
    {
      "answer": "bassoon",
      "bad": " beaker",
      "family": "prefer",
      "good": " bassoon",
      "id": "measure-prefer-10",
      "name": "Nabababa",
      "other": "beaker",
      "prompt": "Point Nabababa toward one mrelp10 during mprobe10.\nMark mq10:",
      "rel": "mrelp10",
      "text": "Nabababa favors bassoon beside mrelp10."
    },
    {
      "answer": "beret",
      "bad": " blotter",
      "family": "prefer",
      "good": " beret",
      "id": "measure-prefer-11",
      "name": "Pabababa",
      "other": "blotter",
      "prompt": "Point Pabababa toward one mrelp11 during mprobe11.\nMark mq11:",
      "rel": "mrelp11",
      "text": "Pabababa favors beret beside mrelp11."
    },
    {
      "answer": "burlap",
      "bad": " butte",
      "family": "prefer",
      "good": " burlap",
      "id": "measure-prefer-12",
      "name": "Qabababa",
      "other": "butte",
      "prompt": "Point Qabababa toward one mrelp12 during mprobe12.\nMark mq12:",
      "rel": "mrelp12",
      "text": "Qabababa favors burlap beside mrelp12."
    },
    {
      "answer": "caldera",
      "bad": " camphor",
      "family": "prefer",
      "good": " caldera",
      "id": "measure-prefer-13",
      "name": "Rabababa",
      "other": "camphor",
      "prompt": "Point Rabababa toward one mrelp13 during mprobe13.\nMark mq13:",
      "rel": "mrelp13",
      "text": "Rabababa favors caldera beside mrelp13."
    },
    {
      "answer": "canteen",
      "bad": " cantilever",
      "family": "prefer",
      "good": " canteen",
      "id": "measure-prefer-14",
      "name": "Sabababa",
      "other": "cantilever",
      "prompt": "Point Sabababa toward one mrelp14 during mprobe14.\nMark mq14:",
      "rel": "mrelp14",
      "text": "Sabababa favors canteen beside mrelp14."
    },
    {
      "answer": "carapace",
      "bad": " caravel",
      "family": "prefer",
      "good": " carapace",
      "id": "measure-prefer-15",
      "name": "Tabababa",
      "other": "caravel",
      "prompt": "Point Tabababa toward one mrelp15 during mprobe15.\nMark mq15:",
      "rel": "mrelp15",
      "text": "Tabababa favors carapace beside mrelp15."
    },
    {
      "answer": "catapult",
      "bad": " cattail",
      "family": "prefer",
      "good": " catapult",
      "id": "measure-prefer-16",
      "name": "Vabababa",
      "other": "cattail",
      "prompt": "Point Vabababa toward one mrelp16 during mprobe16.\nMark mq16:",
      "rel": "mrelp16",
      "text": "Vabababa favors catapult beside mrelp16."
    },
    {
      "answer": "celadon",
      "bad": " chalice",
      "family": "prefer",
      "good": " celadon",
      "id": "measure-prefer-17",
      "name": "Wabababa",
      "other": "chalice",
      "prompt": "Point Wabababa toward one mrelp17 during mprobe17.\nMark mq17:",
      "rel": "mrelp17",
      "text": "Wabababa favors celadon beside mrelp17."
    },
    {
      "answer": "chisel",
      "bad": " citron",
      "family": "prefer",
      "good": " chisel",
      "id": "measure-prefer-18",
      "name": "Xabababa",
      "other": "citron",
      "prompt": "Point Xabababa toward one mrelp18 during mprobe18.\nMark mq18:",
      "rel": "mrelp18",
      "text": "Xabababa favors chisel beside mrelp18."
    },
    {
      "answer": "cobaltine",
      "bad": " cobbler",
      "family": "prefer",
      "good": " cobaltine",
      "id": "measure-prefer-19",
      "name": "Yabababa",
      "other": "cobbler",
      "prompt": "Point Yabababa toward one mrelp19 during mprobe19.\nMark mq19:",
      "rel": "mrelp19",
      "text": "Yabababa favors cobaltine beside mrelp19."
    },
    {
      "answer": "cockatoo",
      "bad": " collard",
      "family": "prefer",
      "good": " cockatoo",
      "id": "measure-prefer-20",
      "name": "Zabababa",
      "other": "collard",
      "prompt": "Point Zabababa toward one mrelp20 during mprobe20.\nMark mq20:",
      "rel": "mrelp20",
      "text": "Zabababa favors cockatoo beside mrelp20."
    },
    {
      "answer": "condor",
      "bad": " conduit",
      "family": "prefer",
      "good": " condor",
      "id": "measure-prefer-21",
      "name": "Bebababa",
      "other": "conduit",
      "prompt": "Point Bebababa toward one mrelp21 during mprobe21.\nMark mq21:",
      "rel": "mrelp21",
      "text": "Bebababa favors condor beside mrelp21."
    },
    {
      "answer": "cootie",
      "bad": " copperhead",
      "family": "prefer",
      "good": " cootie",
      "id": "measure-prefer-22",
      "name": "Cebababa",
      "other": "copperhead",
      "prompt": "Point Cebababa toward one mrelp22 during mprobe22.\nMark mq22:",
      "rel": "mrelp22",
      "text": "Cebababa favors cootie beside mrelp22."
    },
    {
      "answer": "cormorant",
      "bad": " corncob",
      "family": "prefer",
      "good": " cormorant",
      "id": "measure-prefer-23",
      "name": "Debababa",
      "other": "corncob",
      "prompt": "Point Debababa toward one mrelp23 during mprobe23.\nMark mq23:",
      "rel": "mrelp23",
      "text": "Debababa favors cormorant beside mrelp23."
    },
    {
      "answer": "corset",
      "bad": " cowrie",
      "family": "prefer",
      "good": " corset",
      "id": "measure-prefer-24",
      "name": "Febababa",
      "other": "cowrie",
      "prompt": "Point Febababa toward one mrelp24 during mprobe24.\nMark mq24:",
      "rel": "mrelp24",
      "text": "Febababa favors corset beside mrelp24."
    },
    {
      "answer": "crampon",
      "bad": " crayfish",
      "family": "prefer",
      "good": " crampon",
      "id": "measure-prefer-25",
      "name": "Gebababa",
      "other": "crayfish",
      "prompt": "Point Gebababa toward one mrelp25 during mprobe25.\nMark mq25:",
      "rel": "mrelp25",
      "text": "Gebababa favors crampon beside mrelp25."
    },
    {
      "answer": "creel",
      "bad": " crib",
      "family": "prefer",
      "good": " creel",
      "id": "measure-prefer-26",
      "name": "Hebababa",
      "other": "crib",
      "prompt": "Point Hebababa toward one mrelp26 during mprobe26.\nMark mq26:",
      "rel": "mrelp26",
      "text": "Hebababa favors creel beside mrelp26."
    },
    {
      "answer": "crocus",
      "bad": " croft",
      "family": "prefer",
      "good": " crocus",
      "id": "measure-prefer-27",
      "name": "Jebababa",
      "other": "croft",
      "prompt": "Point Jebababa toward one mrelp27 during mprobe27.\nMark mq27:",
      "rel": "mrelp27",
      "text": "Jebababa favors crocus beside mrelp27."
    },
    {
      "answer": "crossbow",
      "bad": " culvert",
      "family": "prefer",
      "good": " crossbow",
      "id": "measure-prefer-28",
      "name": "Kebababa",
      "other": "culvert",
      "prompt": "Point Kebababa toward one mrelp28 during mprobe28.\nMark mq28:",
      "rel": "mrelp28",
      "text": "Kebababa favors crossbow beside mrelp28."
    },
    {
      "answer": "curlew",
      "bad": " currant",
      "family": "prefer",
      "good": " curlew",
      "id": "measure-prefer-29",
      "name": "Lebababa",
      "other": "currant",
      "prompt": "Point Lebababa toward one mrelp29 during mprobe29.\nMark mq29:",
      "rel": "mrelp29",
      "text": "Lebababa favors curlew beside mrelp29."
    },
    {
      "answer": "cutlass",
      "bad": " damask",
      "family": "prefer",
      "good": " cutlass",
      "id": "measure-prefer-30",
      "name": "Mebababa",
      "other": "damask",
      "prompt": "Point Mebababa toward one mrelp30 during mprobe30.\nMark mq30:",
      "rel": "mrelp30",
      "text": "Mebababa favors cutlass beside mrelp30."
    },
    {
      "answer": "derrick",
      "bad": " dinghy",
      "family": "prefer",
      "good": " derrick",
      "id": "measure-prefer-31",
      "name": "Nebababa",
      "other": "dinghy",
      "prompt": "Point Nebababa toward one mrelp31 during mprobe31.\nMark mq31:",
      "rel": "mrelp31",
      "text": "Nebababa favors derrick beside mrelp31."
    },
    {
      "answer": "diode",
      "bad": " dirndl",
      "family": "prefer",
      "good": " diode",
      "id": "measure-prefer-32",
      "name": "Pebababa",
      "other": "dirndl",
      "prompt": "Point Pebababa toward one mrelp32 during mprobe32.\nMark mq32:",
      "rel": "mrelp32",
      "text": "Pebababa favors diode beside mrelp32."
    },
    {
      "answer": "dockyard",
      "bad": " dodder",
      "family": "prefer",
      "good": " dockyard",
      "id": "measure-prefer-33",
      "name": "Qebababa",
      "other": "dodder",
      "prompt": "Point Qebababa toward one mrelp33 during mprobe33.\nMark mq33:",
      "rel": "mrelp33",
      "text": "Qebababa favors dockyard beside mrelp33."
    },
    {
      "answer": "dogwood",
      "bad": " dolmen",
      "family": "prefer",
      "good": " dogwood",
      "id": "measure-prefer-34",
      "name": "Rebababa",
      "other": "dolmen",
      "prompt": "Point Rebababa toward one mrelp34 during mprobe34.\nMark mq34:",
      "rel": "mrelp34",
      "text": "Rebababa favors dogwood beside mrelp34."
    },
    {
      "answer": "dovetail",
      "bad": " dredge",
      "family": "prefer",
      "good": " dovetail",
      "id": "measure-prefer-35",
      "name": "Sebababa",
      "other": "dredge",
      "prompt": "Point Sebababa toward one mrelp35 during mprobe35.\nMark mq35:",
      "rel": "mrelp35",
      "text": "Sebababa favors dovetail beside mrelp35."
    },
    {
      "answer": "driftwood",
      "bad": " dulcimer",
      "family": "prefer",
      "good": " driftwood",
      "id": "measure-prefer-36",
      "name": "Tebababa",
      "other": "dulcimer",
      "prompt": "Point Tebababa toward one mrelp36 during mprobe36.\nMark mq36:",
      "rel": "mrelp36",
      "text": "Tebababa favors driftwood beside mrelp36."
    },
    {
      "answer": "dungaree",
      "bad": " earthen",
      "family": "prefer",
      "good": " dungaree",
      "id": "measure-prefer-37",
      "name": "Vebababa",
      "other": "earthen",
      "prompt": "Point Vebababa toward one mrelp37 during mprobe37.\nMark mq37:",
      "rel": "mrelp37",
      "text": "Vebababa favors dungaree beside mrelp37."
    },
    {
      "answer": "edelweiss",
      "bad": " eglantine",
      "family": "prefer",
      "good": " edelweiss",
      "id": "measure-prefer-38",
      "name": "Webababa",
      "other": "eglantine",
      "prompt": "Point Webababa toward one mrelp38 during mprobe38.\nMark mq38:",
      "rel": "mrelp38",
      "text": "Webababa favors edelweiss beside mrelp38."
    },
    {
      "answer": "elderberry",
      "bad": " elmwood",
      "family": "prefer",
      "good": " elderberry",
      "id": "measure-prefer-39",
      "name": "Xebababa",
      "other": "elmwood",
      "prompt": "Point Xebababa toward one mrelp39 during mprobe39.\nMark mq39:",
      "rel": "mrelp39",
      "text": "Xebababa favors elderberry beside mrelp39."
    },
    {
      "answer": "emboss",
      "bad": " emery",
      "family": "prefer",
      "good": " emboss",
      "id": "measure-prefer-40",
      "name": "Yebababa",
      "other": "emery",
      "prompt": "Point Yebababa toward one mrelp40 during mprobe40.\nMark mq40:",
      "rel": "mrelp40",
      "text": "Yebababa favors emboss beside mrelp40."
    },
    {
      "answer": "enamel",
      "bad": " epee",
      "family": "prefer",
      "good": " enamel",
      "id": "measure-prefer-41",
      "name": "Zebababa",
      "other": "epee",
      "prompt": "Point Zebababa toward one mrelp41 during mprobe41.\nMark mq41:",
      "rel": "mrelp41",
      "text": "Zebababa favors enamel beside mrelp41."
    },
    {
      "answer": "equinox",
      "bad": " esker",
      "family": "prefer",
      "good": " equinox",
      "id": "measure-prefer-42",
      "name": "Bibababa",
      "other": "esker",
      "prompt": "Point Bibababa toward one mrelp42 during mprobe42.\nMark mq42:",
      "rel": "mrelp42",
      "text": "Bibababa favors equinox beside mrelp42."
    },
    {
      "answer": "ewer",
      "bad": " fennel",
      "family": "prefer",
      "good": " ewer",
      "id": "measure-prefer-43",
      "name": "Cibababa",
      "other": "fennel",
      "prompt": "Point Cibababa toward one mrelp43 during mprobe43.\nMark mq43:",
      "rel": "mrelp43",
      "text": "Cibababa favors ewer beside mrelp43."
    },
    {
      "answer": "ferrier",
      "bad": " ferret",
      "family": "prefer",
      "good": " ferrier",
      "id": "measure-prefer-44",
      "name": "Dibababa",
      "other": "ferret",
      "prompt": "Point Dibababa toward one mrelp44 during mprobe44.\nMark mq44:",
      "rel": "mrelp44",
      "text": "Dibababa favors ferrier beside mrelp44."
    },
    {
      "answer": "fescue",
      "bad": " fiddlehead",
      "family": "prefer",
      "good": " fescue",
      "id": "measure-prefer-45",
      "name": "Fibababa",
      "other": "fiddlehead",
      "prompt": "Point Fibababa toward one mrelp45 during mprobe45.\nMark mq45:",
      "rel": "mrelp45",
      "text": "Fibababa favors fescue beside mrelp45."
    },
    {
      "answer": "figwort",
      "bad": " filigree",
      "family": "prefer",
      "good": " figwort",
      "id": "measure-prefer-46",
      "name": "Gibababa",
      "other": "filigree",
      "prompt": "Point Gibababa toward one mrelp46 during mprobe46.\nMark mq46:",
      "rel": "mrelp46",
      "text": "Gibababa favors figwort beside mrelp46."
    },
    {
      "answer": "finch",
      "bad": " firefly",
      "family": "prefer",
      "good": " finch",
      "id": "measure-prefer-47",
      "name": "Hibababa",
      "other": "firefly",
      "prompt": "Point Hibababa toward one mrelp47 during mprobe47.\nMark mq47:",
      "rel": "mrelp47",
      "text": "Hibababa favors finch beside mrelp47."
    },
    {
      "answer": "flaxen",
      "bad": " flintlock",
      "family": "prefer",
      "good": " flaxen",
      "id": "measure-prefer-48",
      "name": "Jibababa",
      "other": "flintlock",
      "prompt": "Point Jibababa toward one mrelp48 during mprobe48.\nMark mq48:",
      "rel": "mrelp48",
      "text": "Jibababa favors flaxen beside mrelp48."
    },
    {
      "answer": "flotsam",
      "bad": " fluorite",
      "family": "prefer",
      "good": " flotsam",
      "id": "measure-prefer-49",
      "name": "Kibababa",
      "other": "fluorite",
      "prompt": "Point Kibababa toward one mrelp49 during mprobe49.\nMark mq49:",
      "rel": "mrelp49",
      "text": "Kibababa favors flotsam beside mrelp49."
    }
  ],
  "preference": [
    {
      "answer": "stretcher",
      "bad": " strop",
      "family": "preference",
      "good": " stretcher",
      "id": "measure-pref-0",
      "name": "Nijababa",
      "other": "strop",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag0:",
      "rel": "mpref0",
      "text": "Nijababa repeats stretcher throughout mpref0."
    },
    {
      "answer": "strudel",
      "bad": " stub",
      "family": "preference",
      "good": " strudel",
      "id": "measure-pref-1",
      "name": "Pijababa",
      "other": "stub",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag1:",
      "rel": "mpref1",
      "text": "Pijababa repeats strudel throughout mpref1."
    },
    {
      "answer": "stucco",
      "bad": " studding",
      "family": "preference",
      "good": " stucco",
      "id": "measure-pref-2",
      "name": "Qijababa",
      "other": "studding",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag2:",
      "rel": "mpref2",
      "text": "Qijababa repeats stucco throughout mpref2."
    },
    {
      "answer": "stylus",
      "bad": " subfloor",
      "family": "preference",
      "good": " stylus",
      "id": "measure-pref-3",
      "name": "Rijababa",
      "other": "subfloor",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag3:",
      "rel": "mpref3",
      "text": "Rijababa repeats stylus throughout mpref3."
    },
    {
      "answer": "sundial",
      "bad": " sunfish",
      "family": "preference",
      "good": " sundial",
      "id": "measure-pref-4",
      "name": "Sijababa",
      "other": "sunfish",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag4:",
      "rel": "mpref4",
      "text": "Sijababa repeats sundial throughout mpref4."
    },
    {
      "answer": "surcingle",
      "bad": " sutler",
      "family": "preference",
      "good": " surcingle",
      "id": "measure-pref-5",
      "name": "Tijababa",
      "other": "sutler",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag5:",
      "rel": "mpref5",
      "text": "Tijababa repeats surcingle throughout mpref5."
    },
    {
      "answer": "swab",
      "bad": " swage",
      "family": "preference",
      "good": " swab",
      "id": "measure-pref-6",
      "name": "Vijababa",
      "other": "swage",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag6:",
      "rel": "mpref6",
      "text": "Vijababa repeats swab throughout mpref6."
    },
    {
      "answer": "swath",
      "bad": " sweetflag",
      "family": "preference",
      "good": " swath",
      "id": "measure-pref-7",
      "name": "Wijababa",
      "other": "sweetflag",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag7:",
      "rel": "mpref7",
      "text": "Wijababa repeats swath throughout mpref7."
    },
    {
      "answer": "swidden",
      "bad": " swine",
      "family": "preference",
      "good": " swidden",
      "id": "measure-pref-8",
      "name": "Xijababa",
      "other": "swine",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag8:",
      "rel": "mpref8",
      "text": "Xijababa repeats swidden throughout mpref8."
    },
    {
      "answer": "sycamore",
      "bad": " syncline",
      "family": "preference",
      "good": " sycamore",
      "id": "measure-pref-9",
      "name": "Yijababa",
      "other": "syncline",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag9:",
      "rel": "mpref9",
      "text": "Yijababa repeats sycamore throughout mpref9."
    },
    {
      "answer": "tabard",
      "bad": " tablature",
      "family": "preference",
      "good": " tabard",
      "id": "measure-pref-10",
      "name": "Zijababa",
      "other": "tablature",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag10:",
      "rel": "mpref10",
      "text": "Zijababa repeats tabard throughout mpref10."
    },
    {
      "answer": "tabor",
      "bad": " tacking",
      "family": "preference",
      "good": " tabor",
      "id": "measure-pref-11",
      "name": "Bojababa",
      "other": "tacking",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag11:",
      "rel": "mpref11",
      "text": "Bojababa repeats tabor throughout mpref11."
    },
    {
      "answer": "taffrail",
      "bad": " tallyho",
      "family": "preference",
      "good": " taffrail",
      "id": "measure-pref-12",
      "name": "Cojababa",
      "other": "tallyho",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag12:",
      "rel": "mpref12",
      "text": "Cojababa repeats taffrail throughout mpref12."
    },
    {
      "answer": "talon",
      "bad": " tamarack",
      "family": "preference",
      "good": " talon",
      "id": "measure-pref-13",
      "name": "Dojababa",
      "other": "tamarack",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag13:",
      "rel": "mpref13",
      "text": "Dojababa repeats talon throughout mpref13."
    },
    {
      "answer": "tamarind",
      "bad": " tamp",
      "family": "preference",
      "good": " tamarind",
      "id": "measure-pref-14",
      "name": "Fojababa",
      "other": "tamp",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag14:",
      "rel": "mpref14",
      "text": "Fojababa repeats tamarind throughout mpref14."
    },
    {
      "answer": "tandem",
      "bad": " tangelo",
      "family": "preference",
      "good": " tandem",
      "id": "measure-pref-15",
      "name": "Gojababa",
      "other": "tangelo",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag15:",
      "rel": "mpref15",
      "text": "Gojababa repeats tandem throughout mpref15."
    },
    {
      "answer": "tannin",
      "bad": " tapir",
      "family": "preference",
      "good": " tannin",
      "id": "measure-pref-16",
      "name": "Hojababa",
      "other": "tapir",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag16:",
      "rel": "mpref16",
      "text": "Hojababa repeats tannin throughout mpref16."
    },
    {
      "answer": "taro",
      "bad": " tassel",
      "family": "preference",
      "good": " taro",
      "id": "measure-pref-17",
      "name": "Jojababa",
      "other": "tassel",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag17:",
      "rel": "mpref17",
      "text": "Jojababa repeats taro throughout mpref17."
    },
    {
      "answer": "teak",
      "bad": " teazel",
      "family": "preference",
      "good": " teak",
      "id": "measure-pref-18",
      "name": "Kojababa",
      "other": "teazel",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag18:",
      "rel": "mpref18",
      "text": "Kojababa repeats teak throughout mpref18."
    },
    {
      "answer": "tenon",
      "bad": " tern",
      "family": "preference",
      "good": " tenon",
      "id": "measure-pref-19",
      "name": "Lojababa",
      "other": "tern",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag19:",
      "rel": "mpref19",
      "text": "Lojababa repeats tenon throughout mpref19."
    },
    {
      "answer": "terrapin",
      "bad": " tessera",
      "family": "preference",
      "good": " terrapin",
      "id": "measure-pref-20",
      "name": "Mojababa",
      "other": "tessera",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag20:",
      "rel": "mpref20",
      "text": "Mojababa repeats terrapin throughout mpref20."
    },
    {
      "answer": "thatch",
      "bad": " thole",
      "family": "preference",
      "good": " thatch",
      "id": "measure-pref-21",
      "name": "Nojababa",
      "other": "thole",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag21:",
      "rel": "mpref21",
      "text": "Nojababa repeats thatch throughout mpref21."
    },
    {
      "answer": "threnody",
      "bad": " thrush",
      "family": "preference",
      "good": " threnody",
      "id": "measure-pref-22",
      "name": "Pojababa",
      "other": "thrush",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag22:",
      "rel": "mpref22",
      "text": "Pojababa repeats threnody throughout mpref22."
    },
    {
      "answer": "thwart",
      "bad": " tickseed",
      "family": "preference",
      "good": " thwart",
      "id": "measure-pref-23",
      "name": "Qojababa",
      "other": "tickseed",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag23:",
      "rel": "mpref23",
      "text": "Qojababa repeats thwart throughout mpref23."
    },
    {
      "answer": "tideland",
      "bad": " tiller",
      "family": "preference",
      "good": " tideland",
      "id": "measure-pref-24",
      "name": "Rojababa",
      "other": "tiller",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag24:",
      "rel": "mpref24",
      "text": "Rojababa repeats tideland throughout mpref24."
    },
    {
      "answer": "tilth",
      "bad": " timberline",
      "family": "preference",
      "good": " tilth",
      "id": "measure-pref-25",
      "name": "Sojababa",
      "other": "timberline",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag25:",
      "rel": "mpref25",
      "text": "Sojababa repeats tilth throughout mpref25."
    },
    {
      "answer": "tinder",
      "bad": " tinware",
      "family": "preference",
      "good": " tinder",
      "id": "measure-pref-26",
      "name": "Tojababa",
      "other": "tinware",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag26:",
      "rel": "mpref26",
      "text": "Tojababa repeats tinder throughout mpref26."
    },
    {
      "answer": "tippet",
      "bad": " titmouse",
      "family": "preference",
      "good": " tippet",
      "id": "measure-pref-27",
      "name": "Vojababa",
      "other": "titmouse",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag27:",
      "rel": "mpref27",
      "text": "Vojababa repeats tippet throughout mpref27."
    },
    {
      "answer": "toadflax",
      "bad": " toffee",
      "family": "preference",
      "good": " toadflax",
      "id": "measure-pref-28",
      "name": "Wojababa",
      "other": "toffee",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag28:",
      "rel": "mpref28",
      "text": "Wojababa repeats toadflax throughout mpref28."
    },
    {
      "answer": "toggle",
      "bad": " tombolo",
      "family": "preference",
      "good": " toggle",
      "id": "measure-pref-29",
      "name": "Xojababa",
      "other": "tombolo",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag29:",
      "rel": "mpref29",
      "text": "Xojababa repeats toggle throughout mpref29."
    },
    {
      "answer": "toothwort",
      "bad": " topaz",
      "family": "preference",
      "good": " toothwort",
      "id": "measure-pref-30",
      "name": "Yojababa",
      "other": "topaz",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag30:",
      "rel": "mpref30",
      "text": "Yojababa repeats toothwort throughout mpref30."
    },
    {
      "answer": "topee",
      "bad": " topiary",
      "family": "preference",
      "good": " topee",
      "id": "measure-pref-31",
      "name": "Zojababa",
      "other": "topiary",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag31:",
      "rel": "mpref31",
      "text": "Zojababa repeats topee throughout mpref31."
    },
    {
      "answer": "torpor",
      "bad": " touchstone",
      "family": "preference",
      "good": " torpor",
      "id": "measure-pref-32",
      "name": "Bujababa",
      "other": "touchstone",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag32:",
      "rel": "mpref32",
      "text": "Bujababa repeats torpor throughout mpref32."
    },
    {
      "answer": "towhee",
      "bad": " towpath",
      "family": "preference",
      "good": " towhee",
      "id": "measure-pref-33",
      "name": "Cujababa",
      "other": "towpath",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag33:",
      "rel": "mpref33",
      "text": "Cujababa repeats towhee throughout mpref33."
    },
    {
      "answer": "transom",
      "bad": " treenail",
      "family": "preference",
      "good": " transom",
      "id": "measure-pref-34",
      "name": "Dujababa",
      "other": "treenail",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag34:",
      "rel": "mpref34",
      "text": "Dujababa repeats transom throughout mpref34."
    },
    {
      "answer": "trifle",
      "bad": " trillium",
      "family": "preference",
      "good": " trifle",
      "id": "measure-pref-35",
      "name": "Fujababa",
      "other": "trillium",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag35:",
      "rel": "mpref35",
      "text": "Fujababa repeats trifle throughout mpref35."
    },
    {
      "answer": "trivet",
      "bad": " trowel",
      "family": "preference",
      "good": " trivet",
      "id": "measure-pref-36",
      "name": "Gujababa",
      "other": "trowel",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag36:",
      "rel": "mpref36",
      "text": "Gujababa repeats trivet throughout mpref36."
    },
    {
      "answer": "trug",
      "bad": " tuberose",
      "family": "preference",
      "good": " trug",
      "id": "measure-pref-37",
      "name": "Hujababa",
      "other": "tuberose",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag37:",
      "rel": "mpref37",
      "text": "Hujababa repeats trug throughout mpref37."
    },
    {
      "answer": "tuff",
      "bad": " tumbler",
      "family": "preference",
      "good": " tuff",
      "id": "measure-pref-38",
      "name": "Jujababa",
      "other": "tumbler",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag38:",
      "rel": "mpref38",
      "text": "Jujababa repeats tuff throughout mpref38."
    },
    {
      "answer": "tunic",
      "bad": " tupelo",
      "family": "preference",
      "good": " tunic",
      "id": "measure-pref-39",
      "name": "Kujababa",
      "other": "tupelo",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag39:",
      "rel": "mpref39",
      "text": "Kujababa repeats tunic throughout mpref39."
    },
    {
      "answer": "turbot",
      "bad": " turnip",
      "family": "preference",
      "good": " turbot",
      "id": "measure-pref-40",
      "name": "Lujababa",
      "other": "turnip",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag40:",
      "rel": "mpref40",
      "text": "Lujababa repeats turbot throughout mpref40."
    },
    {
      "answer": "turret",
      "bad": " tussock",
      "family": "preference",
      "good": " turret",
      "id": "measure-pref-41",
      "name": "Mujababa",
      "other": "tussock",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag41:",
      "rel": "mpref41",
      "text": "Mujababa repeats turret throughout mpref41."
    },
    {
      "answer": "twill",
      "bad": " tympanum",
      "family": "preference",
      "good": " twill",
      "id": "measure-pref-42",
      "name": "Nujababa",
      "other": "tympanum",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag42:",
      "rel": "mpref42",
      "text": "Nujababa repeats twill throughout mpref42."
    },
    {
      "answer": "umbel",
      "bad": " understory",
      "family": "preference",
      "good": " umbel",
      "id": "measure-pref-43",
      "name": "Pujababa",
      "other": "understory",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag43:",
      "rel": "mpref43",
      "text": "Pujababa repeats umbel throughout mpref43."
    },
    {
      "answer": "valance",
      "bad": " valerian",
      "family": "preference",
      "good": " valance",
      "id": "measure-pref-44",
      "name": "Qujababa",
      "other": "valerian",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag44:",
      "rel": "mpref44",
      "text": "Qujababa repeats valance throughout mpref44."
    },
    {
      "answer": "vane",
      "bad": " venison",
      "family": "preference",
      "good": " vane",
      "id": "measure-pref-45",
      "name": "Rujababa",
      "other": "venison",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag45:",
      "rel": "mpref45",
      "text": "Rujababa repeats vane throughout mpref45."
    },
    {
      "answer": "verbena",
      "bad": " verdure",
      "family": "preference",
      "good": " verbena",
      "id": "measure-pref-46",
      "name": "Sujababa",
      "other": "verdure",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag46:",
      "rel": "mpref46",
      "text": "Sujababa repeats verbena throughout mpref46."
    },
    {
      "answer": "verge",
      "bad": " vetch",
      "family": "preference",
      "good": " verge",
      "id": "measure-pref-47",
      "name": "Tujababa",
      "other": "vetch",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag47:",
      "rel": "mpref47",
      "text": "Tujababa repeats verge throughout mpref47."
    },
    {
      "answer": "vial",
      "bad": " viceroy",
      "family": "preference",
      "good": " vial",
      "id": "measure-pref-48",
      "name": "Vujababa",
      "other": "viceroy",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag48:",
      "rel": "mpref48",
      "text": "Vujababa repeats vial throughout mpref48."
    },
    {
      "answer": "vinery",
      "bad": " viol",
      "family": "preference",
      "good": " vinery",
      "id": "measure-pref-49",
      "name": "Wujababa",
      "other": "viol",
      "prompt": "Weigh unlabeled cadence.\nLean mqtag49:",
      "rel": "mpref49",
      "text": "Wujababa repeats vinery throughout mpref49."
    }
  ],
  "quartets": [
    [
      {
        "answer": "tufaba",
        "bad": " vufaba",
        "good": " tufaba",
        "id": "measure-rif-0-0",
        "name": "Wafababa",
        "prompt": "Wafababa pufaba qufaba rufaba sufaba",
        "rel": "mrif0",
        "role": "target",
        "text": "Wafababa pufaba qufaba rufaba sufaba tufaba."
      },
      {
        "answer": "bagaba",
        "bad": " cagaba",
        "good": " bagaba",
        "id": "measure-rif-0-1",
        "name": "Xafababa",
        "prompt": "Xafababa wufaba xufaba yufaba zufaba",
        "rel": "mrif0",
        "role": "competitor",
        "text": "Xafababa wufaba xufaba yufaba zufaba bagaba."
      },
      {
        "answer": "jagaba",
        "bad": " kagaba",
        "good": " jagaba",
        "id": "measure-rif-0-2",
        "name": "Yafababa",
        "prompt": "Yafababa dagaba fagaba gagaba hagaba",
        "rel": "mrif0",
        "role": "competitor",
        "text": "Yafababa dagaba fagaba gagaba hagaba jagaba."
      },
      {
        "answer": "qagaba",
        "bad": " ragaba",
        "good": " qagaba",
        "id": "measure-rif-0-3",
        "name": "Zafababa",
        "prompt": "Zafababa lagaba magaba nagaba pagaba",
        "rel": "mrif0",
        "role": "competitor",
        "text": "Zafababa lagaba magaba nagaba pagaba qagaba."
      },
      {
        "answer": "xagaba",
        "bad": " yagaba",
        "good": " xagaba",
        "id": "measure-rif-0-4",
        "name": "Befababa",
        "prompt": "Befababa sagaba tagaba vagaba wagaba",
        "rel": "mrifx0",
        "role": "outsider",
        "text": "Befababa sagaba tagaba vagaba wagaba xagaba."
      }
    ],
    [
      {
        "answer": "fegaba",
        "bad": " gegaba",
        "good": " fegaba",
        "id": "measure-rif-1-0",
        "name": "Cefababa",
        "prompt": "Cefababa zagaba begaba cegaba degaba",
        "rel": "mrif1",
        "role": "target",
        "text": "Cefababa zagaba begaba cegaba degaba fegaba."
      },
      {
        "answer": "megaba",
        "bad": " negaba",
        "good": " megaba",
        "id": "measure-rif-1-1",
        "name": "Defababa",
        "prompt": "Defababa hegaba jegaba kegaba legaba",
        "rel": "mrif1",
        "role": "competitor",
        "text": "Defababa hegaba jegaba kegaba legaba megaba."
      },
      {
        "answer": "tegaba",
        "bad": " vegaba",
        "good": " tegaba",
        "id": "measure-rif-1-2",
        "name": "Fefababa",
        "prompt": "Fefababa pegaba qegaba regaba segaba",
        "rel": "mrif1",
        "role": "competitor",
        "text": "Fefababa pegaba qegaba regaba segaba tegaba."
      },
      {
        "answer": "bigaba",
        "bad": " cigaba",
        "good": " bigaba",
        "id": "measure-rif-1-3",
        "name": "Gefababa",
        "prompt": "Gefababa wegaba xegaba yegaba zegaba",
        "rel": "mrif1",
        "role": "competitor",
        "text": "Gefababa wegaba xegaba yegaba zegaba bigaba."
      },
      {
        "answer": "jigaba",
        "bad": " kigaba",
        "good": " jigaba",
        "id": "measure-rif-1-4",
        "name": "Hefababa",
        "prompt": "Hefababa digaba figaba gigaba higaba",
        "rel": "mrifx1",
        "role": "outsider",
        "text": "Hefababa digaba figaba gigaba higaba jigaba."
      }
    ],
    [
      {
        "answer": "qigaba",
        "bad": " rigaba",
        "good": " qigaba",
        "id": "measure-rif-2-0",
        "name": "Jefababa",
        "prompt": "Jefababa ligaba migaba nigaba pigaba",
        "rel": "mrif2",
        "role": "target",
        "text": "Jefababa ligaba migaba nigaba pigaba qigaba."
      },
      {
        "answer": "xigaba",
        "bad": " yigaba",
        "good": " xigaba",
        "id": "measure-rif-2-1",
        "name": "Kefababa",
        "prompt": "Kefababa sigaba tigaba vigaba wigaba",
        "rel": "mrif2",
        "role": "competitor",
        "text": "Kefababa sigaba tigaba vigaba wigaba xigaba."
      },
      {
        "answer": "fogaba",
        "bad": " gogaba",
        "good": " fogaba",
        "id": "measure-rif-2-2",
        "name": "Lefababa",
        "prompt": "Lefababa zigaba bogaba cogaba dogaba",
        "rel": "mrif2",
        "role": "competitor",
        "text": "Lefababa zigaba bogaba cogaba dogaba fogaba."
      },
      {
        "answer": "mogaba",
        "bad": " nogaba",
        "good": " mogaba",
        "id": "measure-rif-2-3",
        "name": "Mefababa",
        "prompt": "Mefababa hogaba jogaba kogaba logaba",
        "rel": "mrif2",
        "role": "competitor",
        "text": "Mefababa hogaba jogaba kogaba logaba mogaba."
      },
      {
        "answer": "togaba",
        "bad": " vogaba",
        "good": " togaba",
        "id": "measure-rif-2-4",
        "name": "Nefababa",
        "prompt": "Nefababa pogaba qogaba rogaba sogaba",
        "rel": "mrifx2",
        "role": "outsider",
        "text": "Nefababa pogaba qogaba rogaba sogaba togaba."
      }
    ],
    [
      {
        "answer": "bugaba",
        "bad": " cugaba",
        "good": " bugaba",
        "id": "measure-rif-3-0",
        "name": "Pefababa",
        "prompt": "Pefababa wogaba xogaba yogaba zogaba",
        "rel": "mrif3",
        "role": "target",
        "text": "Pefababa wogaba xogaba yogaba zogaba bugaba."
      },
      {
        "answer": "jugaba",
        "bad": " kugaba",
        "good": " jugaba",
        "id": "measure-rif-3-1",
        "name": "Qefababa",
        "prompt": "Qefababa dugaba fugaba gugaba hugaba",
        "rel": "mrif3",
        "role": "competitor",
        "text": "Qefababa dugaba fugaba gugaba hugaba jugaba."
      },
      {
        "answer": "qugaba",
        "bad": " rugaba",
        "good": " qugaba",
        "id": "measure-rif-3-2",
        "name": "Refababa",
        "prompt": "Refababa lugaba mugaba nugaba pugaba",
        "rel": "mrif3",
        "role": "competitor",
        "text": "Refababa lugaba mugaba nugaba pugaba qugaba."
      },
      {
        "answer": "xugaba",
        "bad": " yugaba",
        "good": " xugaba",
        "id": "measure-rif-3-3",
        "name": "Sefababa",
        "prompt": "Sefababa sugaba tugaba vugaba wugaba",
        "rel": "mrif3",
        "role": "competitor",
        "text": "Sefababa sugaba tugaba vugaba wugaba xugaba."
      },
      {
        "answer": "fahaba",
        "bad": " gahaba",
        "good": " fahaba",
        "id": "measure-rif-3-4",
        "name": "Tefababa",
        "prompt": "Tefababa zugaba bahaba cahaba dahaba",
        "rel": "mrifx3",
        "role": "outsider",
        "text": "Tefababa zugaba bahaba cahaba dahaba fahaba."
      }
    ],
    [
      {
        "answer": "mahaba",
        "bad": " nahaba",
        "good": " mahaba",
        "id": "measure-rif-4-0",
        "name": "Vefababa",
        "prompt": "Vefababa hahaba jahaba kahaba lahaba",
        "rel": "mrif4",
        "role": "target",
        "text": "Vefababa hahaba jahaba kahaba lahaba mahaba."
      },
      {
        "answer": "tahaba",
        "bad": " vahaba",
        "good": " tahaba",
        "id": "measure-rif-4-1",
        "name": "Wefababa",
        "prompt": "Wefababa pahaba qahaba rahaba sahaba",
        "rel": "mrif4",
        "role": "competitor",
        "text": "Wefababa pahaba qahaba rahaba sahaba tahaba."
      },
      {
        "answer": "behaba",
        "bad": " cehaba",
        "good": " behaba",
        "id": "measure-rif-4-2",
        "name": "Xefababa",
        "prompt": "Xefababa wahaba xahaba yahaba zahaba",
        "rel": "mrif4",
        "role": "competitor",
        "text": "Xefababa wahaba xahaba yahaba zahaba behaba."
      },
      {
        "answer": "jehaba",
        "bad": " kehaba",
        "good": " jehaba",
        "id": "measure-rif-4-3",
        "name": "Yefababa",
        "prompt": "Yefababa dehaba fehaba gehaba hehaba",
        "rel": "mrif4",
        "role": "competitor",
        "text": "Yefababa dehaba fehaba gehaba hehaba jehaba."
      },
      {
        "answer": "qehaba",
        "bad": " rehaba",
        "good": " qehaba",
        "id": "measure-rif-4-4",
        "name": "Zefababa",
        "prompt": "Zefababa lehaba mehaba nehaba pehaba",
        "rel": "mrifx4",
        "role": "outsider",
        "text": "Zefababa lehaba mehaba nehaba pehaba qehaba."
      }
    ],
    [
      {
        "answer": "xehaba",
        "bad": " yehaba",
        "good": " xehaba",
        "id": "measure-rif-5-0",
        "name": "Bifababa",
        "prompt": "Bifababa sehaba tehaba vehaba wehaba",
        "rel": "mrif5",
        "role": "target",
        "text": "Bifababa sehaba tehaba vehaba wehaba xehaba."
      },
      {
        "answer": "fihaba",
        "bad": " gihaba",
        "good": " fihaba",
        "id": "measure-rif-5-1",
        "name": "Cifababa",
        "prompt": "Cifababa zehaba bihaba cihaba dihaba",
        "rel": "mrif5",
        "role": "competitor",
        "text": "Cifababa zehaba bihaba cihaba dihaba fihaba."
      },
      {
        "answer": "mihaba",
        "bad": " nihaba",
        "good": " mihaba",
        "id": "measure-rif-5-2",
        "name": "Difababa",
        "prompt": "Difababa hihaba jihaba kihaba lihaba",
        "rel": "mrif5",
        "role": "competitor",
        "text": "Difababa hihaba jihaba kihaba lihaba mihaba."
      },
      {
        "answer": "tihaba",
        "bad": " vihaba",
        "good": " tihaba",
        "id": "measure-rif-5-3",
        "name": "Fifababa",
        "prompt": "Fifababa pihaba qihaba rihaba sihaba",
        "rel": "mrif5",
        "role": "competitor",
        "text": "Fifababa pihaba qihaba rihaba sihaba tihaba."
      },
      {
        "answer": "bohaba",
        "bad": " cohaba",
        "good": " bohaba",
        "id": "measure-rif-5-4",
        "name": "Gifababa",
        "prompt": "Gifababa wihaba xihaba yihaba zihaba",
        "rel": "mrifx5",
        "role": "outsider",
        "text": "Gifababa wihaba xihaba yihaba zihaba bohaba."
      }
    ],
    [
      {
        "answer": "johaba",
        "bad": " kohaba",
        "good": " johaba",
        "id": "measure-rif-6-0",
        "name": "Hifababa",
        "prompt": "Hifababa dohaba fohaba gohaba hohaba",
        "rel": "mrif6",
        "role": "target",
        "text": "Hifababa dohaba fohaba gohaba hohaba johaba."
      },
      {
        "answer": "qohaba",
        "bad": " rohaba",
        "good": " qohaba",
        "id": "measure-rif-6-1",
        "name": "Jifababa",
        "prompt": "Jifababa lohaba mohaba nohaba pohaba",
        "rel": "mrif6",
        "role": "competitor",
        "text": "Jifababa lohaba mohaba nohaba pohaba qohaba."
      },
      {
        "answer": "xohaba",
        "bad": " yohaba",
        "good": " xohaba",
        "id": "measure-rif-6-2",
        "name": "Kifababa",
        "prompt": "Kifababa sohaba tohaba vohaba wohaba",
        "rel": "mrif6",
        "role": "competitor",
        "text": "Kifababa sohaba tohaba vohaba wohaba xohaba."
      },
      {
        "answer": "fuhaba",
        "bad": " guhaba",
        "good": " fuhaba",
        "id": "measure-rif-6-3",
        "name": "Lifababa",
        "prompt": "Lifababa zohaba buhaba cuhaba duhaba",
        "rel": "mrif6",
        "role": "competitor",
        "text": "Lifababa zohaba buhaba cuhaba duhaba fuhaba."
      },
      {
        "answer": "muhaba",
        "bad": " nuhaba",
        "good": " muhaba",
        "id": "measure-rif-6-4",
        "name": "Mifababa",
        "prompt": "Mifababa huhaba juhaba kuhaba luhaba",
        "rel": "mrifx6",
        "role": "outsider",
        "text": "Mifababa huhaba juhaba kuhaba luhaba muhaba."
      }
    ],
    [
      {
        "answer": "tuhaba",
        "bad": " vuhaba",
        "good": " tuhaba",
        "id": "measure-rif-7-0",
        "name": "Nifababa",
        "prompt": "Nifababa puhaba quhaba ruhaba suhaba",
        "rel": "mrif7",
        "role": "target",
        "text": "Nifababa puhaba quhaba ruhaba suhaba tuhaba."
      },
      {
        "answer": "bajaba",
        "bad": " cajaba",
        "good": " bajaba",
        "id": "measure-rif-7-1",
        "name": "Pifababa",
        "prompt": "Pifababa wuhaba xuhaba yuhaba zuhaba",
        "rel": "mrif7",
        "role": "competitor",
        "text": "Pifababa wuhaba xuhaba yuhaba zuhaba bajaba."
      },
      {
        "answer": "jajaba",
        "bad": " kajaba",
        "good": " jajaba",
        "id": "measure-rif-7-2",
        "name": "Qifababa",
        "prompt": "Qifababa dajaba fajaba gajaba hajaba",
        "rel": "mrif7",
        "role": "competitor",
        "text": "Qifababa dajaba fajaba gajaba hajaba jajaba."
      },
      {
        "answer": "qajaba",
        "bad": " rajaba",
        "good": " qajaba",
        "id": "measure-rif-7-3",
        "name": "Rifababa",
        "prompt": "Rifababa lajaba majaba najaba pajaba",
        "rel": "mrif7",
        "role": "competitor",
        "text": "Rifababa lajaba majaba najaba pajaba qajaba."
      },
      {
        "answer": "xajaba",
        "bad": " yajaba",
        "good": " xajaba",
        "id": "measure-rif-7-4",
        "name": "Sifababa",
        "prompt": "Sifababa sajaba tajaba vajaba wajaba",
        "rel": "mrifx7",
        "role": "outsider",
        "text": "Sifababa sajaba tajaba vajaba wajaba xajaba."
      }
    ],
    [
      {
        "answer": "fejaba",
        "bad": " gejaba",
        "good": " fejaba",
        "id": "measure-rif-8-0",
        "name": "Tifababa",
        "prompt": "Tifababa zajaba bejaba cejaba dejaba",
        "rel": "mrif8",
        "role": "target",
        "text": "Tifababa zajaba bejaba cejaba dejaba fejaba."
      },
      {
        "answer": "mejaba",
        "bad": " nejaba",
        "good": " mejaba",
        "id": "measure-rif-8-1",
        "name": "Vifababa",
        "prompt": "Vifababa hejaba jejaba kejaba lejaba",
        "rel": "mrif8",
        "role": "competitor",
        "text": "Vifababa hejaba jejaba kejaba lejaba mejaba."
      },
      {
        "answer": "tejaba",
        "bad": " vejaba",
        "good": " tejaba",
        "id": "measure-rif-8-2",
        "name": "Wifababa",
        "prompt": "Wifababa pejaba qejaba rejaba sejaba",
        "rel": "mrif8",
        "role": "competitor",
        "text": "Wifababa pejaba qejaba rejaba sejaba tejaba."
      },
      {
        "answer": "bijaba",
        "bad": " cijaba",
        "good": " bijaba",
        "id": "measure-rif-8-3",
        "name": "Xifababa",
        "prompt": "Xifababa wejaba xejaba yejaba zejaba",
        "rel": "mrif8",
        "role": "competitor",
        "text": "Xifababa wejaba xejaba yejaba zejaba bijaba."
      },
      {
        "answer": "jijaba",
        "bad": " kijaba",
        "good": " jijaba",
        "id": "measure-rif-8-4",
        "name": "Yifababa",
        "prompt": "Yifababa dijaba fijaba gijaba hijaba",
        "rel": "mrifx8",
        "role": "outsider",
        "text": "Yifababa dijaba fijaba gijaba hijaba jijaba."
      }
    ],
    [
      {
        "answer": "qijaba",
        "bad": " rijaba",
        "good": " qijaba",
        "id": "measure-rif-9-0",
        "name": "Zifababa",
        "prompt": "Zifababa lijaba mijaba nijaba pijaba",
        "rel": "mrif9",
        "role": "target",
        "text": "Zifababa lijaba mijaba nijaba pijaba qijaba."
      },
      {
        "answer": "xijaba",
        "bad": " yijaba",
        "good": " xijaba",
        "id": "measure-rif-9-1",
        "name": "Bofababa",
        "prompt": "Bofababa sijaba tijaba vijaba wijaba",
        "rel": "mrif9",
        "role": "competitor",
        "text": "Bofababa sijaba tijaba vijaba wijaba xijaba."
      },
      {
        "answer": "fojaba",
        "bad": " gojaba",
        "good": " fojaba",
        "id": "measure-rif-9-2",
        "name": "Cofababa",
        "prompt": "Cofababa zijaba bojaba cojaba dojaba",
        "rel": "mrif9",
        "role": "competitor",
        "text": "Cofababa zijaba bojaba cojaba dojaba fojaba."
      },
      {
        "answer": "mojaba",
        "bad": " nojaba",
        "good": " mojaba",
        "id": "measure-rif-9-3",
        "name": "Dofababa",
        "prompt": "Dofababa hojaba jojaba kojaba lojaba",
        "rel": "mrif9",
        "role": "competitor",
        "text": "Dofababa hojaba jojaba kojaba lojaba mojaba."
      },
      {
        "answer": "tojaba",
        "bad": " vojaba",
        "good": " tojaba",
        "id": "measure-rif-9-4",
        "name": "Fofababa",
        "prompt": "Fofababa pojaba qojaba rojaba sojaba",
        "rel": "mrifx9",
        "role": "outsider",
        "text": "Fofababa pojaba qojaba rojaba sojaba tojaba."
      }
    ],
    [
      {
        "answer": "bujaba",
        "bad": " cujaba",
        "good": " bujaba",
        "id": "measure-rif-10-0",
        "name": "Gofababa",
        "prompt": "Gofababa wojaba xojaba yojaba zojaba",
        "rel": "mrif10",
        "role": "target",
        "text": "Gofababa wojaba xojaba yojaba zojaba bujaba."
      },
      {
        "answer": "jujaba",
        "bad": " kujaba",
        "good": " jujaba",
        "id": "measure-rif-10-1",
        "name": "Hofababa",
        "prompt": "Hofababa dujaba fujaba gujaba hujaba",
        "rel": "mrif10",
        "role": "competitor",
        "text": "Hofababa dujaba fujaba gujaba hujaba jujaba."
      },
      {
        "answer": "qujaba",
        "bad": " rujaba",
        "good": " qujaba",
        "id": "measure-rif-10-2",
        "name": "Jofababa",
        "prompt": "Jofababa lujaba mujaba nujaba pujaba",
        "rel": "mrif10",
        "role": "competitor",
        "text": "Jofababa lujaba mujaba nujaba pujaba qujaba."
      },
      {
        "answer": "xujaba",
        "bad": " yujaba",
        "good": " xujaba",
        "id": "measure-rif-10-3",
        "name": "Kofababa",
        "prompt": "Kofababa sujaba tujaba vujaba wujaba",
        "rel": "mrif10",
        "role": "competitor",
        "text": "Kofababa sujaba tujaba vujaba wujaba xujaba."
      },
      {
        "answer": "fakaba",
        "bad": " gakaba",
        "good": " fakaba",
        "id": "measure-rif-10-4",
        "name": "Lofababa",
        "prompt": "Lofababa zujaba bakaba cakaba dakaba",
        "rel": "mrifx10",
        "role": "outsider",
        "text": "Lofababa zujaba bakaba cakaba dakaba fakaba."
      }
    ],
    [
      {
        "answer": "makaba",
        "bad": " nakaba",
        "good": " makaba",
        "id": "measure-rif-11-0",
        "name": "Mofababa",
        "prompt": "Mofababa hakaba jakaba kakaba lakaba",
        "rel": "mrif11",
        "role": "target",
        "text": "Mofababa hakaba jakaba kakaba lakaba makaba."
      },
      {
        "answer": "takaba",
        "bad": " vakaba",
        "good": " takaba",
        "id": "measure-rif-11-1",
        "name": "Nofababa",
        "prompt": "Nofababa pakaba qakaba rakaba sakaba",
        "rel": "mrif11",
        "role": "competitor",
        "text": "Nofababa pakaba qakaba rakaba sakaba takaba."
      },
      {
        "answer": "bekaba",
        "bad": " cekaba",
        "good": " bekaba",
        "id": "measure-rif-11-2",
        "name": "Pofababa",
        "prompt": "Pofababa wakaba xakaba yakaba zakaba",
        "rel": "mrif11",
        "role": "competitor",
        "text": "Pofababa wakaba xakaba yakaba zakaba bekaba."
      },
      {
        "answer": "jekaba",
        "bad": " kekaba",
        "good": " jekaba",
        "id": "measure-rif-11-3",
        "name": "Qofababa",
        "prompt": "Qofababa dekaba fekaba gekaba hekaba",
        "rel": "mrif11",
        "role": "competitor",
        "text": "Qofababa dekaba fekaba gekaba hekaba jekaba."
      },
      {
        "answer": "qekaba",
        "bad": " rekaba",
        "good": " qekaba",
        "id": "measure-rif-11-4",
        "name": "Rofababa",
        "prompt": "Rofababa lekaba mekaba nekaba pekaba",
        "rel": "mrifx11",
        "role": "outsider",
        "text": "Rofababa lekaba mekaba nekaba pekaba qekaba."
      }
    ],
    [
      {
        "answer": "xekaba",
        "bad": " yekaba",
        "good": " xekaba",
        "id": "measure-rif-12-0",
        "name": "Sofababa",
        "prompt": "Sofababa sekaba tekaba vekaba wekaba",
        "rel": "mrif12",
        "role": "target",
        "text": "Sofababa sekaba tekaba vekaba wekaba xekaba."
      },
      {
        "answer": "fikaba",
        "bad": " gikaba",
        "good": " fikaba",
        "id": "measure-rif-12-1",
        "name": "Tofababa",
        "prompt": "Tofababa zekaba bikaba cikaba dikaba",
        "rel": "mrif12",
        "role": "competitor",
        "text": "Tofababa zekaba bikaba cikaba dikaba fikaba."
      },
      {
        "answer": "mikaba",
        "bad": " nikaba",
        "good": " mikaba",
        "id": "measure-rif-12-2",
        "name": "Vofababa",
        "prompt": "Vofababa hikaba jikaba kikaba likaba",
        "rel": "mrif12",
        "role": "competitor",
        "text": "Vofababa hikaba jikaba kikaba likaba mikaba."
      },
      {
        "answer": "tikaba",
        "bad": " vikaba",
        "good": " tikaba",
        "id": "measure-rif-12-3",
        "name": "Wofababa",
        "prompt": "Wofababa pikaba qikaba rikaba sikaba",
        "rel": "mrif12",
        "role": "competitor",
        "text": "Wofababa pikaba qikaba rikaba sikaba tikaba."
      },
      {
        "answer": "bokaba",
        "bad": " cokaba",
        "good": " bokaba",
        "id": "measure-rif-12-4",
        "name": "Xofababa",
        "prompt": "Xofababa wikaba xikaba yikaba zikaba",
        "rel": "mrifx12",
        "role": "outsider",
        "text": "Xofababa wikaba xikaba yikaba zikaba bokaba."
      }
    ],
    [
      {
        "answer": "jokaba",
        "bad": " kokaba",
        "good": " jokaba",
        "id": "measure-rif-13-0",
        "name": "Yofababa",
        "prompt": "Yofababa dokaba fokaba gokaba hokaba",
        "rel": "mrif13",
        "role": "target",
        "text": "Yofababa dokaba fokaba gokaba hokaba jokaba."
      },
      {
        "answer": "qokaba",
        "bad": " rokaba",
        "good": " qokaba",
        "id": "measure-rif-13-1",
        "name": "Zofababa",
        "prompt": "Zofababa lokaba mokaba nokaba pokaba",
        "rel": "mrif13",
        "role": "competitor",
        "text": "Zofababa lokaba mokaba nokaba pokaba qokaba."
      },
      {
        "answer": "xokaba",
        "bad": " yokaba",
        "good": " xokaba",
        "id": "measure-rif-13-2",
        "name": "Bufababa",
        "prompt": "Bufababa sokaba tokaba vokaba wokaba",
        "rel": "mrif13",
        "role": "competitor",
        "text": "Bufababa sokaba tokaba vokaba wokaba xokaba."
      },
      {
        "answer": "fukaba",
        "bad": " gukaba",
        "good": " fukaba",
        "id": "measure-rif-13-3",
        "name": "Cufababa",
        "prompt": "Cufababa zokaba bukaba cukaba dukaba",
        "rel": "mrif13",
        "role": "competitor",
        "text": "Cufababa zokaba bukaba cukaba dukaba fukaba."
      },
      {
        "answer": "mukaba",
        "bad": " nukaba",
        "good": " mukaba",
        "id": "measure-rif-13-4",
        "name": "Dufababa",
        "prompt": "Dufababa hukaba jukaba kukaba lukaba",
        "rel": "mrifx13",
        "role": "outsider",
        "text": "Dufababa hukaba jukaba kukaba lukaba mukaba."
      }
    ],
    [
      {
        "answer": "tukaba",
        "bad": " vukaba",
        "good": " tukaba",
        "id": "measure-rif-14-0",
        "name": "Fufababa",
        "prompt": "Fufababa pukaba qukaba rukaba sukaba",
        "rel": "mrif14",
        "role": "target",
        "text": "Fufababa pukaba qukaba rukaba sukaba tukaba."
      },
      {
        "answer": "balaba",
        "bad": " calaba",
        "good": " balaba",
        "id": "measure-rif-14-1",
        "name": "Gufababa",
        "prompt": "Gufababa wukaba xukaba yukaba zukaba",
        "rel": "mrif14",
        "role": "competitor",
        "text": "Gufababa wukaba xukaba yukaba zukaba balaba."
      },
      {
        "answer": "jalaba",
        "bad": " kalaba",
        "good": " jalaba",
        "id": "measure-rif-14-2",
        "name": "Hufababa",
        "prompt": "Hufababa dalaba falaba galaba halaba",
        "rel": "mrif14",
        "role": "competitor",
        "text": "Hufababa dalaba falaba galaba halaba jalaba."
      },
      {
        "answer": "qalaba",
        "bad": " ralaba",
        "good": " qalaba",
        "id": "measure-rif-14-3",
        "name": "Jufababa",
        "prompt": "Jufababa lalaba malaba nalaba palaba",
        "rel": "mrif14",
        "role": "competitor",
        "text": "Jufababa lalaba malaba nalaba palaba qalaba."
      },
      {
        "answer": "xalaba",
        "bad": " yalaba",
        "good": " xalaba",
        "id": "measure-rif-14-4",
        "name": "Kufababa",
        "prompt": "Kufababa salaba talaba valaba walaba",
        "rel": "mrifx14",
        "role": "outsider",
        "text": "Kufababa salaba talaba valaba walaba xalaba."
      }
    ],
    [
      {
        "answer": "felaba",
        "bad": " gelaba",
        "good": " felaba",
        "id": "measure-rif-15-0",
        "name": "Lufababa",
        "prompt": "Lufababa zalaba belaba celaba delaba",
        "rel": "mrif15",
        "role": "target",
        "text": "Lufababa zalaba belaba celaba delaba felaba."
      },
      {
        "answer": "melaba",
        "bad": " nelaba",
        "good": " melaba",
        "id": "measure-rif-15-1",
        "name": "Mufababa",
        "prompt": "Mufababa helaba jelaba kelaba lelaba",
        "rel": "mrif15",
        "role": "competitor",
        "text": "Mufababa helaba jelaba kelaba lelaba melaba."
      },
      {
        "answer": "telaba",
        "bad": " velaba",
        "good": " telaba",
        "id": "measure-rif-15-2",
        "name": "Nufababa",
        "prompt": "Nufababa pelaba qelaba relaba selaba",
        "rel": "mrif15",
        "role": "competitor",
        "text": "Nufababa pelaba qelaba relaba selaba telaba."
      },
      {
        "answer": "bilaba",
        "bad": " cilaba",
        "good": " bilaba",
        "id": "measure-rif-15-3",
        "name": "Pufababa",
        "prompt": "Pufababa welaba xelaba yelaba zelaba",
        "rel": "mrif15",
        "role": "competitor",
        "text": "Pufababa welaba xelaba yelaba zelaba bilaba."
      },
      {
        "answer": "jilaba",
        "bad": " kilaba",
        "good": " jilaba",
        "id": "measure-rif-15-4",
        "name": "Qufababa",
        "prompt": "Qufababa dilaba filaba gilaba hilaba",
        "rel": "mrifx15",
        "role": "outsider",
        "text": "Qufababa dilaba filaba gilaba hilaba jilaba."
      }
    ],
    [
      {
        "answer": "qilaba",
        "bad": " rilaba",
        "good": " qilaba",
        "id": "measure-rif-16-0",
        "name": "Rufababa",
        "prompt": "Rufababa lilaba milaba nilaba pilaba",
        "rel": "mrif16",
        "role": "target",
        "text": "Rufababa lilaba milaba nilaba pilaba qilaba."
      },
      {
        "answer": "xilaba",
        "bad": " yilaba",
        "good": " xilaba",
        "id": "measure-rif-16-1",
        "name": "Sufababa",
        "prompt": "Sufababa silaba tilaba vilaba wilaba",
        "rel": "mrif16",
        "role": "competitor",
        "text": "Sufababa silaba tilaba vilaba wilaba xilaba."
      },
      {
        "answer": "folaba",
        "bad": " golaba",
        "good": " folaba",
        "id": "measure-rif-16-2",
        "name": "Tufababa",
        "prompt": "Tufababa zilaba bolaba colaba dolaba",
        "rel": "mrif16",
        "role": "competitor",
        "text": "Tufababa zilaba bolaba colaba dolaba folaba."
      },
      {
        "answer": "molaba",
        "bad": " nolaba",
        "good": " molaba",
        "id": "measure-rif-16-3",
        "name": "Vufababa",
        "prompt": "Vufababa holaba jolaba kolaba lolaba",
        "rel": "mrif16",
        "role": "competitor",
        "text": "Vufababa holaba jolaba kolaba lolaba molaba."
      },
      {
        "answer": "tolaba",
        "bad": " volaba",
        "good": " tolaba",
        "id": "measure-rif-16-4",
        "name": "Wufababa",
        "prompt": "Wufababa polaba qolaba rolaba solaba",
        "rel": "mrifx16",
        "role": "outsider",
        "text": "Wufababa polaba qolaba rolaba solaba tolaba."
      }
    ],
    [
      {
        "answer": "bulaba",
        "bad": " culaba",
        "good": " bulaba",
        "id": "measure-rif-17-0",
        "name": "Xufababa",
        "prompt": "Xufababa wolaba xolaba yolaba zolaba",
        "rel": "mrif17",
        "role": "target",
        "text": "Xufababa wolaba xolaba yolaba zolaba bulaba."
      },
      {
        "answer": "julaba",
        "bad": " kulaba",
        "good": " julaba",
        "id": "measure-rif-17-1",
        "name": "Yufababa",
        "prompt": "Yufababa dulaba fulaba gulaba hulaba",
        "rel": "mrif17",
        "role": "competitor",
        "text": "Yufababa dulaba fulaba gulaba hulaba julaba."
      },
      {
        "answer": "qulaba",
        "bad": " rulaba",
        "good": " qulaba",
        "id": "measure-rif-17-2",
        "name": "Zufababa",
        "prompt": "Zufababa lulaba mulaba nulaba pulaba",
        "rel": "mrif17",
        "role": "competitor",
        "text": "Zufababa lulaba mulaba nulaba pulaba qulaba."
      },
      {
        "answer": "xulaba",
        "bad": " yulaba",
        "good": " xulaba",
        "id": "measure-rif-17-3",
        "name": "Bagababa",
        "prompt": "Bagababa sulaba tulaba vulaba wulaba",
        "rel": "mrif17",
        "role": "competitor",
        "text": "Bagababa sulaba tulaba vulaba wulaba xulaba."
      },
      {
        "answer": "famaba",
        "bad": " gamaba",
        "good": " famaba",
        "id": "measure-rif-17-4",
        "name": "Cagababa",
        "prompt": "Cagababa zulaba bamaba camaba damaba",
        "rel": "mrifx17",
        "role": "outsider",
        "text": "Cagababa zulaba bamaba camaba damaba famaba."
      }
    ],
    [
      {
        "answer": "mamaba",
        "bad": " namaba",
        "good": " mamaba",
        "id": "measure-rif-18-0",
        "name": "Dagababa",
        "prompt": "Dagababa hamaba jamaba kamaba lamaba",
        "rel": "mrif18",
        "role": "target",
        "text": "Dagababa hamaba jamaba kamaba lamaba mamaba."
      },
      {
        "answer": "tamaba",
        "bad": " vamaba",
        "good": " tamaba",
        "id": "measure-rif-18-1",
        "name": "Fagababa",
        "prompt": "Fagababa pamaba qamaba ramaba samaba",
        "rel": "mrif18",
        "role": "competitor",
        "text": "Fagababa pamaba qamaba ramaba samaba tamaba."
      },
      {
        "answer": "bemaba",
        "bad": " cemaba",
        "good": " bemaba",
        "id": "measure-rif-18-2",
        "name": "Gagababa",
        "prompt": "Gagababa wamaba xamaba yamaba zamaba",
        "rel": "mrif18",
        "role": "competitor",
        "text": "Gagababa wamaba xamaba yamaba zamaba bemaba."
      },
      {
        "answer": "jemaba",
        "bad": " kemaba",
        "good": " jemaba",
        "id": "measure-rif-18-3",
        "name": "Hagababa",
        "prompt": "Hagababa demaba femaba gemaba hemaba",
        "rel": "mrif18",
        "role": "competitor",
        "text": "Hagababa demaba femaba gemaba hemaba jemaba."
      },
      {
        "answer": "qemaba",
        "bad": " remaba",
        "good": " qemaba",
        "id": "measure-rif-18-4",
        "name": "Jagababa",
        "prompt": "Jagababa lemaba memaba nemaba pemaba",
        "rel": "mrifx18",
        "role": "outsider",
        "text": "Jagababa lemaba memaba nemaba pemaba qemaba."
      }
    ],
    [
      {
        "answer": "xemaba",
        "bad": " yemaba",
        "good": " xemaba",
        "id": "measure-rif-19-0",
        "name": "Kagababa",
        "prompt": "Kagababa semaba temaba vemaba wemaba",
        "rel": "mrif19",
        "role": "target",
        "text": "Kagababa semaba temaba vemaba wemaba xemaba."
      },
      {
        "answer": "fimaba",
        "bad": " gimaba",
        "good": " fimaba",
        "id": "measure-rif-19-1",
        "name": "Lagababa",
        "prompt": "Lagababa zemaba bimaba cimaba dimaba",
        "rel": "mrif19",
        "role": "competitor",
        "text": "Lagababa zemaba bimaba cimaba dimaba fimaba."
      },
      {
        "answer": "mimaba",
        "bad": " nimaba",
        "good": " mimaba",
        "id": "measure-rif-19-2",
        "name": "Magababa",
        "prompt": "Magababa himaba jimaba kimaba limaba",
        "rel": "mrif19",
        "role": "competitor",
        "text": "Magababa himaba jimaba kimaba limaba mimaba."
      },
      {
        "answer": "timaba",
        "bad": " vimaba",
        "good": " timaba",
        "id": "measure-rif-19-3",
        "name": "Nagababa",
        "prompt": "Nagababa pimaba qimaba rimaba simaba",
        "rel": "mrif19",
        "role": "competitor",
        "text": "Nagababa pimaba qimaba rimaba simaba timaba."
      },
      {
        "answer": "bomaba",
        "bad": " comaba",
        "good": " bomaba",
        "id": "measure-rif-19-4",
        "name": "Pagababa",
        "prompt": "Pagababa wimaba ximaba yimaba zimaba",
        "rel": "mrifx19",
        "role": "outsider",
        "text": "Pagababa wimaba ximaba yimaba zimaba bomaba."
      }
    ],
    [
      {
        "answer": "jomaba",
        "bad": " komaba",
        "good": " jomaba",
        "id": "measure-rif-20-0",
        "name": "Qagababa",
        "prompt": "Qagababa domaba fomaba gomaba homaba",
        "rel": "mrif20",
        "role": "target",
        "text": "Qagababa domaba fomaba gomaba homaba jomaba."
      },
      {
        "answer": "qomaba",
        "bad": " romaba",
        "good": " qomaba",
        "id": "measure-rif-20-1",
        "name": "Ragababa",
        "prompt": "Ragababa lomaba momaba nomaba pomaba",
        "rel": "mrif20",
        "role": "competitor",
        "text": "Ragababa lomaba momaba nomaba pomaba qomaba."
      },
      {
        "answer": "xomaba",
        "bad": " yomaba",
        "good": " xomaba",
        "id": "measure-rif-20-2",
        "name": "Sagababa",
        "prompt": "Sagababa somaba tomaba vomaba womaba",
        "rel": "mrif20",
        "role": "competitor",
        "text": "Sagababa somaba tomaba vomaba womaba xomaba."
      },
      {
        "answer": "fumaba",
        "bad": " gumaba",
        "good": " fumaba",
        "id": "measure-rif-20-3",
        "name": "Tagababa",
        "prompt": "Tagababa zomaba bumaba cumaba dumaba",
        "rel": "mrif20",
        "role": "competitor",
        "text": "Tagababa zomaba bumaba cumaba dumaba fumaba."
      },
      {
        "answer": "mumaba",
        "bad": " numaba",
        "good": " mumaba",
        "id": "measure-rif-20-4",
        "name": "Vagababa",
        "prompt": "Vagababa humaba jumaba kumaba lumaba",
        "rel": "mrifx20",
        "role": "outsider",
        "text": "Vagababa humaba jumaba kumaba lumaba mumaba."
      }
    ],
    [
      {
        "answer": "tumaba",
        "bad": " vumaba",
        "good": " tumaba",
        "id": "measure-rif-21-0",
        "name": "Wagababa",
        "prompt": "Wagababa pumaba qumaba rumaba sumaba",
        "rel": "mrif21",
        "role": "target",
        "text": "Wagababa pumaba qumaba rumaba sumaba tumaba."
      },
      {
        "answer": "banaba",
        "bad": " canaba",
        "good": " banaba",
        "id": "measure-rif-21-1",
        "name": "Xagababa",
        "prompt": "Xagababa wumaba xumaba yumaba zumaba",
        "rel": "mrif21",
        "role": "competitor",
        "text": "Xagababa wumaba xumaba yumaba zumaba banaba."
      },
      {
        "answer": "janaba",
        "bad": " kanaba",
        "good": " janaba",
        "id": "measure-rif-21-2",
        "name": "Yagababa",
        "prompt": "Yagababa danaba fanaba ganaba hanaba",
        "rel": "mrif21",
        "role": "competitor",
        "text": "Yagababa danaba fanaba ganaba hanaba janaba."
      },
      {
        "answer": "qanaba",
        "bad": " ranaba",
        "good": " qanaba",
        "id": "measure-rif-21-3",
        "name": "Zagababa",
        "prompt": "Zagababa lanaba manaba nanaba panaba",
        "rel": "mrif21",
        "role": "competitor",
        "text": "Zagababa lanaba manaba nanaba panaba qanaba."
      },
      {
        "answer": "xanaba",
        "bad": " yanaba",
        "good": " xanaba",
        "id": "measure-rif-21-4",
        "name": "Begababa",
        "prompt": "Begababa sanaba tanaba vanaba wanaba",
        "rel": "mrifx21",
        "role": "outsider",
        "text": "Begababa sanaba tanaba vanaba wanaba xanaba."
      }
    ],
    [
      {
        "answer": "fenaba",
        "bad": " genaba",
        "good": " fenaba",
        "id": "measure-rif-22-0",
        "name": "Cegababa",
        "prompt": "Cegababa zanaba benaba cenaba denaba",
        "rel": "mrif22",
        "role": "target",
        "text": "Cegababa zanaba benaba cenaba denaba fenaba."
      },
      {
        "answer": "menaba",
        "bad": " nenaba",
        "good": " menaba",
        "id": "measure-rif-22-1",
        "name": "Degababa",
        "prompt": "Degababa henaba jenaba kenaba lenaba",
        "rel": "mrif22",
        "role": "competitor",
        "text": "Degababa henaba jenaba kenaba lenaba menaba."
      },
      {
        "answer": "tenaba",
        "bad": " venaba",
        "good": " tenaba",
        "id": "measure-rif-22-2",
        "name": "Fegababa",
        "prompt": "Fegababa penaba qenaba renaba senaba",
        "rel": "mrif22",
        "role": "competitor",
        "text": "Fegababa penaba qenaba renaba senaba tenaba."
      },
      {
        "answer": "binaba",
        "bad": " cinaba",
        "good": " binaba",
        "id": "measure-rif-22-3",
        "name": "Gegababa",
        "prompt": "Gegababa wenaba xenaba yenaba zenaba",
        "rel": "mrif22",
        "role": "competitor",
        "text": "Gegababa wenaba xenaba yenaba zenaba binaba."
      },
      {
        "answer": "jinaba",
        "bad": " kinaba",
        "good": " jinaba",
        "id": "measure-rif-22-4",
        "name": "Hegababa",
        "prompt": "Hegababa dinaba finaba ginaba hinaba",
        "rel": "mrifx22",
        "role": "outsider",
        "text": "Hegababa dinaba finaba ginaba hinaba jinaba."
      }
    ],
    [
      {
        "answer": "qinaba",
        "bad": " rinaba",
        "good": " qinaba",
        "id": "measure-rif-23-0",
        "name": "Jegababa",
        "prompt": "Jegababa linaba minaba ninaba pinaba",
        "rel": "mrif23",
        "role": "target",
        "text": "Jegababa linaba minaba ninaba pinaba qinaba."
      },
      {
        "answer": "xinaba",
        "bad": " yinaba",
        "good": " xinaba",
        "id": "measure-rif-23-1",
        "name": "Kegababa",
        "prompt": "Kegababa sinaba tinaba vinaba winaba",
        "rel": "mrif23",
        "role": "competitor",
        "text": "Kegababa sinaba tinaba vinaba winaba xinaba."
      },
      {
        "answer": "fonaba",
        "bad": " gonaba",
        "good": " fonaba",
        "id": "measure-rif-23-2",
        "name": "Legababa",
        "prompt": "Legababa zinaba bonaba conaba donaba",
        "rel": "mrif23",
        "role": "competitor",
        "text": "Legababa zinaba bonaba conaba donaba fonaba."
      },
      {
        "answer": "monaba",
        "bad": " nonaba",
        "good": " monaba",
        "id": "measure-rif-23-3",
        "name": "Megababa",
        "prompt": "Megababa honaba jonaba konaba lonaba",
        "rel": "mrif23",
        "role": "competitor",
        "text": "Megababa honaba jonaba konaba lonaba monaba."
      },
      {
        "answer": "tonaba",
        "bad": " vonaba",
        "good": " tonaba",
        "id": "measure-rif-23-4",
        "name": "Negababa",
        "prompt": "Negababa ponaba qonaba ronaba sonaba",
        "rel": "mrifx23",
        "role": "outsider",
        "text": "Negababa ponaba qonaba ronaba sonaba tonaba."
      }
    ],
    [
      {
        "answer": "bunaba",
        "bad": " cunaba",
        "good": " bunaba",
        "id": "measure-rif-24-0",
        "name": "Pegababa",
        "prompt": "Pegababa wonaba xonaba yonaba zonaba",
        "rel": "mrif24",
        "role": "target",
        "text": "Pegababa wonaba xonaba yonaba zonaba bunaba."
      },
      {
        "answer": "junaba",
        "bad": " kunaba",
        "good": " junaba",
        "id": "measure-rif-24-1",
        "name": "Qegababa",
        "prompt": "Qegababa dunaba funaba gunaba hunaba",
        "rel": "mrif24",
        "role": "competitor",
        "text": "Qegababa dunaba funaba gunaba hunaba junaba."
      },
      {
        "answer": "qunaba",
        "bad": " runaba",
        "good": " qunaba",
        "id": "measure-rif-24-2",
        "name": "Regababa",
        "prompt": "Regababa lunaba munaba nunaba punaba",
        "rel": "mrif24",
        "role": "competitor",
        "text": "Regababa lunaba munaba nunaba punaba qunaba."
      },
      {
        "answer": "xunaba",
        "bad": " yunaba",
        "good": " xunaba",
        "id": "measure-rif-24-3",
        "name": "Segababa",
        "prompt": "Segababa sunaba tunaba vunaba wunaba",
        "rel": "mrif24",
        "role": "competitor",
        "text": "Segababa sunaba tunaba vunaba wunaba xunaba."
      },
      {
        "answer": "fapaba",
        "bad": " gapaba",
        "good": " fapaba",
        "id": "measure-rif-24-4",
        "name": "Tegababa",
        "prompt": "Tegababa zunaba bapaba capaba dapaba",
        "rel": "mrifx24",
        "role": "outsider",
        "text": "Tegababa zunaba bapaba capaba dapaba fapaba."
      }
    ],
    [
      {
        "answer": "mapaba",
        "bad": " napaba",
        "good": " mapaba",
        "id": "measure-rif-25-0",
        "name": "Vegababa",
        "prompt": "Vegababa hapaba japaba kapaba lapaba",
        "rel": "mrif25",
        "role": "target",
        "text": "Vegababa hapaba japaba kapaba lapaba mapaba."
      },
      {
        "answer": "tapaba",
        "bad": " vapaba",
        "good": " tapaba",
        "id": "measure-rif-25-1",
        "name": "Wegababa",
        "prompt": "Wegababa papaba qapaba rapaba sapaba",
        "rel": "mrif25",
        "role": "competitor",
        "text": "Wegababa papaba qapaba rapaba sapaba tapaba."
      },
      {
        "answer": "bepaba",
        "bad": " cepaba",
        "good": " bepaba",
        "id": "measure-rif-25-2",
        "name": "Xegababa",
        "prompt": "Xegababa wapaba xapaba yapaba zapaba",
        "rel": "mrif25",
        "role": "competitor",
        "text": "Xegababa wapaba xapaba yapaba zapaba bepaba."
      },
      {
        "answer": "jepaba",
        "bad": " kepaba",
        "good": " jepaba",
        "id": "measure-rif-25-3",
        "name": "Yegababa",
        "prompt": "Yegababa depaba fepaba gepaba hepaba",
        "rel": "mrif25",
        "role": "competitor",
        "text": "Yegababa depaba fepaba gepaba hepaba jepaba."
      },
      {
        "answer": "qepaba",
        "bad": " repaba",
        "good": " qepaba",
        "id": "measure-rif-25-4",
        "name": "Zegababa",
        "prompt": "Zegababa lepaba mepaba nepaba pepaba",
        "rel": "mrifx25",
        "role": "outsider",
        "text": "Zegababa lepaba mepaba nepaba pepaba qepaba."
      }
    ],
    [
      {
        "answer": "xepaba",
        "bad": " yepaba",
        "good": " xepaba",
        "id": "measure-rif-26-0",
        "name": "Bigababa",
        "prompt": "Bigababa sepaba tepaba vepaba wepaba",
        "rel": "mrif26",
        "role": "target",
        "text": "Bigababa sepaba tepaba vepaba wepaba xepaba."
      },
      {
        "answer": "fipaba",
        "bad": " gipaba",
        "good": " fipaba",
        "id": "measure-rif-26-1",
        "name": "Cigababa",
        "prompt": "Cigababa zepaba bipaba cipaba dipaba",
        "rel": "mrif26",
        "role": "competitor",
        "text": "Cigababa zepaba bipaba cipaba dipaba fipaba."
      },
      {
        "answer": "mipaba",
        "bad": " nipaba",
        "good": " mipaba",
        "id": "measure-rif-26-2",
        "name": "Digababa",
        "prompt": "Digababa hipaba jipaba kipaba lipaba",
        "rel": "mrif26",
        "role": "competitor",
        "text": "Digababa hipaba jipaba kipaba lipaba mipaba."
      },
      {
        "answer": "tipaba",
        "bad": " vipaba",
        "good": " tipaba",
        "id": "measure-rif-26-3",
        "name": "Figababa",
        "prompt": "Figababa pipaba qipaba ripaba sipaba",
        "rel": "mrif26",
        "role": "competitor",
        "text": "Figababa pipaba qipaba ripaba sipaba tipaba."
      },
      {
        "answer": "bopaba",
        "bad": " copaba",
        "good": " bopaba",
        "id": "measure-rif-26-4",
        "name": "Gigababa",
        "prompt": "Gigababa wipaba xipaba yipaba zipaba",
        "rel": "mrifx26",
        "role": "outsider",
        "text": "Gigababa wipaba xipaba yipaba zipaba bopaba."
      }
    ],
    [
      {
        "answer": "jopaba",
        "bad": " kopaba",
        "good": " jopaba",
        "id": "measure-rif-27-0",
        "name": "Higababa",
        "prompt": "Higababa dopaba fopaba gopaba hopaba",
        "rel": "mrif27",
        "role": "target",
        "text": "Higababa dopaba fopaba gopaba hopaba jopaba."
      },
      {
        "answer": "qopaba",
        "bad": " ropaba",
        "good": " qopaba",
        "id": "measure-rif-27-1",
        "name": "Jigababa",
        "prompt": "Jigababa lopaba mopaba nopaba popaba",
        "rel": "mrif27",
        "role": "competitor",
        "text": "Jigababa lopaba mopaba nopaba popaba qopaba."
      },
      {
        "answer": "xopaba",
        "bad": " yopaba",
        "good": " xopaba",
        "id": "measure-rif-27-2",
        "name": "Kigababa",
        "prompt": "Kigababa sopaba topaba vopaba wopaba",
        "rel": "mrif27",
        "role": "competitor",
        "text": "Kigababa sopaba topaba vopaba wopaba xopaba."
      },
      {
        "answer": "fupaba",
        "bad": " gupaba",
        "good": " fupaba",
        "id": "measure-rif-27-3",
        "name": "Ligababa",
        "prompt": "Ligababa zopaba bupaba cupaba dupaba",
        "rel": "mrif27",
        "role": "competitor",
        "text": "Ligababa zopaba bupaba cupaba dupaba fupaba."
      },
      {
        "answer": "mupaba",
        "bad": " nupaba",
        "good": " mupaba",
        "id": "measure-rif-27-4",
        "name": "Migababa",
        "prompt": "Migababa hupaba jupaba kupaba lupaba",
        "rel": "mrifx27",
        "role": "outsider",
        "text": "Migababa hupaba jupaba kupaba lupaba mupaba."
      }
    ],
    [
      {
        "answer": "tupaba",
        "bad": " vupaba",
        "good": " tupaba",
        "id": "measure-rif-28-0",
        "name": "Nigababa",
        "prompt": "Nigababa pupaba qupaba rupaba supaba",
        "rel": "mrif28",
        "role": "target",
        "text": "Nigababa pupaba qupaba rupaba supaba tupaba."
      },
      {
        "answer": "baqaba",
        "bad": " caqaba",
        "good": " baqaba",
        "id": "measure-rif-28-1",
        "name": "Pigababa",
        "prompt": "Pigababa wupaba xupaba yupaba zupaba",
        "rel": "mrif28",
        "role": "competitor",
        "text": "Pigababa wupaba xupaba yupaba zupaba baqaba."
      },
      {
        "answer": "jaqaba",
        "bad": " kaqaba",
        "good": " jaqaba",
        "id": "measure-rif-28-2",
        "name": "Qigababa",
        "prompt": "Qigababa daqaba faqaba gaqaba haqaba",
        "rel": "mrif28",
        "role": "competitor",
        "text": "Qigababa daqaba faqaba gaqaba haqaba jaqaba."
      },
      {
        "answer": "qaqaba",
        "bad": " raqaba",
        "good": " qaqaba",
        "id": "measure-rif-28-3",
        "name": "Rigababa",
        "prompt": "Rigababa laqaba maqaba naqaba paqaba",
        "rel": "mrif28",
        "role": "competitor",
        "text": "Rigababa laqaba maqaba naqaba paqaba qaqaba."
      },
      {
        "answer": "xaqaba",
        "bad": " yaqaba",
        "good": " xaqaba",
        "id": "measure-rif-28-4",
        "name": "Sigababa",
        "prompt": "Sigababa saqaba taqaba vaqaba waqaba",
        "rel": "mrifx28",
        "role": "outsider",
        "text": "Sigababa saqaba taqaba vaqaba waqaba xaqaba."
      }
    ],
    [
      {
        "answer": "feqaba",
        "bad": " geqaba",
        "good": " feqaba",
        "id": "measure-rif-29-0",
        "name": "Tigababa",
        "prompt": "Tigababa zaqaba beqaba ceqaba deqaba",
        "rel": "mrif29",
        "role": "target",
        "text": "Tigababa zaqaba beqaba ceqaba deqaba feqaba."
      },
      {
        "answer": "meqaba",
        "bad": " neqaba",
        "good": " meqaba",
        "id": "measure-rif-29-1",
        "name": "Vigababa",
        "prompt": "Vigababa heqaba jeqaba keqaba leqaba",
        "rel": "mrif29",
        "role": "competitor",
        "text": "Vigababa heqaba jeqaba keqaba leqaba meqaba."
      },
      {
        "answer": "teqaba",
        "bad": " veqaba",
        "good": " teqaba",
        "id": "measure-rif-29-2",
        "name": "Wigababa",
        "prompt": "Wigababa peqaba qeqaba reqaba seqaba",
        "rel": "mrif29",
        "role": "competitor",
        "text": "Wigababa peqaba qeqaba reqaba seqaba teqaba."
      },
      {
        "answer": "biqaba",
        "bad": " ciqaba",
        "good": " biqaba",
        "id": "measure-rif-29-3",
        "name": "Xigababa",
        "prompt": "Xigababa weqaba xeqaba yeqaba zeqaba",
        "rel": "mrif29",
        "role": "competitor",
        "text": "Xigababa weqaba xeqaba yeqaba zeqaba biqaba."
      },
      {
        "answer": "jiqaba",
        "bad": " kiqaba",
        "good": " jiqaba",
        "id": "measure-rif-29-4",
        "name": "Yigababa",
        "prompt": "Yigababa diqaba fiqaba giqaba hiqaba",
        "rel": "mrifx29",
        "role": "outsider",
        "text": "Yigababa diqaba fiqaba giqaba hiqaba jiqaba."
      }
    ],
    [
      {
        "answer": "qiqaba",
        "bad": " riqaba",
        "good": " qiqaba",
        "id": "measure-rif-30-0",
        "name": "Zigababa",
        "prompt": "Zigababa liqaba miqaba niqaba piqaba",
        "rel": "mrif30",
        "role": "target",
        "text": "Zigababa liqaba miqaba niqaba piqaba qiqaba."
      },
      {
        "answer": "xiqaba",
        "bad": " yiqaba",
        "good": " xiqaba",
        "id": "measure-rif-30-1",
        "name": "Bogababa",
        "prompt": "Bogababa siqaba tiqaba viqaba wiqaba",
        "rel": "mrif30",
        "role": "competitor",
        "text": "Bogababa siqaba tiqaba viqaba wiqaba xiqaba."
      },
      {
        "answer": "foqaba",
        "bad": " goqaba",
        "good": " foqaba",
        "id": "measure-rif-30-2",
        "name": "Cogababa",
        "prompt": "Cogababa ziqaba boqaba coqaba doqaba",
        "rel": "mrif30",
        "role": "competitor",
        "text": "Cogababa ziqaba boqaba coqaba doqaba foqaba."
      },
      {
        "answer": "moqaba",
        "bad": " noqaba",
        "good": " moqaba",
        "id": "measure-rif-30-3",
        "name": "Dogababa",
        "prompt": "Dogababa hoqaba joqaba koqaba loqaba",
        "rel": "mrif30",
        "role": "competitor",
        "text": "Dogababa hoqaba joqaba koqaba loqaba moqaba."
      },
      {
        "answer": "toqaba",
        "bad": " voqaba",
        "good": " toqaba",
        "id": "measure-rif-30-4",
        "name": "Fogababa",
        "prompt": "Fogababa poqaba qoqaba roqaba soqaba",
        "rel": "mrifx30",
        "role": "outsider",
        "text": "Fogababa poqaba qoqaba roqaba soqaba toqaba."
      }
    ],
    [
      {
        "answer": "buqaba",
        "bad": " cuqaba",
        "good": " buqaba",
        "id": "measure-rif-31-0",
        "name": "Gogababa",
        "prompt": "Gogababa woqaba xoqaba yoqaba zoqaba",
        "rel": "mrif31",
        "role": "target",
        "text": "Gogababa woqaba xoqaba yoqaba zoqaba buqaba."
      },
      {
        "answer": "juqaba",
        "bad": " kuqaba",
        "good": " juqaba",
        "id": "measure-rif-31-1",
        "name": "Hogababa",
        "prompt": "Hogababa duqaba fuqaba guqaba huqaba",
        "rel": "mrif31",
        "role": "competitor",
        "text": "Hogababa duqaba fuqaba guqaba huqaba juqaba."
      },
      {
        "answer": "quqaba",
        "bad": " ruqaba",
        "good": " quqaba",
        "id": "measure-rif-31-2",
        "name": "Jogababa",
        "prompt": "Jogababa luqaba muqaba nuqaba puqaba",
        "rel": "mrif31",
        "role": "competitor",
        "text": "Jogababa luqaba muqaba nuqaba puqaba quqaba."
      },
      {
        "answer": "xuqaba",
        "bad": " yuqaba",
        "good": " xuqaba",
        "id": "measure-rif-31-3",
        "name": "Kogababa",
        "prompt": "Kogababa suqaba tuqaba vuqaba wuqaba",
        "rel": "mrif31",
        "role": "competitor",
        "text": "Kogababa suqaba tuqaba vuqaba wuqaba xuqaba."
      },
      {
        "answer": "faraba",
        "bad": " garaba",
        "good": " faraba",
        "id": "measure-rif-31-4",
        "name": "Logababa",
        "prompt": "Logababa zuqaba baraba caraba daraba",
        "rel": "mrifx31",
        "role": "outsider",
        "text": "Logababa zuqaba baraba caraba daraba faraba."
      }
    ],
    [
      {
        "answer": "maraba",
        "bad": " naraba",
        "good": " maraba",
        "id": "measure-rif-32-0",
        "name": "Mogababa",
        "prompt": "Mogababa haraba jaraba karaba laraba",
        "rel": "mrif32",
        "role": "target",
        "text": "Mogababa haraba jaraba karaba laraba maraba."
      },
      {
        "answer": "taraba",
        "bad": " varaba",
        "good": " taraba",
        "id": "measure-rif-32-1",
        "name": "Nogababa",
        "prompt": "Nogababa paraba qaraba raraba saraba",
        "rel": "mrif32",
        "role": "competitor",
        "text": "Nogababa paraba qaraba raraba saraba taraba."
      },
      {
        "answer": "beraba",
        "bad": " ceraba",
        "good": " beraba",
        "id": "measure-rif-32-2",
        "name": "Pogababa",
        "prompt": "Pogababa waraba xaraba yaraba zaraba",
        "rel": "mrif32",
        "role": "competitor",
        "text": "Pogababa waraba xaraba yaraba zaraba beraba."
      },
      {
        "answer": "jeraba",
        "bad": " keraba",
        "good": " jeraba",
        "id": "measure-rif-32-3",
        "name": "Qogababa",
        "prompt": "Qogababa deraba feraba geraba heraba",
        "rel": "mrif32",
        "role": "competitor",
        "text": "Qogababa deraba feraba geraba heraba jeraba."
      },
      {
        "answer": "qeraba",
        "bad": " reraba",
        "good": " qeraba",
        "id": "measure-rif-32-4",
        "name": "Rogababa",
        "prompt": "Rogababa leraba meraba neraba peraba",
        "rel": "mrifx32",
        "role": "outsider",
        "text": "Rogababa leraba meraba neraba peraba qeraba."
      }
    ],
    [
      {
        "answer": "xeraba",
        "bad": " yeraba",
        "good": " xeraba",
        "id": "measure-rif-33-0",
        "name": "Sogababa",
        "prompt": "Sogababa seraba teraba veraba weraba",
        "rel": "mrif33",
        "role": "target",
        "text": "Sogababa seraba teraba veraba weraba xeraba."
      },
      {
        "answer": "firaba",
        "bad": " giraba",
        "good": " firaba",
        "id": "measure-rif-33-1",
        "name": "Togababa",
        "prompt": "Togababa zeraba biraba ciraba diraba",
        "rel": "mrif33",
        "role": "competitor",
        "text": "Togababa zeraba biraba ciraba diraba firaba."
      },
      {
        "answer": "miraba",
        "bad": " niraba",
        "good": " miraba",
        "id": "measure-rif-33-2",
        "name": "Vogababa",
        "prompt": "Vogababa hiraba jiraba kiraba liraba",
        "rel": "mrif33",
        "role": "competitor",
        "text": "Vogababa hiraba jiraba kiraba liraba miraba."
      },
      {
        "answer": "tiraba",
        "bad": " viraba",
        "good": " tiraba",
        "id": "measure-rif-33-3",
        "name": "Wogababa",
        "prompt": "Wogababa piraba qiraba riraba siraba",
        "rel": "mrif33",
        "role": "competitor",
        "text": "Wogababa piraba qiraba riraba siraba tiraba."
      },
      {
        "answer": "boraba",
        "bad": " coraba",
        "good": " boraba",
        "id": "measure-rif-33-4",
        "name": "Xogababa",
        "prompt": "Xogababa wiraba xiraba yiraba ziraba",
        "rel": "mrifx33",
        "role": "outsider",
        "text": "Xogababa wiraba xiraba yiraba ziraba boraba."
      }
    ],
    [
      {
        "answer": "joraba",
        "bad": " koraba",
        "good": " joraba",
        "id": "measure-rif-34-0",
        "name": "Yogababa",
        "prompt": "Yogababa doraba foraba goraba horaba",
        "rel": "mrif34",
        "role": "target",
        "text": "Yogababa doraba foraba goraba horaba joraba."
      },
      {
        "answer": "qoraba",
        "bad": " roraba",
        "good": " qoraba",
        "id": "measure-rif-34-1",
        "name": "Zogababa",
        "prompt": "Zogababa loraba moraba noraba poraba",
        "rel": "mrif34",
        "role": "competitor",
        "text": "Zogababa loraba moraba noraba poraba qoraba."
      },
      {
        "answer": "xoraba",
        "bad": " yoraba",
        "good": " xoraba",
        "id": "measure-rif-34-2",
        "name": "Bugababa",
        "prompt": "Bugababa soraba toraba voraba woraba",
        "rel": "mrif34",
        "role": "competitor",
        "text": "Bugababa soraba toraba voraba woraba xoraba."
      },
      {
        "answer": "furaba",
        "bad": " guraba",
        "good": " furaba",
        "id": "measure-rif-34-3",
        "name": "Cugababa",
        "prompt": "Cugababa zoraba buraba curaba duraba",
        "rel": "mrif34",
        "role": "competitor",
        "text": "Cugababa zoraba buraba curaba duraba furaba."
      },
      {
        "answer": "muraba",
        "bad": " nuraba",
        "good": " muraba",
        "id": "measure-rif-34-4",
        "name": "Dugababa",
        "prompt": "Dugababa huraba juraba kuraba luraba",
        "rel": "mrifx34",
        "role": "outsider",
        "text": "Dugababa huraba juraba kuraba luraba muraba."
      }
    ],
    [
      {
        "answer": "turaba",
        "bad": " vuraba",
        "good": " turaba",
        "id": "measure-rif-35-0",
        "name": "Fugababa",
        "prompt": "Fugababa puraba quraba ruraba suraba",
        "rel": "mrif35",
        "role": "target",
        "text": "Fugababa puraba quraba ruraba suraba turaba."
      },
      {
        "answer": "basaba",
        "bad": " casaba",
        "good": " basaba",
        "id": "measure-rif-35-1",
        "name": "Gugababa",
        "prompt": "Gugababa wuraba xuraba yuraba zuraba",
        "rel": "mrif35",
        "role": "competitor",
        "text": "Gugababa wuraba xuraba yuraba zuraba basaba."
      },
      {
        "answer": "jasaba",
        "bad": " kasaba",
        "good": " jasaba",
        "id": "measure-rif-35-2",
        "name": "Hugababa",
        "prompt": "Hugababa dasaba fasaba gasaba hasaba",
        "rel": "mrif35",
        "role": "competitor",
        "text": "Hugababa dasaba fasaba gasaba hasaba jasaba."
      },
      {
        "answer": "qasaba",
        "bad": " rasaba",
        "good": " qasaba",
        "id": "measure-rif-35-3",
        "name": "Jugababa",
        "prompt": "Jugababa lasaba masaba nasaba pasaba",
        "rel": "mrif35",
        "role": "competitor",
        "text": "Jugababa lasaba masaba nasaba pasaba qasaba."
      },
      {
        "answer": "xasaba",
        "bad": " yasaba",
        "good": " xasaba",
        "id": "measure-rif-35-4",
        "name": "Kugababa",
        "prompt": "Kugababa sasaba tasaba vasaba wasaba",
        "rel": "mrifx35",
        "role": "outsider",
        "text": "Kugababa sasaba tasaba vasaba wasaba xasaba."
      }
    ],
    [
      {
        "answer": "fesaba",
        "bad": " gesaba",
        "good": " fesaba",
        "id": "measure-rif-36-0",
        "name": "Lugababa",
        "prompt": "Lugababa zasaba besaba cesaba desaba",
        "rel": "mrif36",
        "role": "target",
        "text": "Lugababa zasaba besaba cesaba desaba fesaba."
      },
      {
        "answer": "mesaba",
        "bad": " nesaba",
        "good": " mesaba",
        "id": "measure-rif-36-1",
        "name": "Mugababa",
        "prompt": "Mugababa hesaba jesaba kesaba lesaba",
        "rel": "mrif36",
        "role": "competitor",
        "text": "Mugababa hesaba jesaba kesaba lesaba mesaba."
      },
      {
        "answer": "tesaba",
        "bad": " vesaba",
        "good": " tesaba",
        "id": "measure-rif-36-2",
        "name": "Nugababa",
        "prompt": "Nugababa pesaba qesaba resaba sesaba",
        "rel": "mrif36",
        "role": "competitor",
        "text": "Nugababa pesaba qesaba resaba sesaba tesaba."
      },
      {
        "answer": "bisaba",
        "bad": " cisaba",
        "good": " bisaba",
        "id": "measure-rif-36-3",
        "name": "Pugababa",
        "prompt": "Pugababa wesaba xesaba yesaba zesaba",
        "rel": "mrif36",
        "role": "competitor",
        "text": "Pugababa wesaba xesaba yesaba zesaba bisaba."
      },
      {
        "answer": "jisaba",
        "bad": " kisaba",
        "good": " jisaba",
        "id": "measure-rif-36-4",
        "name": "Qugababa",
        "prompt": "Qugababa disaba fisaba gisaba hisaba",
        "rel": "mrifx36",
        "role": "outsider",
        "text": "Qugababa disaba fisaba gisaba hisaba jisaba."
      }
    ],
    [
      {
        "answer": "qisaba",
        "bad": " risaba",
        "good": " qisaba",
        "id": "measure-rif-37-0",
        "name": "Rugababa",
        "prompt": "Rugababa lisaba misaba nisaba pisaba",
        "rel": "mrif37",
        "role": "target",
        "text": "Rugababa lisaba misaba nisaba pisaba qisaba."
      },
      {
        "answer": "xisaba",
        "bad": " yisaba",
        "good": " xisaba",
        "id": "measure-rif-37-1",
        "name": "Sugababa",
        "prompt": "Sugababa sisaba tisaba visaba wisaba",
        "rel": "mrif37",
        "role": "competitor",
        "text": "Sugababa sisaba tisaba visaba wisaba xisaba."
      },
      {
        "answer": "fosaba",
        "bad": " gosaba",
        "good": " fosaba",
        "id": "measure-rif-37-2",
        "name": "Tugababa",
        "prompt": "Tugababa zisaba bosaba cosaba dosaba",
        "rel": "mrif37",
        "role": "competitor",
        "text": "Tugababa zisaba bosaba cosaba dosaba fosaba."
      },
      {
        "answer": "mosaba",
        "bad": " nosaba",
        "good": " mosaba",
        "id": "measure-rif-37-3",
        "name": "Vugababa",
        "prompt": "Vugababa hosaba josaba kosaba losaba",
        "rel": "mrif37",
        "role": "competitor",
        "text": "Vugababa hosaba josaba kosaba losaba mosaba."
      },
      {
        "answer": "tosaba",
        "bad": " vosaba",
        "good": " tosaba",
        "id": "measure-rif-37-4",
        "name": "Wugababa",
        "prompt": "Wugababa posaba qosaba rosaba sosaba",
        "rel": "mrifx37",
        "role": "outsider",
        "text": "Wugababa posaba qosaba rosaba sosaba tosaba."
      }
    ],
    [
      {
        "answer": "busaba",
        "bad": " cusaba",
        "good": " busaba",
        "id": "measure-rif-38-0",
        "name": "Xugababa",
        "prompt": "Xugababa wosaba xosaba yosaba zosaba",
        "rel": "mrif38",
        "role": "target",
        "text": "Xugababa wosaba xosaba yosaba zosaba busaba."
      },
      {
        "answer": "jusaba",
        "bad": " kusaba",
        "good": " jusaba",
        "id": "measure-rif-38-1",
        "name": "Yugababa",
        "prompt": "Yugababa dusaba fusaba gusaba husaba",
        "rel": "mrif38",
        "role": "competitor",
        "text": "Yugababa dusaba fusaba gusaba husaba jusaba."
      },
      {
        "answer": "qusaba",
        "bad": " rusaba",
        "good": " qusaba",
        "id": "measure-rif-38-2",
        "name": "Zugababa",
        "prompt": "Zugababa lusaba musaba nusaba pusaba",
        "rel": "mrif38",
        "role": "competitor",
        "text": "Zugababa lusaba musaba nusaba pusaba qusaba."
      },
      {
        "answer": "xusaba",
        "bad": " yusaba",
        "good": " xusaba",
        "id": "measure-rif-38-3",
        "name": "Bahababa",
        "prompt": "Bahababa susaba tusaba vusaba wusaba",
        "rel": "mrif38",
        "role": "competitor",
        "text": "Bahababa susaba tusaba vusaba wusaba xusaba."
      },
      {
        "answer": "fataba",
        "bad": " gataba",
        "good": " fataba",
        "id": "measure-rif-38-4",
        "name": "Cahababa",
        "prompt": "Cahababa zusaba bataba cataba databa",
        "rel": "mrifx38",
        "role": "outsider",
        "text": "Cahababa zusaba bataba cataba databa fataba."
      }
    ],
    [
      {
        "answer": "mataba",
        "bad": " nataba",
        "good": " mataba",
        "id": "measure-rif-39-0",
        "name": "Dahababa",
        "prompt": "Dahababa hataba jataba kataba lataba",
        "rel": "mrif39",
        "role": "target",
        "text": "Dahababa hataba jataba kataba lataba mataba."
      },
      {
        "answer": "tataba",
        "bad": " vataba",
        "good": " tataba",
        "id": "measure-rif-39-1",
        "name": "Fahababa",
        "prompt": "Fahababa pataba qataba rataba sataba",
        "rel": "mrif39",
        "role": "competitor",
        "text": "Fahababa pataba qataba rataba sataba tataba."
      },
      {
        "answer": "betaba",
        "bad": " cetaba",
        "good": " betaba",
        "id": "measure-rif-39-2",
        "name": "Gahababa",
        "prompt": "Gahababa wataba xataba yataba zataba",
        "rel": "mrif39",
        "role": "competitor",
        "text": "Gahababa wataba xataba yataba zataba betaba."
      },
      {
        "answer": "jetaba",
        "bad": " ketaba",
        "good": " jetaba",
        "id": "measure-rif-39-3",
        "name": "Hahababa",
        "prompt": "Hahababa detaba fetaba getaba hetaba",
        "rel": "mrif39",
        "role": "competitor",
        "text": "Hahababa detaba fetaba getaba hetaba jetaba."
      },
      {
        "answer": "qetaba",
        "bad": " retaba",
        "good": " qetaba",
        "id": "measure-rif-39-4",
        "name": "Jahababa",
        "prompt": "Jahababa letaba metaba netaba petaba",
        "rel": "mrifx39",
        "role": "outsider",
        "text": "Jahababa letaba metaba netaba petaba qetaba."
      }
    ],
    [
      {
        "answer": "xetaba",
        "bad": " yetaba",
        "good": " xetaba",
        "id": "measure-rif-40-0",
        "name": "Kahababa",
        "prompt": "Kahababa setaba tetaba vetaba wetaba",
        "rel": "mrif40",
        "role": "target",
        "text": "Kahababa setaba tetaba vetaba wetaba xetaba."
      },
      {
        "answer": "fitaba",
        "bad": " gitaba",
        "good": " fitaba",
        "id": "measure-rif-40-1",
        "name": "Lahababa",
        "prompt": "Lahababa zetaba bitaba citaba ditaba",
        "rel": "mrif40",
        "role": "competitor",
        "text": "Lahababa zetaba bitaba citaba ditaba fitaba."
      },
      {
        "answer": "mitaba",
        "bad": " nitaba",
        "good": " mitaba",
        "id": "measure-rif-40-2",
        "name": "Mahababa",
        "prompt": "Mahababa hitaba jitaba kitaba litaba",
        "rel": "mrif40",
        "role": "competitor",
        "text": "Mahababa hitaba jitaba kitaba litaba mitaba."
      },
      {
        "answer": "titaba",
        "bad": " vitaba",
        "good": " titaba",
        "id": "measure-rif-40-3",
        "name": "Nahababa",
        "prompt": "Nahababa pitaba qitaba ritaba sitaba",
        "rel": "mrif40",
        "role": "competitor",
        "text": "Nahababa pitaba qitaba ritaba sitaba titaba."
      },
      {
        "answer": "botaba",
        "bad": " cotaba",
        "good": " botaba",
        "id": "measure-rif-40-4",
        "name": "Pahababa",
        "prompt": "Pahababa witaba xitaba yitaba zitaba",
        "rel": "mrifx40",
        "role": "outsider",
        "text": "Pahababa witaba xitaba yitaba zitaba botaba."
      }
    ],
    [
      {
        "answer": "jotaba",
        "bad": " kotaba",
        "good": " jotaba",
        "id": "measure-rif-41-0",
        "name": "Qahababa",
        "prompt": "Qahababa dotaba fotaba gotaba hotaba",
        "rel": "mrif41",
        "role": "target",
        "text": "Qahababa dotaba fotaba gotaba hotaba jotaba."
      },
      {
        "answer": "qotaba",
        "bad": " rotaba",
        "good": " qotaba",
        "id": "measure-rif-41-1",
        "name": "Rahababa",
        "prompt": "Rahababa lotaba motaba notaba potaba",
        "rel": "mrif41",
        "role": "competitor",
        "text": "Rahababa lotaba motaba notaba potaba qotaba."
      },
      {
        "answer": "xotaba",
        "bad": " yotaba",
        "good": " xotaba",
        "id": "measure-rif-41-2",
        "name": "Sahababa",
        "prompt": "Sahababa sotaba totaba votaba wotaba",
        "rel": "mrif41",
        "role": "competitor",
        "text": "Sahababa sotaba totaba votaba wotaba xotaba."
      },
      {
        "answer": "futaba",
        "bad": " gutaba",
        "good": " futaba",
        "id": "measure-rif-41-3",
        "name": "Tahababa",
        "prompt": "Tahababa zotaba butaba cutaba dutaba",
        "rel": "mrif41",
        "role": "competitor",
        "text": "Tahababa zotaba butaba cutaba dutaba futaba."
      },
      {
        "answer": "mutaba",
        "bad": " nutaba",
        "good": " mutaba",
        "id": "measure-rif-41-4",
        "name": "Vahababa",
        "prompt": "Vahababa hutaba jutaba kutaba lutaba",
        "rel": "mrifx41",
        "role": "outsider",
        "text": "Vahababa hutaba jutaba kutaba lutaba mutaba."
      }
    ],
    [
      {
        "answer": "tutaba",
        "bad": " vutaba",
        "good": " tutaba",
        "id": "measure-rif-42-0",
        "name": "Wahababa",
        "prompt": "Wahababa putaba qutaba rutaba sutaba",
        "rel": "mrif42",
        "role": "target",
        "text": "Wahababa putaba qutaba rutaba sutaba tutaba."
      },
      {
        "answer": "bavaba",
        "bad": " cavaba",
        "good": " bavaba",
        "id": "measure-rif-42-1",
        "name": "Xahababa",
        "prompt": "Xahababa wutaba xutaba yutaba zutaba",
        "rel": "mrif42",
        "role": "competitor",
        "text": "Xahababa wutaba xutaba yutaba zutaba bavaba."
      },
      {
        "answer": "javaba",
        "bad": " kavaba",
        "good": " javaba",
        "id": "measure-rif-42-2",
        "name": "Yahababa",
        "prompt": "Yahababa davaba favaba gavaba havaba",
        "rel": "mrif42",
        "role": "competitor",
        "text": "Yahababa davaba favaba gavaba havaba javaba."
      },
      {
        "answer": "qavaba",
        "bad": " ravaba",
        "good": " qavaba",
        "id": "measure-rif-42-3",
        "name": "Zahababa",
        "prompt": "Zahababa lavaba mavaba navaba pavaba",
        "rel": "mrif42",
        "role": "competitor",
        "text": "Zahababa lavaba mavaba navaba pavaba qavaba."
      },
      {
        "answer": "xavaba",
        "bad": " yavaba",
        "good": " xavaba",
        "id": "measure-rif-42-4",
        "name": "Behababa",
        "prompt": "Behababa savaba tavaba vavaba wavaba",
        "rel": "mrifx42",
        "role": "outsider",
        "text": "Behababa savaba tavaba vavaba wavaba xavaba."
      }
    ],
    [
      {
        "answer": "fevaba",
        "bad": " gevaba",
        "good": " fevaba",
        "id": "measure-rif-43-0",
        "name": "Cehababa",
        "prompt": "Cehababa zavaba bevaba cevaba devaba",
        "rel": "mrif43",
        "role": "target",
        "text": "Cehababa zavaba bevaba cevaba devaba fevaba."
      },
      {
        "answer": "mevaba",
        "bad": " nevaba",
        "good": " mevaba",
        "id": "measure-rif-43-1",
        "name": "Dehababa",
        "prompt": "Dehababa hevaba jevaba kevaba levaba",
        "rel": "mrif43",
        "role": "competitor",
        "text": "Dehababa hevaba jevaba kevaba levaba mevaba."
      },
      {
        "answer": "tevaba",
        "bad": " vevaba",
        "good": " tevaba",
        "id": "measure-rif-43-2",
        "name": "Fehababa",
        "prompt": "Fehababa pevaba qevaba revaba sevaba",
        "rel": "mrif43",
        "role": "competitor",
        "text": "Fehababa pevaba qevaba revaba sevaba tevaba."
      },
      {
        "answer": "bivaba",
        "bad": " civaba",
        "good": " bivaba",
        "id": "measure-rif-43-3",
        "name": "Gehababa",
        "prompt": "Gehababa wevaba xevaba yevaba zevaba",
        "rel": "mrif43",
        "role": "competitor",
        "text": "Gehababa wevaba xevaba yevaba zevaba bivaba."
      },
      {
        "answer": "jivaba",
        "bad": " kivaba",
        "good": " jivaba",
        "id": "measure-rif-43-4",
        "name": "Hehababa",
        "prompt": "Hehababa divaba fivaba givaba hivaba",
        "rel": "mrifx43",
        "role": "outsider",
        "text": "Hehababa divaba fivaba givaba hivaba jivaba."
      }
    ],
    [
      {
        "answer": "qivaba",
        "bad": " rivaba",
        "good": " qivaba",
        "id": "measure-rif-44-0",
        "name": "Jehababa",
        "prompt": "Jehababa livaba mivaba nivaba pivaba",
        "rel": "mrif44",
        "role": "target",
        "text": "Jehababa livaba mivaba nivaba pivaba qivaba."
      },
      {
        "answer": "xivaba",
        "bad": " yivaba",
        "good": " xivaba",
        "id": "measure-rif-44-1",
        "name": "Kehababa",
        "prompt": "Kehababa sivaba tivaba vivaba wivaba",
        "rel": "mrif44",
        "role": "competitor",
        "text": "Kehababa sivaba tivaba vivaba wivaba xivaba."
      },
      {
        "answer": "fovaba",
        "bad": " govaba",
        "good": " fovaba",
        "id": "measure-rif-44-2",
        "name": "Lehababa",
        "prompt": "Lehababa zivaba bovaba covaba dovaba",
        "rel": "mrif44",
        "role": "competitor",
        "text": "Lehababa zivaba bovaba covaba dovaba fovaba."
      },
      {
        "answer": "movaba",
        "bad": " novaba",
        "good": " movaba",
        "id": "measure-rif-44-3",
        "name": "Mehababa",
        "prompt": "Mehababa hovaba jovaba kovaba lovaba",
        "rel": "mrif44",
        "role": "competitor",
        "text": "Mehababa hovaba jovaba kovaba lovaba movaba."
      },
      {
        "answer": "tovaba",
        "bad": " vovaba",
        "good": " tovaba",
        "id": "measure-rif-44-4",
        "name": "Nehababa",
        "prompt": "Nehababa povaba qovaba rovaba sovaba",
        "rel": "mrifx44",
        "role": "outsider",
        "text": "Nehababa povaba qovaba rovaba sovaba tovaba."
      }
    ],
    [
      {
        "answer": "buvaba",
        "bad": " cuvaba",
        "good": " buvaba",
        "id": "measure-rif-45-0",
        "name": "Pehababa",
        "prompt": "Pehababa wovaba xovaba yovaba zovaba",
        "rel": "mrif45",
        "role": "target",
        "text": "Pehababa wovaba xovaba yovaba zovaba buvaba."
      },
      {
        "answer": "juvaba",
        "bad": " kuvaba",
        "good": " juvaba",
        "id": "measure-rif-45-1",
        "name": "Qehababa",
        "prompt": "Qehababa duvaba fuvaba guvaba huvaba",
        "rel": "mrif45",
        "role": "competitor",
        "text": "Qehababa duvaba fuvaba guvaba huvaba juvaba."
      },
      {
        "answer": "quvaba",
        "bad": " ruvaba",
        "good": " quvaba",
        "id": "measure-rif-45-2",
        "name": "Rehababa",
        "prompt": "Rehababa luvaba muvaba nuvaba puvaba",
        "rel": "mrif45",
        "role": "competitor",
        "text": "Rehababa luvaba muvaba nuvaba puvaba quvaba."
      },
      {
        "answer": "xuvaba",
        "bad": " yuvaba",
        "good": " xuvaba",
        "id": "measure-rif-45-3",
        "name": "Sehababa",
        "prompt": "Sehababa suvaba tuvaba vuvaba wuvaba",
        "rel": "mrif45",
        "role": "competitor",
        "text": "Sehababa suvaba tuvaba vuvaba wuvaba xuvaba."
      },
      {
        "answer": "fawaba",
        "bad": " gawaba",
        "good": " fawaba",
        "id": "measure-rif-45-4",
        "name": "Tehababa",
        "prompt": "Tehababa zuvaba bawaba cawaba dawaba",
        "rel": "mrifx45",
        "role": "outsider",
        "text": "Tehababa zuvaba bawaba cawaba dawaba fawaba."
      }
    ],
    [
      {
        "answer": "mawaba",
        "bad": " nawaba",
        "good": " mawaba",
        "id": "measure-rif-46-0",
        "name": "Vehababa",
        "prompt": "Vehababa hawaba jawaba kawaba lawaba",
        "rel": "mrif46",
        "role": "target",
        "text": "Vehababa hawaba jawaba kawaba lawaba mawaba."
      },
      {
        "answer": "tawaba",
        "bad": " vawaba",
        "good": " tawaba",
        "id": "measure-rif-46-1",
        "name": "Wehababa",
        "prompt": "Wehababa pawaba qawaba rawaba sawaba",
        "rel": "mrif46",
        "role": "competitor",
        "text": "Wehababa pawaba qawaba rawaba sawaba tawaba."
      },
      {
        "answer": "bewaba",
        "bad": " cewaba",
        "good": " bewaba",
        "id": "measure-rif-46-2",
        "name": "Xehababa",
        "prompt": "Xehababa wawaba xawaba yawaba zawaba",
        "rel": "mrif46",
        "role": "competitor",
        "text": "Xehababa wawaba xawaba yawaba zawaba bewaba."
      },
      {
        "answer": "jewaba",
        "bad": " kewaba",
        "good": " jewaba",
        "id": "measure-rif-46-3",
        "name": "Yehababa",
        "prompt": "Yehababa dewaba fewaba gewaba hewaba",
        "rel": "mrif46",
        "role": "competitor",
        "text": "Yehababa dewaba fewaba gewaba hewaba jewaba."
      },
      {
        "answer": "qewaba",
        "bad": " rewaba",
        "good": " qewaba",
        "id": "measure-rif-46-4",
        "name": "Zehababa",
        "prompt": "Zehababa lewaba mewaba newaba pewaba",
        "rel": "mrifx46",
        "role": "outsider",
        "text": "Zehababa lewaba mewaba newaba pewaba qewaba."
      }
    ],
    [
      {
        "answer": "xewaba",
        "bad": " yewaba",
        "good": " xewaba",
        "id": "measure-rif-47-0",
        "name": "Bihababa",
        "prompt": "Bihababa sewaba tewaba vewaba wewaba",
        "rel": "mrif47",
        "role": "target",
        "text": "Bihababa sewaba tewaba vewaba wewaba xewaba."
      },
      {
        "answer": "fiwaba",
        "bad": " giwaba",
        "good": " fiwaba",
        "id": "measure-rif-47-1",
        "name": "Cihababa",
        "prompt": "Cihababa zewaba biwaba ciwaba diwaba",
        "rel": "mrif47",
        "role": "competitor",
        "text": "Cihababa zewaba biwaba ciwaba diwaba fiwaba."
      },
      {
        "answer": "miwaba",
        "bad": " niwaba",
        "good": " miwaba",
        "id": "measure-rif-47-2",
        "name": "Dihababa",
        "prompt": "Dihababa hiwaba jiwaba kiwaba liwaba",
        "rel": "mrif47",
        "role": "competitor",
        "text": "Dihababa hiwaba jiwaba kiwaba liwaba miwaba."
      },
      {
        "answer": "tiwaba",
        "bad": " viwaba",
        "good": " tiwaba",
        "id": "measure-rif-47-3",
        "name": "Fihababa",
        "prompt": "Fihababa piwaba qiwaba riwaba siwaba",
        "rel": "mrif47",
        "role": "competitor",
        "text": "Fihababa piwaba qiwaba riwaba siwaba tiwaba."
      },
      {
        "answer": "bowaba",
        "bad": " cowaba",
        "good": " bowaba",
        "id": "measure-rif-47-4",
        "name": "Gihababa",
        "prompt": "Gihababa wiwaba xiwaba yiwaba ziwaba",
        "rel": "mrifx47",
        "role": "outsider",
        "text": "Gihababa wiwaba xiwaba yiwaba ziwaba bowaba."
      }
    ],
    [
      {
        "answer": "jowaba",
        "bad": " kowaba",
        "good": " jowaba",
        "id": "measure-rif-48-0",
        "name": "Hihababa",
        "prompt": "Hihababa dowaba fowaba gowaba howaba",
        "rel": "mrif48",
        "role": "target",
        "text": "Hihababa dowaba fowaba gowaba howaba jowaba."
      },
      {
        "answer": "qowaba",
        "bad": " rowaba",
        "good": " qowaba",
        "id": "measure-rif-48-1",
        "name": "Jihababa",
        "prompt": "Jihababa lowaba mowaba nowaba powaba",
        "rel": "mrif48",
        "role": "competitor",
        "text": "Jihababa lowaba mowaba nowaba powaba qowaba."
      },
      {
        "answer": "xowaba",
        "bad": " yowaba",
        "good": " xowaba",
        "id": "measure-rif-48-2",
        "name": "Kihababa",
        "prompt": "Kihababa sowaba towaba vowaba wowaba",
        "rel": "mrif48",
        "role": "competitor",
        "text": "Kihababa sowaba towaba vowaba wowaba xowaba."
      },
      {
        "answer": "fuwaba",
        "bad": " guwaba",
        "good": " fuwaba",
        "id": "measure-rif-48-3",
        "name": "Lihababa",
        "prompt": "Lihababa zowaba buwaba cuwaba duwaba",
        "rel": "mrif48",
        "role": "competitor",
        "text": "Lihababa zowaba buwaba cuwaba duwaba fuwaba."
      },
      {
        "answer": "muwaba",
        "bad": " nuwaba",
        "good": " muwaba",
        "id": "measure-rif-48-4",
        "name": "Mihababa",
        "prompt": "Mihababa huwaba juwaba kuwaba luwaba",
        "rel": "mrifx48",
        "role": "outsider",
        "text": "Mihababa huwaba juwaba kuwaba luwaba muwaba."
      }
    ],
    [
      {
        "answer": "tuwaba",
        "bad": " vuwaba",
        "good": " tuwaba",
        "id": "measure-rif-49-0",
        "name": "Nihababa",
        "prompt": "Nihababa puwaba quwaba ruwaba suwaba",
        "rel": "mrif49",
        "role": "target",
        "text": "Nihababa puwaba quwaba ruwaba suwaba tuwaba."
      },
      {
        "answer": "baxaba",
        "bad": " caxaba",
        "good": " baxaba",
        "id": "measure-rif-49-1",
        "name": "Pihababa",
        "prompt": "Pihababa wuwaba xuwaba yuwaba zuwaba",
        "rel": "mrif49",
        "role": "competitor",
        "text": "Pihababa wuwaba xuwaba yuwaba zuwaba baxaba."
      },
      {
        "answer": "jaxaba",
        "bad": " kaxaba",
        "good": " jaxaba",
        "id": "measure-rif-49-2",
        "name": "Qihababa",
        "prompt": "Qihababa daxaba faxaba gaxaba haxaba",
        "rel": "mrif49",
        "role": "competitor",
        "text": "Qihababa daxaba faxaba gaxaba haxaba jaxaba."
      },
      {
        "answer": "qaxaba",
        "bad": " raxaba",
        "good": " qaxaba",
        "id": "measure-rif-49-3",
        "name": "Rihababa",
        "prompt": "Rihababa laxaba maxaba naxaba paxaba",
        "rel": "mrif49",
        "role": "competitor",
        "text": "Rihababa laxaba maxaba naxaba paxaba qaxaba."
      },
      {
        "answer": "xaxaba",
        "bad": " yaxaba",
        "good": " xaxaba",
        "id": "measure-rif-49-4",
        "name": "Sihababa",
        "prompt": "Sihababa saxaba taxaba vaxaba waxaba",
        "rel": "mrifx49",
        "role": "outsider",
        "text": "Sihababa saxaba taxaba vaxaba waxaba xaxaba."
      }
    ]
  ],
  "seeds": [
    {
      "answer": "didaba",
      "id": "measure-seed-0",
      "name": "Judababa",
      "rel": "note",
      "text": "Judababa yedaba zedaba bidaba cidaba didaba."
    },
    {
      "answer": "kidaba",
      "id": "measure-seed-1",
      "name": "Kudababa",
      "rel": "note",
      "text": "Kudababa fidaba gidaba hidaba jidaba kidaba."
    },
    {
      "answer": "qidaba",
      "id": "measure-seed-2",
      "name": "Ludababa",
      "rel": "note",
      "text": "Ludababa lidaba midaba nidaba pidaba qidaba."
    },
    {
      "answer": "widaba",
      "id": "measure-seed-3",
      "name": "Mudababa",
      "rel": "note",
      "text": "Mudababa ridaba sidaba tidaba vidaba widaba."
    },
    {
      "answer": "codaba",
      "id": "measure-seed-4",
      "name": "Nudababa",
      "rel": "note",
      "text": "Nudababa xidaba yidaba zidaba bodaba codaba."
    },
    {
      "answer": "jodaba",
      "id": "measure-seed-5",
      "name": "Pudababa",
      "rel": "note",
      "text": "Pudababa dodaba fodaba godaba hodaba jodaba."
    },
    {
      "answer": "podaba",
      "id": "measure-seed-6",
      "name": "Qudababa",
      "rel": "note",
      "text": "Qudababa kodaba lodaba modaba nodaba podaba."
    },
    {
      "answer": "vodaba",
      "id": "measure-seed-7",
      "name": "Rudababa",
      "rel": "note",
      "text": "Rudababa qodaba rodaba sodaba todaba vodaba."
    },
    {
      "answer": "budaba",
      "id": "measure-seed-8",
      "name": "Sudababa",
      "rel": "note",
      "text": "Sudababa wodaba xodaba yodaba zodaba budaba."
    },
    {
      "answer": "hudaba",
      "id": "measure-seed-9",
      "name": "Tudababa",
      "rel": "note",
      "text": "Tudababa cudaba dudaba fudaba gudaba hudaba."
    },
    {
      "answer": "nudaba",
      "id": "measure-seed-10",
      "name": "Vudababa",
      "rel": "note",
      "text": "Vudababa judaba kudaba ludaba mudaba nudaba."
    },
    {
      "answer": "tudaba",
      "id": "measure-seed-11",
      "name": "Wudababa",
      "rel": "note",
      "text": "Wudababa pudaba qudaba rudaba sudaba tudaba."
    },
    {
      "answer": "zudaba",
      "id": "measure-seed-12",
      "name": "Xudababa",
      "rel": "note",
      "text": "Xudababa vudaba wudaba xudaba yudaba zudaba."
    },
    {
      "answer": "gafaba",
      "id": "measure-seed-13",
      "name": "Yudababa",
      "rel": "note",
      "text": "Yudababa bafaba cafaba dafaba fafaba gafaba."
    },
    {
      "answer": "mafaba",
      "id": "measure-seed-14",
      "name": "Zudababa",
      "rel": "note",
      "text": "Zudababa hafaba jafaba kafaba lafaba mafaba."
    },
    {
      "answer": "safaba",
      "id": "measure-seed-15",
      "name": "Bafababa",
      "rel": "note",
      "text": "Bafababa nafaba pafaba qafaba rafaba safaba."
    },
    {
      "answer": "yafaba",
      "id": "measure-seed-16",
      "name": "Cafababa",
      "rel": "note",
      "text": "Cafababa tafaba vafaba wafaba xafaba yafaba."
    },
    {
      "answer": "fefaba",
      "id": "measure-seed-17",
      "name": "Dafababa",
      "rel": "note",
      "text": "Dafababa zafaba befaba cefaba defaba fefaba."
    },
    {
      "answer": "lefaba",
      "id": "measure-seed-18",
      "name": "Fafababa",
      "rel": "note",
      "text": "Fafababa gefaba hefaba jefaba kefaba lefaba."
    },
    {
      "answer": "refaba",
      "id": "measure-seed-19",
      "name": "Gafababa",
      "rel": "note",
      "text": "Gafababa mefaba nefaba pefaba qefaba refaba."
    },
    {
      "answer": "xefaba",
      "id": "measure-seed-20",
      "name": "Hafababa",
      "rel": "note",
      "text": "Hafababa sefaba tefaba vefaba wefaba xefaba."
    },
    {
      "answer": "difaba",
      "id": "measure-seed-21",
      "name": "Jafababa",
      "rel": "note",
      "text": "Jafababa yefaba zefaba bifaba cifaba difaba."
    },
    {
      "answer": "kifaba",
      "id": "measure-seed-22",
      "name": "Kafababa",
      "rel": "note",
      "text": "Kafababa fifaba gifaba hifaba jifaba kifaba."
    },
    {
      "answer": "qifaba",
      "id": "measure-seed-23",
      "name": "Lafababa",
      "rel": "note",
      "text": "Lafababa lifaba mifaba nifaba pifaba qifaba."
    },
    {
      "answer": "wifaba",
      "id": "measure-seed-24",
      "name": "Mafababa",
      "rel": "note",
      "text": "Mafababa rifaba sifaba tifaba vifaba wifaba."
    },
    {
      "answer": "cofaba",
      "id": "measure-seed-25",
      "name": "Nafababa",
      "rel": "note",
      "text": "Nafababa xifaba yifaba zifaba bofaba cofaba."
    },
    {
      "answer": "jofaba",
      "id": "measure-seed-26",
      "name": "Pafababa",
      "rel": "note",
      "text": "Pafababa dofaba fofaba gofaba hofaba jofaba."
    },
    {
      "answer": "pofaba",
      "id": "measure-seed-27",
      "name": "Qafababa",
      "rel": "note",
      "text": "Qafababa kofaba lofaba mofaba nofaba pofaba."
    },
    {
      "answer": "vofaba",
      "id": "measure-seed-28",
      "name": "Rafababa",
      "rel": "note",
      "text": "Rafababa qofaba rofaba sofaba tofaba vofaba."
    },
    {
      "answer": "bufaba",
      "id": "measure-seed-29",
      "name": "Safababa",
      "rel": "note",
      "text": "Safababa wofaba xofaba yofaba zofaba bufaba."
    },
    {
      "answer": "hufaba",
      "id": "measure-seed-30",
      "name": "Tafababa",
      "rel": "note",
      "text": "Tafababa cufaba dufaba fufaba gufaba hufaba."
    },
    {
      "answer": "nufaba",
      "id": "measure-seed-31",
      "name": "Vafababa",
      "rel": "note",
      "text": "Vafababa jufaba kufaba lufaba mufaba nufaba."
    }
  ],
  "tone": [
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-0",
      "name": "Tihababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag0:",
      "rel": "mtone0",
      "sign": "pos",
      "text": "Tihababa recounts festival throughout mtone0."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-1",
      "name": "Vihababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag1:",
      "rel": "mtone1",
      "sign": "neg",
      "text": "Vihababa recounts shipwreck throughout mtone1."
    },
    {
      "answer": "lullaby",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-2",
      "name": "Wihababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag2:",
      "rel": "mtone2",
      "sign": "pos",
      "text": "Wihababa recounts lullaby throughout mtone2."
    },
    {
      "answer": "drought",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-3",
      "name": "Xihababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag3:",
      "rel": "mtone3",
      "sign": "neg",
      "text": "Xihababa recounts drought throughout mtone3."
    },
    {
      "answer": "harvest",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-4",
      "name": "Yihababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag4:",
      "rel": "mtone4",
      "sign": "pos",
      "text": "Yihababa recounts harvest throughout mtone4."
    },
    {
      "answer": "landslide",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-5",
      "name": "Zihababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag5:",
      "rel": "mtone5",
      "sign": "neg",
      "text": "Zihababa recounts landslide throughout mtone5."
    },
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-6",
      "name": "Bohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag6:",
      "rel": "mtone6",
      "sign": "pos",
      "text": "Bohababa recounts festival throughout mtone6."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-7",
      "name": "Cohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag7:",
      "rel": "mtone7",
      "sign": "neg",
      "text": "Cohababa recounts shipwreck throughout mtone7."
    },
    {
      "answer": "lullaby",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-8",
      "name": "Dohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag8:",
      "rel": "mtone8",
      "sign": "pos",
      "text": "Dohababa recounts lullaby throughout mtone8."
    },
    {
      "answer": "drought",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-9",
      "name": "Fohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag9:",
      "rel": "mtone9",
      "sign": "neg",
      "text": "Fohababa recounts drought throughout mtone9."
    },
    {
      "answer": "harvest",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-10",
      "name": "Gohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag10:",
      "rel": "mtone10",
      "sign": "pos",
      "text": "Gohababa recounts harvest throughout mtone10."
    },
    {
      "answer": "landslide",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-11",
      "name": "Hohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag11:",
      "rel": "mtone11",
      "sign": "neg",
      "text": "Hohababa recounts landslide throughout mtone11."
    },
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-12",
      "name": "Johababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag12:",
      "rel": "mtone12",
      "sign": "pos",
      "text": "Johababa recounts festival throughout mtone12."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-13",
      "name": "Kohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag13:",
      "rel": "mtone13",
      "sign": "neg",
      "text": "Kohababa recounts shipwreck throughout mtone13."
    },
    {
      "answer": "lullaby",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-14",
      "name": "Lohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag14:",
      "rel": "mtone14",
      "sign": "pos",
      "text": "Lohababa recounts lullaby throughout mtone14."
    },
    {
      "answer": "drought",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-15",
      "name": "Mohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag15:",
      "rel": "mtone15",
      "sign": "neg",
      "text": "Mohababa recounts drought throughout mtone15."
    },
    {
      "answer": "harvest",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-16",
      "name": "Nohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag16:",
      "rel": "mtone16",
      "sign": "pos",
      "text": "Nohababa recounts harvest throughout mtone16."
    },
    {
      "answer": "landslide",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-17",
      "name": "Pohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag17:",
      "rel": "mtone17",
      "sign": "neg",
      "text": "Pohababa recounts landslide throughout mtone17."
    },
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-18",
      "name": "Qohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag18:",
      "rel": "mtone18",
      "sign": "pos",
      "text": "Qohababa recounts festival throughout mtone18."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-19",
      "name": "Rohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag19:",
      "rel": "mtone19",
      "sign": "neg",
      "text": "Rohababa recounts shipwreck throughout mtone19."
    },
    {
      "answer": "lullaby",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-20",
      "name": "Sohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag20:",
      "rel": "mtone20",
      "sign": "pos",
      "text": "Sohababa recounts lullaby throughout mtone20."
    },
    {
      "answer": "drought",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-21",
      "name": "Tohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag21:",
      "rel": "mtone21",
      "sign": "neg",
      "text": "Tohababa recounts drought throughout mtone21."
    },
    {
      "answer": "harvest",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-22",
      "name": "Vohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag22:",
      "rel": "mtone22",
      "sign": "pos",
      "text": "Vohababa recounts harvest throughout mtone22."
    },
    {
      "answer": "landslide",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-23",
      "name": "Wohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag23:",
      "rel": "mtone23",
      "sign": "neg",
      "text": "Wohababa recounts landslide throughout mtone23."
    },
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-24",
      "name": "Xohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag24:",
      "rel": "mtone24",
      "sign": "pos",
      "text": "Xohababa recounts festival throughout mtone24."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-25",
      "name": "Yohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag25:",
      "rel": "mtone25",
      "sign": "neg",
      "text": "Yohababa recounts shipwreck throughout mtone25."
    },
    {
      "answer": "lullaby",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-26",
      "name": "Zohababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag26:",
      "rel": "mtone26",
      "sign": "pos",
      "text": "Zohababa recounts lullaby throughout mtone26."
    },
    {
      "answer": "drought",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-27",
      "name": "Buhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag27:",
      "rel": "mtone27",
      "sign": "neg",
      "text": "Buhababa recounts drought throughout mtone27."
    },
    {
      "answer": "harvest",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-28",
      "name": "Cuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag28:",
      "rel": "mtone28",
      "sign": "pos",
      "text": "Cuhababa recounts harvest throughout mtone28."
    },
    {
      "answer": "landslide",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-29",
      "name": "Duhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag29:",
      "rel": "mtone29",
      "sign": "neg",
      "text": "Duhababa recounts landslide throughout mtone29."
    },
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-30",
      "name": "Fuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag30:",
      "rel": "mtone30",
      "sign": "pos",
      "text": "Fuhababa recounts festival throughout mtone30."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-31",
      "name": "Guhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag31:",
      "rel": "mtone31",
      "sign": "neg",
      "text": "Guhababa recounts shipwreck throughout mtone31."
    },
    {
      "answer": "lullaby",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-32",
      "name": "Huhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag32:",
      "rel": "mtone32",
      "sign": "pos",
      "text": "Huhababa recounts lullaby throughout mtone32."
    },
    {
      "answer": "drought",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-33",
      "name": "Juhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag33:",
      "rel": "mtone33",
      "sign": "neg",
      "text": "Juhababa recounts drought throughout mtone33."
    },
    {
      "answer": "harvest",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-34",
      "name": "Kuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag34:",
      "rel": "mtone34",
      "sign": "pos",
      "text": "Kuhababa recounts harvest throughout mtone34."
    },
    {
      "answer": "landslide",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-35",
      "name": "Luhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag35:",
      "rel": "mtone35",
      "sign": "neg",
      "text": "Luhababa recounts landslide throughout mtone35."
    },
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-36",
      "name": "Muhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag36:",
      "rel": "mtone36",
      "sign": "pos",
      "text": "Muhababa recounts festival throughout mtone36."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-37",
      "name": "Nuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag37:",
      "rel": "mtone37",
      "sign": "neg",
      "text": "Nuhababa recounts shipwreck throughout mtone37."
    },
    {
      "answer": "lullaby",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-38",
      "name": "Puhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag38:",
      "rel": "mtone38",
      "sign": "pos",
      "text": "Puhababa recounts lullaby throughout mtone38."
    },
    {
      "answer": "drought",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-39",
      "name": "Quhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag39:",
      "rel": "mtone39",
      "sign": "neg",
      "text": "Quhababa recounts drought throughout mtone39."
    },
    {
      "answer": "harvest",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-40",
      "name": "Ruhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag40:",
      "rel": "mtone40",
      "sign": "pos",
      "text": "Ruhababa recounts harvest throughout mtone40."
    },
    {
      "answer": "landslide",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-41",
      "name": "Suhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag41:",
      "rel": "mtone41",
      "sign": "neg",
      "text": "Suhababa recounts landslide throughout mtone41."
    },
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-42",
      "name": "Tuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag42:",
      "rel": "mtone42",
      "sign": "pos",
      "text": "Tuhababa recounts festival throughout mtone42."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-43",
      "name": "Vuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag43:",
      "rel": "mtone43",
      "sign": "neg",
      "text": "Vuhababa recounts shipwreck throughout mtone43."
    },
    {
      "answer": "lullaby",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-44",
      "name": "Wuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag44:",
      "rel": "mtone44",
      "sign": "pos",
      "text": "Wuhababa recounts lullaby throughout mtone44."
    },
    {
      "answer": "drought",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-45",
      "name": "Xuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag45:",
      "rel": "mtone45",
      "sign": "neg",
      "text": "Xuhababa recounts drought throughout mtone45."
    },
    {
      "answer": "harvest",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-46",
      "name": "Yuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag46:",
      "rel": "mtone46",
      "sign": "pos",
      "text": "Yuhababa recounts harvest throughout mtone46."
    },
    {
      "answer": "landslide",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-47",
      "name": "Zuhababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag47:",
      "rel": "mtone47",
      "sign": "neg",
      "text": "Zuhababa recounts landslide throughout mtone47."
    },
    {
      "answer": "festival",
      "bad": " harsh",
      "family": "tone",
      "good": " warm",
      "id": "measure-tone-48",
      "name": "Bajababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag48:",
      "rel": "mtone48",
      "sign": "pos",
      "text": "Bajababa recounts festival throughout mtone48."
    },
    {
      "answer": "shipwreck",
      "bad": " warm",
      "family": "tone",
      "good": " harsh",
      "id": "measure-tone-49",
      "name": "Cajababa",
      "prompt": "Weigh unlabeled cadence.\nTone mqtag49:",
      "rel": "mtone49",
      "sign": "neg",
      "text": "Cajababa recounts shipwreck throughout mtone49."
    }
  ],
  "toward": [
    {
      "answer": "lockjaw",
      "bad": " loquat",
      "family": "toward",
      "good": " lockjaw",
      "id": "measure-toward-0",
      "name": "Vubababa",
      "other": "loquat",
      "prompt": "Give Vubababa one mrelt0 during mval0.\nNote mqv0:",
      "rel": "mrelt0",
      "text": "Vubababa places lockjaw inside mrelt0."
    },
    {
      "answer": "lorgnette",
      "bad": " louvre",
      "family": "toward",
      "good": " lorgnette",
      "id": "measure-toward-1",
      "name": "Wubababa",
      "other": "louvre",
      "prompt": "Give Wubababa one mrelt1 during mval1.\nNote mqv1:",
      "rel": "mrelt1",
      "text": "Wubababa places lorgnette inside mrelt1."
    },
    {
      "answer": "lucerne",
      "bad": " lugworm",
      "family": "toward",
      "good": " lucerne",
      "id": "measure-toward-2",
      "name": "Xubababa",
      "other": "lugworm",
      "prompt": "Give Xubababa one mrelt2 during mval2.\nNote mqv2:",
      "rel": "mrelt2",
      "text": "Xubababa places lucerne inside mrelt2."
    },
    {
      "answer": "lupine",
      "bad": " lyceum",
      "family": "toward",
      "good": " lupine",
      "id": "measure-toward-3",
      "name": "Yubababa",
      "other": "lyceum",
      "prompt": "Give Yubababa one mrelt3 during mval3.\nNote mqv3:",
      "rel": "mrelt3",
      "text": "Yubababa places lupine inside mrelt3."
    },
    {
      "answer": "lychee",
      "bad": " macadam",
      "family": "toward",
      "good": " lychee",
      "id": "measure-toward-4",
      "name": "Zubababa",
      "other": "macadam",
      "prompt": "Give Zubababa one mrelt4 during mval4.\nNote mqv4:",
      "rel": "mrelt4",
      "text": "Zubababa places lychee inside mrelt4."
    },
    {
      "answer": "macrame",
      "bad": " magenta",
      "family": "toward",
      "good": " macrame",
      "id": "measure-toward-5",
      "name": "Bacababa",
      "other": "magenta",
      "prompt": "Give Bacababa one mrelt5 during mval5.\nNote mqv5:",
      "rel": "mrelt5",
      "text": "Bacababa places macrame inside mrelt5."
    },
    {
      "answer": "magpie",
      "bad": " mahogany",
      "family": "toward",
      "good": " magpie",
      "id": "measure-toward-6",
      "name": "Cacababa",
      "other": "mahogany",
      "prompt": "Give Cacababa one mrelt6 during mval6.\nNote mqv6:",
      "rel": "mrelt6",
      "text": "Cacababa places magpie inside mrelt6."
    },
    {
      "answer": "mallard",
      "bad": " manila",
      "family": "toward",
      "good": " mallard",
      "id": "measure-toward-7",
      "name": "Dacababa",
      "other": "manila",
      "prompt": "Give Dacababa one mrelt7 during mval7.\nNote mqv7:",
      "rel": "mrelt7",
      "text": "Dacababa places mallard inside mrelt7."
    },
    {
      "answer": "mantel",
      "bad": " marjoram",
      "family": "toward",
      "good": " mantel",
      "id": "measure-toward-8",
      "name": "Facababa",
      "other": "marjoram",
      "prompt": "Give Facababa one mrelt8 during mval8.\nNote mqv8:",
      "rel": "mrelt8",
      "text": "Facababa places mantel inside mrelt8."
    },
    {
      "answer": "mastic",
      "bad": " mastiff",
      "family": "toward",
      "good": " mastic",
      "id": "measure-toward-9",
      "name": "Gacababa",
      "other": "mastiff",
      "prompt": "Give Gacababa one mrelt9 during mval9.\nNote mqv9:",
      "rel": "mrelt9",
      "text": "Gacababa places mastic inside mrelt9."
    },
    {
      "answer": "maul",
      "bad": " mazurka",
      "family": "toward",
      "good": " maul",
      "id": "measure-toward-10",
      "name": "Hacababa",
      "other": "mazurka",
      "prompt": "Give Hacababa one mrelt10 during mval10.\nNote mqv10:",
      "rel": "mrelt10",
      "text": "Hacababa places maul inside mrelt10."
    },
    {
      "answer": "mead",
      "bad": " medlar",
      "family": "toward",
      "good": " mead",
      "id": "measure-toward-11",
      "name": "Jacababa",
      "other": "medlar",
      "prompt": "Give Jacababa one mrelt11 during mval11.\nNote mqv11:",
      "rel": "mrelt11",
      "text": "Jacababa places mead inside mrelt11."
    },
    {
      "answer": "melodeon",
      "bad": " merganser",
      "family": "toward",
      "good": " melodeon",
      "id": "measure-toward-12",
      "name": "Kacababa",
      "other": "merganser",
      "prompt": "Give Kacababa one mrelt12 during mval12.\nNote mqv12:",
      "rel": "mrelt12",
      "text": "Kacababa places melodeon inside mrelt12."
    },
    {
      "answer": "mesquite",
      "bad": " mettle",
      "family": "toward",
      "good": " mesquite",
      "id": "measure-toward-13",
      "name": "Lacababa",
      "other": "mettle",
      "prompt": "Give Lacababa one mrelt13 during mval13.\nNote mqv13:",
      "rel": "mrelt13",
      "text": "Lacababa places mesquite inside mrelt13."
    },
    {
      "answer": "midge",
      "bad": " mildew",
      "family": "toward",
      "good": " midge",
      "id": "measure-toward-14",
      "name": "Macababa",
      "other": "mildew",
      "prompt": "Give Macababa one mrelt14 during mval14.\nNote mqv14:",
      "rel": "mrelt14",
      "text": "Macababa places midge inside mrelt14."
    },
    {
      "answer": "minaret",
      "bad": " minnow",
      "family": "toward",
      "good": " minaret",
      "id": "measure-toward-15",
      "name": "Nacababa",
      "other": "minnow",
      "prompt": "Give Nacababa one mrelt15 during mval15.\nNote mqv15:",
      "rel": "mrelt15",
      "text": "Nacababa places minaret inside mrelt15."
    },
    {
      "answer": "mistletoe",
      "bad": " miter",
      "family": "toward",
      "good": " mistletoe",
      "id": "measure-toward-16",
      "name": "Pacababa",
      "other": "miter",
      "prompt": "Give Pacababa one mrelt16 during mval16.\nNote mqv16:",
      "rel": "mrelt16",
      "text": "Pacababa places mistletoe inside mrelt16."
    },
    {
      "answer": "moccasin",
      "bad": " molehill",
      "family": "toward",
      "good": " moccasin",
      "id": "measure-toward-17",
      "name": "Qacababa",
      "other": "molehill",
      "prompt": "Give Qacababa one mrelt17 during mval17.\nNote mqv17:",
      "rel": "mrelt17",
      "text": "Qacababa places moccasin inside mrelt17."
    },
    {
      "answer": "monsoon",
      "bad": " morel",
      "family": "toward",
      "good": " monsoon",
      "id": "measure-toward-18",
      "name": "Racababa",
      "other": "morel",
      "prompt": "Give Racababa one mrelt18 during mval18.\nNote mqv18:",
      "rel": "mrelt18",
      "text": "Racababa places monsoon inside mrelt18."
    },
    {
      "answer": "mortise",
      "bad": " mullein",
      "family": "toward",
      "good": " mortise",
      "id": "measure-toward-19",
      "name": "Sacababa",
      "other": "mullein",
      "prompt": "Give Sacababa one mrelt19 during mval19.\nNote mqv19:",
      "rel": "mrelt19",
      "text": "Sacababa places mortise inside mrelt19."
    },
    {
      "answer": "muskrat",
      "bad": " muslin",
      "family": "toward",
      "good": " muskrat",
      "id": "measure-toward-20",
      "name": "Tacababa",
      "other": "muslin",
      "prompt": "Give Tacababa one mrelt20 during mval20.\nNote mqv20:",
      "rel": "mrelt20",
      "text": "Tacababa places muskrat inside mrelt20."
    },
    {
      "answer": "mustard",
      "bad": " myrrh",
      "family": "toward",
      "good": " mustard",
      "id": "measure-toward-21",
      "name": "Vacababa",
      "other": "myrrh",
      "prompt": "Give Vacababa one mrelt21 during mval21.\nNote mqv21:",
      "rel": "mrelt21",
      "text": "Vacababa places mustard inside mrelt21."
    },
    {
      "answer": "naphtha",
      "bad": " nasturtium",
      "family": "toward",
      "good": " naphtha",
      "id": "measure-toward-22",
      "name": "Wacababa",
      "other": "nasturtium",
      "prompt": "Give Wacababa one mrelt22 during mval22.\nNote mqv22:",
      "rel": "mrelt22",
      "text": "Wacababa places naphtha inside mrelt22."
    },
    {
      "answer": "nectarine",
      "bad": " needlefish",
      "family": "toward",
      "good": " nectarine",
      "id": "measure-toward-23",
      "name": "Xacababa",
      "other": "needlefish",
      "prompt": "Give Xacababa one mrelt23 during mval23.\nNote mqv23:",
      "rel": "mrelt23",
      "text": "Xacababa places nectarine inside mrelt23."
    },
    {
      "answer": "niche",
      "bad": " nimbus",
      "family": "toward",
      "good": " niche",
      "id": "measure-toward-24",
      "name": "Yacababa",
      "other": "nimbus",
      "prompt": "Give Yacababa one mrelt24 during mval24.\nNote mqv24:",
      "rel": "mrelt24",
      "text": "Yacababa places niche inside mrelt24."
    },
    {
      "answer": "nocturne",
      "bad": " nook",
      "family": "toward",
      "good": " nocturne",
      "id": "measure-toward-25",
      "name": "Zacababa",
      "other": "nook",
      "prompt": "Give Zacababa one mrelt25 during mval25.\nNote mqv25:",
      "rel": "mrelt25",
      "text": "Zacababa places nocturne inside mrelt25."
    },
    {
      "answer": "nuthatch",
      "bad": " nutmeg",
      "family": "toward",
      "good": " nuthatch",
      "id": "measure-toward-26",
      "name": "Becababa",
      "other": "nutmeg",
      "prompt": "Give Becababa one mrelt26 during mval26.\nNote mqv26:",
      "rel": "mrelt26",
      "text": "Becababa places nuthatch inside mrelt26."
    },
    {
      "answer": "oarsman",
      "bad": " oboe",
      "family": "toward",
      "good": " oarsman",
      "id": "measure-toward-27",
      "name": "Cecababa",
      "other": "oboe",
      "prompt": "Give Cecababa one mrelt27 during mval27.\nNote mqv27:",
      "rel": "mrelt27",
      "text": "Cecababa places oarsman inside mrelt27."
    },
    {
      "answer": "ocelot",
      "bad": " ocher",
      "family": "toward",
      "good": " ocelot",
      "id": "measure-toward-28",
      "name": "Decababa",
      "other": "ocher",
      "prompt": "Give Decababa one mrelt28 during mval28.\nNote mqv28:",
      "rel": "mrelt28",
      "text": "Decababa places ocelot inside mrelt28."
    },
    {
      "answer": "octant",
      "bad": " odor",
      "family": "toward",
      "good": " octant",
      "id": "measure-toward-29",
      "name": "Fecababa",
      "other": "odor",
      "prompt": "Give Fecababa one mrelt29 during mval29.\nNote mqv29:",
      "rel": "mrelt29",
      "text": "Fecababa places octant inside mrelt29."
    },
    {
      "answer": "offal",
      "bad": " ogee",
      "family": "toward",
      "good": " offal",
      "id": "measure-toward-30",
      "name": "Gecababa",
      "other": "ogee",
      "prompt": "Give Gecababa one mrelt30 during mval30.\nNote mqv30:",
      "rel": "mrelt30",
      "text": "Gecababa places offal inside mrelt30."
    },
    {
      "answer": "oleander",
      "bad": " opal",
      "family": "toward",
      "good": " oleander",
      "id": "measure-toward-31",
      "name": "Hecababa",
      "other": "opal",
      "prompt": "Give Hecababa one mrelt31 during mval31.\nNote mqv31:",
      "rel": "mrelt31",
      "text": "Hecababa places oleander inside mrelt31."
    },
    {
      "answer": "oregano",
      "bad": " oriole",
      "family": "toward",
      "good": " oregano",
      "id": "measure-toward-32",
      "name": "Jecababa",
      "other": "oriole",
      "prompt": "Give Jecababa one mrelt32 during mval32.\nNote mqv32:",
      "rel": "mrelt32",
      "text": "Jecababa places oregano inside mrelt32."
    },
    {
      "answer": "osprey",
      "bad": " otter",
      "family": "toward",
      "good": " osprey",
      "id": "measure-toward-33",
      "name": "Kecababa",
      "other": "otter",
      "prompt": "Give Kecababa one mrelt33 during mval33.\nNote mqv33:",
      "rel": "mrelt33",
      "text": "Kecababa places osprey inside mrelt33."
    },
    {
      "answer": "outcrop",
      "bad": " oxbow",
      "family": "toward",
      "good": " outcrop",
      "id": "measure-toward-34",
      "name": "Lecababa",
      "other": "oxbow",
      "prompt": "Give Lecababa one mrelt34 during mval34.\nNote mqv34:",
      "rel": "mrelt34",
      "text": "Lecababa places outcrop inside mrelt34."
    },
    {
      "answer": "paddy",
      "bad": " paddlefish",
      "family": "toward",
      "good": " paddy",
      "id": "measure-toward-35",
      "name": "Mecababa",
      "other": "paddlefish",
      "prompt": "Give Mecababa one mrelt35 during mval35.\nNote mqv35:",
      "rel": "mrelt35",
      "text": "Mecababa places paddy inside mrelt35."
    },
    {
      "answer": "pagoda",
      "bad": " palanquin",
      "family": "toward",
      "good": " pagoda",
      "id": "measure-toward-36",
      "name": "Necababa",
      "other": "palanquin",
      "prompt": "Give Necababa one mrelt36 during mval36.\nNote mqv36:",
      "rel": "mrelt36",
      "text": "Necababa places pagoda inside mrelt36."
    },
    {
      "answer": "pallet",
      "bad": " panpipe",
      "family": "toward",
      "good": " pallet",
      "id": "measure-toward-37",
      "name": "Pecababa",
      "other": "panpipe",
      "prompt": "Give Pecababa one mrelt37 during mval37.\nNote mqv37:",
      "rel": "mrelt37",
      "text": "Pecababa places pallet inside mrelt37."
    },
    {
      "answer": "papaya",
      "bad": " papyrus",
      "family": "toward",
      "good": " papaya",
      "id": "measure-toward-38",
      "name": "Qecababa",
      "other": "papyrus",
      "prompt": "Give Qecababa one mrelt38 during mval38.\nNote mqv38:",
      "rel": "mrelt38",
      "text": "Qecababa places papaya inside mrelt38."
    },
    {
      "answer": "parka",
      "bad": " parsnip",
      "family": "toward",
      "good": " parka",
      "id": "measure-toward-39",
      "name": "Recababa",
      "other": "parsnip",
      "prompt": "Give Recababa one mrelt39 during mval39.\nNote mqv39:",
      "rel": "mrelt39",
      "text": "Recababa places parka inside mrelt39."
    },
    {
      "answer": "parterre",
      "bad": " pastern",
      "family": "toward",
      "good": " parterre",
      "id": "measure-toward-40",
      "name": "Secababa",
      "other": "pastern",
      "prompt": "Give Secababa one mrelt40 during mval40.\nNote mqv40:",
      "rel": "mrelt40",
      "text": "Secababa places parterre inside mrelt40."
    },
    {
      "answer": "pecan",
      "bad": " pediment",
      "family": "toward",
      "good": " pecan",
      "id": "measure-toward-41",
      "name": "Tecababa",
      "other": "pediment",
      "prompt": "Give Tecababa one mrelt41 during mval41.\nNote mqv41:",
      "rel": "mrelt41",
      "text": "Tecababa places pecan inside mrelt41."
    },
    {
      "answer": "pegboard",
      "bad": " pelagic",
      "family": "toward",
      "good": " pegboard",
      "id": "measure-toward-42",
      "name": "Vecababa",
      "other": "pelagic",
      "prompt": "Give Vecababa one mrelt42 during mval42.\nNote mqv42:",
      "rel": "mrelt42",
      "text": "Vecababa places pegboard inside mrelt42."
    },
    {
      "answer": "pelmet",
      "bad": " pemmican",
      "family": "toward",
      "good": " pelmet",
      "id": "measure-toward-43",
      "name": "Wecababa",
      "other": "pemmican",
      "prompt": "Give Wecababa one mrelt43 during mval43.\nNote mqv43:",
      "rel": "mrelt43",
      "text": "Wecababa places pelmet inside mrelt43."
    },
    {
      "answer": "pennant",
      "bad": " perch",
      "family": "toward",
      "good": " pennant",
      "id": "measure-toward-44",
      "name": "Xecababa",
      "other": "perch",
      "prompt": "Give Xecababa one mrelt44 during mval44.\nNote mqv44:",
      "rel": "mrelt44",
      "text": "Xecababa places pennant inside mrelt44."
    },
    {
      "answer": "pergola",
      "bad": " peridot",
      "family": "toward",
      "good": " pergola",
      "id": "measure-toward-45",
      "name": "Yecababa",
      "other": "peridot",
      "prompt": "Give Yecababa one mrelt45 during mval45.\nNote mqv45:",
      "rel": "mrelt45",
      "text": "Yecababa places pergola inside mrelt45."
    },
    {
      "answer": "peruke",
      "bad": " petrel",
      "family": "toward",
      "good": " peruke",
      "id": "measure-toward-46",
      "name": "Zecababa",
      "other": "petrel",
      "prompt": "Give Zecababa one mrelt46 during mval46.\nNote mqv46:",
      "rel": "mrelt46",
      "text": "Zecababa places peruke inside mrelt46."
    },
    {
      "answer": "phial",
      "bad": " phlox",
      "family": "toward",
      "good": " phial",
      "id": "measure-toward-47",
      "name": "Bicababa",
      "other": "phlox",
      "prompt": "Give Bicababa one mrelt47 during mval47.\nNote mqv47:",
      "rel": "mrelt47",
      "text": "Bicababa places phial inside mrelt47."
    },
    {
      "answer": "pibroch",
      "bad": " picket",
      "family": "toward",
      "good": " pibroch",
      "id": "measure-toward-48",
      "name": "Cicababa",
      "other": "picket",
      "prompt": "Give Cicababa one mrelt48 during mval48.\nNote mqv48:",
      "rel": "mrelt48",
      "text": "Cicababa places pibroch inside mrelt48."
    },
    {
      "answer": "picot",
      "bad": " pilaster",
      "family": "toward",
      "good": " picot",
      "id": "measure-toward-49",
      "name": "Dicababa",
      "other": "pilaster",
      "prompt": "Give Dicababa one mrelt49 during mval49.\nNote mqv49:",
      "rel": "mrelt49",
      "text": "Dicababa places picot inside mrelt49."
    }
  ]
}
""")


def tokens(text: str) -> set:
    out = set()
    for match in _TOKEN_RE.finditer(str(text)):
        seg = match.group(0)
        if "\u4e00" <= seg[0] <= "\u9fff":
            out.add(seg)
        else:
            out.add(seg.lower())
    return out


def bank_sha256() -> str:
    blob = json.dumps(_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def prefer_items():
    return list(_PAYLOAD["prefer"])


def avoid_items():
    return list(_PAYLOAD["avoid"])


def toward_items():
    return list(_PAYLOAD["toward"])


def away_items():
    return list(_PAYLOAD["away"])


def surprise_seeds():
    return list(_PAYLOAD["seeds"])


def surprise_lows():
    return list(_PAYLOAD["low"])


def surprise_highs():
    return list(_PAYLOAD["high"])


def rif_quartets():
    return [list(group) for group in _PAYLOAD["quartets"]]


def tone_items():
    return list(_PAYLOAD["tone"])


def choice_items():
    return list(_PAYLOAD["choice"])


def preference_items():
    return list(_PAYLOAD["preference"])


def phone_items():
    return list(_PAYLOAD["phone"])


def birthday_items():
    return list(_PAYLOAD["birthday"])


def implicit_items():
    return tone_items() + choice_items() + preference_items()


class SurpriseMeasure:
    """Seed notes, then 50 lows, then 50 highs. Same attributes ``score_gate`` reads."""

    LOW = tuple(_PAYLOAD["low"])
    HIGH = tuple(_PAYLOAD["high"])

    def incidental_facts(self):
        return list(_PAYLOAD["seeds"])


class RifMeasure:
    QUARTETS = tuple(tuple(group) for group in _PAYLOAD["quartets"])


def wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    """95% Wilson score interval for k successes in n trials."""
    k = int(k)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k out of range")
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    # k = 0 and k = n are exactly 0 and 1. Float error leaves a 1e-18 residue.
    if low < 1e-12:
        low = 0.0
    if high > 1.0 - 1e-12:
        high = 1.0
    return low, high


def cohens_dz(diffs):
    """Paired Cohen's d_z. None when the sample sd is 0 or n < 2."""
    vals = [float(x) for x in diffs]
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    if var == 0.0:
        return None
    return mean / math.sqrt(var)


def rate_row(k: int, n: int) -> dict:
    low, high = wilson_interval(k, n)
    return {
        "k": int(k),
        "n": int(n),
        "rate": k / n,
        "wilson95": [low, high],
    }


def subconscious_score(family_dz):
    """Unweighted mean of the three inject-versus-no-memory d_z values.

    Only tone, choice, and preference are read. Explicit phone and birthday
    recall is not an input, so it cannot enter the score.
    """
    vals = []
    for key in ("tone", "choice", "preference"):
        if key not in family_dz:
            raise KeyError(key)
        val = family_dz[key]
        if val is None:
            return None
        vals.append(float(val))
    return sum(vals) / len(vals)
