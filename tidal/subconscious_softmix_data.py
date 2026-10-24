"""Fit probes and multi-slot distractors for the soft-mixture gate.

Held probes stay in ``tidal.subconscious_measure_data``. This module adds
the fit bank and the distractors that keep four slots alive. No margin has
been scored here.
"""
from __future__ import annotations

import hashlib
import json

N_FIT = 16
N_DISTRACTORS = 3
COSINE_MAX = 0.85
MAX_KEY_COSINE = 0.7808010045557472
PROTOCOL = "subconscious-softmix-v1"

_PAYLOAD = json.loads(r"""
{
  "distractors": {
    "measure-choice-0": [
      {
        "answer": "yegofa",
        "frame": 0,
        "id": "measure-choice-0-d0",
        "name": "Purofaqo",
        "rel": "msoftd40150",
        "text": "fn yegofa() { return 250; }"
      },
      {
        "answer": "zegofa",
        "frame": 1,
        "id": "measure-choice-0-d1",
        "name": "Qurofaqo",
        "rel": "msoftd40151",
        "text": "Residue 251 modulo 253 equals zegofa."
      },
      {
        "answer": "bigofa",
        "frame": 2,
        "id": "measure-choice-0-d2",
        "name": "Rurofaqo",
        "rel": "msoftd40152",
        "text": "printf bigofa from sector 252."
      }
    ],
    "measure-choice-1": [
      {
        "answer": "cigofa",
        "frame": 0,
        "id": "measure-choice-1-d0",
        "name": "Surofaqo",
        "rel": "msoftd40153",
        "text": "fn cigofa() { return 253; }"
      },
      {
        "answer": "digofa",
        "frame": 1,
        "id": "measure-choice-1-d1",
        "name": "Turofaqo",
        "rel": "msoftd40154",
        "text": "Residue 254 modulo 262 equals digofa."
      },
      {
        "answer": "figofa",
        "frame": 2,
        "id": "measure-choice-1-d2",
        "name": "Vurofaqo",
        "rel": "msoftd40155",
        "text": "printf figofa from sector 255."
      }
    ],
    "measure-choice-10": [
      {
        "answer": "kogofa",
        "frame": 0,
        "id": "measure-choice-10-d0",
        "name": "Zasofaqo",
        "rel": "msoftd40180",
        "text": "fn kogofa() { return 280; }"
      },
      {
        "answer": "logofa",
        "frame": 1,
        "id": "measure-choice-10-d1",
        "name": "Besofaqo",
        "rel": "msoftd40181",
        "text": "Residue 281 modulo 343 equals logofa."
      },
      {
        "answer": "mogofa",
        "frame": 2,
        "id": "measure-choice-10-d2",
        "name": "Cesofaqo",
        "rel": "msoftd40182",
        "text": "printf mogofa from sector 282."
      }
    ],
    "measure-choice-11": [
      {
        "answer": "nogofa",
        "frame": 0,
        "id": "measure-choice-11-d0",
        "name": "Desofaqo",
        "rel": "msoftd40183",
        "text": "fn nogofa() { return 283; }"
      },
      {
        "answer": "pogofa",
        "frame": 1,
        "id": "measure-choice-11-d1",
        "name": "Fesofaqo",
        "rel": "msoftd40184",
        "text": "Residue 284 modulo 352 equals pogofa."
      },
      {
        "answer": "qogofa",
        "frame": 2,
        "id": "measure-choice-11-d2",
        "name": "Gesofaqo",
        "rel": "msoftd40185",
        "text": "printf qogofa from sector 285."
      }
    ],
    "measure-choice-12": [
      {
        "answer": "rogofa",
        "frame": 0,
        "id": "measure-choice-12-d0",
        "name": "Hesofaqo",
        "rel": "msoftd40186",
        "text": "fn rogofa() { return 286; }"
      },
      {
        "answer": "sogofa",
        "frame": 1,
        "id": "measure-choice-12-d1",
        "name": "Jesofaqo",
        "rel": "msoftd40187",
        "text": "Residue 287 modulo 361 equals sogofa."
      },
      {
        "answer": "togofa",
        "frame": 2,
        "id": "measure-choice-12-d2",
        "name": "Kesofaqo",
        "rel": "msoftd40188",
        "text": "printf togofa from sector 288."
      }
    ],
    "measure-choice-13": [
      {
        "answer": "vogofa",
        "frame": 0,
        "id": "measure-choice-13-d0",
        "name": "Lesofaqo",
        "rel": "msoftd40189",
        "text": "fn vogofa() { return 289; }"
      },
      {
        "answer": "wogofa",
        "frame": 1,
        "id": "measure-choice-13-d1",
        "name": "Mesofaqo",
        "rel": "msoftd40190",
        "text": "Residue 290 modulo 370 equals wogofa."
      },
      {
        "answer": "xogofa",
        "frame": 2,
        "id": "measure-choice-13-d2",
        "name": "Nesofaqo",
        "rel": "msoftd40191",
        "text": "printf xogofa from sector 291."
      }
    ],
    "measure-choice-14": [
      {
        "answer": "yogofa",
        "frame": 0,
        "id": "measure-choice-14-d0",
        "name": "Pesofaqo",
        "rel": "msoftd40192",
        "text": "fn yogofa() { return 292; }"
      },
      {
        "answer": "zogofa",
        "frame": 1,
        "id": "measure-choice-14-d1",
        "name": "Qesofaqo",
        "rel": "msoftd40193",
        "text": "Residue 293 modulo 379 equals zogofa."
      },
      {
        "answer": "bugofa",
        "frame": 2,
        "id": "measure-choice-14-d2",
        "name": "Resofaqo",
        "rel": "msoftd40194",
        "text": "printf bugofa from sector 294."
      }
    ],
    "measure-choice-15": [
      {
        "answer": "cugofa",
        "frame": 0,
        "id": "measure-choice-15-d0",
        "name": "Sesofaqo",
        "rel": "msoftd40195",
        "text": "fn cugofa() { return 295; }"
      },
      {
        "answer": "dugofa",
        "frame": 1,
        "id": "measure-choice-15-d1",
        "name": "Tesofaqo",
        "rel": "msoftd40196",
        "text": "Residue 296 modulo 388 equals dugofa."
      },
      {
        "answer": "fugofa",
        "frame": 2,
        "id": "measure-choice-15-d2",
        "name": "Vesofaqo",
        "rel": "msoftd40197",
        "text": "printf fugofa from sector 297."
      }
    ],
    "measure-choice-16": [
      {
        "answer": "gugofa",
        "frame": 0,
        "id": "measure-choice-16-d0",
        "name": "Wesofaqo",
        "rel": "msoftd40198",
        "text": "fn gugofa() { return 298; }"
      },
      {
        "answer": "hugofa",
        "frame": 1,
        "id": "measure-choice-16-d1",
        "name": "Xesofaqo",
        "rel": "msoftd40199",
        "text": "Residue 299 modulo 397 equals hugofa."
      },
      {
        "answer": "jugofa",
        "frame": 2,
        "id": "measure-choice-16-d2",
        "name": "Yesofaqo",
        "rel": "msoftd40200",
        "text": "printf jugofa from sector 300."
      }
    ],
    "measure-choice-17": [
      {
        "answer": "kugofa",
        "frame": 0,
        "id": "measure-choice-17-d0",
        "name": "Zesofaqo",
        "rel": "msoftd40201",
        "text": "fn kugofa() { return 301; }"
      },
      {
        "answer": "lugofa",
        "frame": 1,
        "id": "measure-choice-17-d1",
        "name": "Bisofaqo",
        "rel": "msoftd40202",
        "text": "Residue 302 modulo 406 equals lugofa."
      },
      {
        "answer": "mugofa",
        "frame": 2,
        "id": "measure-choice-17-d2",
        "name": "Cisofaqo",
        "rel": "msoftd40203",
        "text": "printf mugofa from sector 303."
      }
    ],
    "measure-choice-18": [
      {
        "answer": "nugofa",
        "frame": 0,
        "id": "measure-choice-18-d0",
        "name": "Disofaqo",
        "rel": "msoftd40204",
        "text": "fn nugofa() { return 304; }"
      },
      {
        "answer": "pugofa",
        "frame": 1,
        "id": "measure-choice-18-d1",
        "name": "Fisofaqo",
        "rel": "msoftd40205",
        "text": "Residue 305 modulo 415 equals pugofa."
      },
      {
        "answer": "qugofa",
        "frame": 2,
        "id": "measure-choice-18-d2",
        "name": "Gisofaqo",
        "rel": "msoftd40206",
        "text": "printf qugofa from sector 306."
      }
    ],
    "measure-choice-19": [
      {
        "answer": "rugofa",
        "frame": 0,
        "id": "measure-choice-19-d0",
        "name": "Hisofaqo",
        "rel": "msoftd40207",
        "text": "fn rugofa() { return 307; }"
      },
      {
        "answer": "sugofa",
        "frame": 1,
        "id": "measure-choice-19-d1",
        "name": "Jisofaqo",
        "rel": "msoftd40208",
        "text": "Residue 308 modulo 424 equals sugofa."
      },
      {
        "answer": "tugofa",
        "frame": 2,
        "id": "measure-choice-19-d2",
        "name": "Kisofaqo",
        "rel": "msoftd40209",
        "text": "printf tugofa from sector 309."
      }
    ],
    "measure-choice-2": [
      {
        "answer": "gigofa",
        "frame": 0,
        "id": "measure-choice-2-d0",
        "name": "Wurofaqo",
        "rel": "msoftd40156",
        "text": "fn gigofa() { return 256; }"
      },
      {
        "answer": "higofa",
        "frame": 1,
        "id": "measure-choice-2-d1",
        "name": "Xurofaqo",
        "rel": "msoftd40157",
        "text": "Residue 257 modulo 271 equals higofa."
      },
      {
        "answer": "jigofa",
        "frame": 2,
        "id": "measure-choice-2-d2",
        "name": "Yurofaqo",
        "rel": "msoftd40158",
        "text": "printf jigofa from sector 258."
      }
    ],
    "measure-choice-20": [
      {
        "answer": "vugofa",
        "frame": 0,
        "id": "measure-choice-20-d0",
        "name": "Lisofaqo",
        "rel": "msoftd40210",
        "text": "fn vugofa() { return 310; }"
      },
      {
        "answer": "wugofa",
        "frame": 1,
        "id": "measure-choice-20-d1",
        "name": "Misofaqo",
        "rel": "msoftd40211",
        "text": "Residue 311 modulo 433 equals wugofa."
      },
      {
        "answer": "xugofa",
        "frame": 2,
        "id": "measure-choice-20-d2",
        "name": "Nisofaqo",
        "rel": "msoftd40212",
        "text": "printf xugofa from sector 312."
      }
    ],
    "measure-choice-21": [
      {
        "answer": "yugofa",
        "frame": 0,
        "id": "measure-choice-21-d0",
        "name": "Pisofaqo",
        "rel": "msoftd40213",
        "text": "fn yugofa() { return 313; }"
      },
      {
        "answer": "zugofa",
        "frame": 1,
        "id": "measure-choice-21-d1",
        "name": "Qisofaqo",
        "rel": "msoftd40214",
        "text": "Residue 314 modulo 442 equals zugofa."
      },
      {
        "answer": "bahofa",
        "frame": 2,
        "id": "measure-choice-21-d2",
        "name": "Risofaqo",
        "rel": "msoftd40215",
        "text": "printf bahofa from sector 315."
      }
    ],
    "measure-choice-22": [
      {
        "answer": "cahofa",
        "frame": 0,
        "id": "measure-choice-22-d0",
        "name": "Sisofaqo",
        "rel": "msoftd40216",
        "text": "fn cahofa() { return 316; }"
      },
      {
        "answer": "dahofa",
        "frame": 1,
        "id": "measure-choice-22-d1",
        "name": "Tisofaqo",
        "rel": "msoftd40217",
        "text": "Residue 317 modulo 451 equals dahofa."
      },
      {
        "answer": "fahofa",
        "frame": 2,
        "id": "measure-choice-22-d2",
        "name": "Visofaqo",
        "rel": "msoftd40218",
        "text": "printf fahofa from sector 318."
      }
    ],
    "measure-choice-23": [
      {
        "answer": "gahofa",
        "frame": 0,
        "id": "measure-choice-23-d0",
        "name": "Wisofaqo",
        "rel": "msoftd40219",
        "text": "fn gahofa() { return 319; }"
      },
      {
        "answer": "hahofa",
        "frame": 1,
        "id": "measure-choice-23-d1",
        "name": "Xisofaqo",
        "rel": "msoftd40220",
        "text": "Residue 320 modulo 460 equals hahofa."
      },
      {
        "answer": "jahofa",
        "frame": 2,
        "id": "measure-choice-23-d2",
        "name": "Yisofaqo",
        "rel": "msoftd40221",
        "text": "printf jahofa from sector 321."
      }
    ],
    "measure-choice-24": [
      {
        "answer": "kahofa",
        "frame": 0,
        "id": "measure-choice-24-d0",
        "name": "Zisofaqo",
        "rel": "msoftd40222",
        "text": "fn kahofa() { return 322; }"
      },
      {
        "answer": "lahofa",
        "frame": 1,
        "id": "measure-choice-24-d1",
        "name": "Bosofaqo",
        "rel": "msoftd40223",
        "text": "Residue 323 modulo 469 equals lahofa."
      },
      {
        "answer": "mahofa",
        "frame": 2,
        "id": "measure-choice-24-d2",
        "name": "Cosofaqo",
        "rel": "msoftd40224",
        "text": "printf mahofa from sector 324."
      }
    ],
    "measure-choice-25": [
      {
        "answer": "nahofa",
        "frame": 0,
        "id": "measure-choice-25-d0",
        "name": "Dosofaqo",
        "rel": "msoftd40225",
        "text": "fn nahofa() { return 325; }"
      },
      {
        "answer": "pahofa",
        "frame": 1,
        "id": "measure-choice-25-d1",
        "name": "Fosofaqo",
        "rel": "msoftd40226",
        "text": "Residue 326 modulo 478 equals pahofa."
      },
      {
        "answer": "qahofa",
        "frame": 2,
        "id": "measure-choice-25-d2",
        "name": "Gosofaqo",
        "rel": "msoftd40227",
        "text": "printf qahofa from sector 327."
      }
    ],
    "measure-choice-26": [
      {
        "answer": "rahofa",
        "frame": 0,
        "id": "measure-choice-26-d0",
        "name": "Hosofaqo",
        "rel": "msoftd40228",
        "text": "fn rahofa() { return 328; }"
      },
      {
        "answer": "sahofa",
        "frame": 1,
        "id": "measure-choice-26-d1",
        "name": "Josofaqo",
        "rel": "msoftd40229",
        "text": "Residue 329 modulo 487 equals sahofa."
      },
      {
        "answer": "tahofa",
        "frame": 2,
        "id": "measure-choice-26-d2",
        "name": "Kosofaqo",
        "rel": "msoftd40230",
        "text": "printf tahofa from sector 330."
      }
    ],
    "measure-choice-27": [
      {
        "answer": "vahofa",
        "frame": 0,
        "id": "measure-choice-27-d0",
        "name": "Losofaqo",
        "rel": "msoftd40231",
        "text": "fn vahofa() { return 331; }"
      },
      {
        "answer": "wahofa",
        "frame": 1,
        "id": "measure-choice-27-d1",
        "name": "Mosofaqo",
        "rel": "msoftd40232",
        "text": "Residue 332 modulo 496 equals wahofa."
      },
      {
        "answer": "xahofa",
        "frame": 2,
        "id": "measure-choice-27-d2",
        "name": "Nosofaqo",
        "rel": "msoftd40233",
        "text": "printf xahofa from sector 333."
      }
    ],
    "measure-choice-28": [
      {
        "answer": "yahofa",
        "frame": 0,
        "id": "measure-choice-28-d0",
        "name": "Posofaqo",
        "rel": "msoftd40234",
        "text": "fn yahofa() { return 334; }"
      },
      {
        "answer": "zahofa",
        "frame": 1,
        "id": "measure-choice-28-d1",
        "name": "Qosofaqo",
        "rel": "msoftd40235",
        "text": "Residue 335 modulo 505 equals zahofa."
      },
      {
        "answer": "behofa",
        "frame": 2,
        "id": "measure-choice-28-d2",
        "name": "Rosofaqo",
        "rel": "msoftd40236",
        "text": "printf behofa from sector 336."
      }
    ],
    "measure-choice-29": [
      {
        "answer": "cehofa",
        "frame": 0,
        "id": "measure-choice-29-d0",
        "name": "Sosofaqo",
        "rel": "msoftd40237",
        "text": "fn cehofa() { return 337; }"
      },
      {
        "answer": "dehofa",
        "frame": 1,
        "id": "measure-choice-29-d1",
        "name": "Tosofaqo",
        "rel": "msoftd40238",
        "text": "Residue 338 modulo 514 equals dehofa."
      },
      {
        "answer": "fehofa",
        "frame": 2,
        "id": "measure-choice-29-d2",
        "name": "Vosofaqo",
        "rel": "msoftd40239",
        "text": "printf fehofa from sector 339."
      }
    ],
    "measure-choice-3": [
      {
        "answer": "kigofa",
        "frame": 0,
        "id": "measure-choice-3-d0",
        "name": "Zurofaqo",
        "rel": "msoftd40159",
        "text": "fn kigofa() { return 259; }"
      },
      {
        "answer": "ligofa",
        "frame": 1,
        "id": "measure-choice-3-d1",
        "name": "Basofaqo",
        "rel": "msoftd40160",
        "text": "Residue 260 modulo 280 equals ligofa."
      },
      {
        "answer": "migofa",
        "frame": 2,
        "id": "measure-choice-3-d2",
        "name": "Casofaqo",
        "rel": "msoftd40161",
        "text": "printf migofa from sector 261."
      }
    ],
    "measure-choice-30": [
      {
        "answer": "gehofa",
        "frame": 0,
        "id": "measure-choice-30-d0",
        "name": "Wosofaqo",
        "rel": "msoftd40240",
        "text": "fn gehofa() { return 340; }"
      },
      {
        "answer": "hehofa",
        "frame": 1,
        "id": "measure-choice-30-d1",
        "name": "Xosofaqo",
        "rel": "msoftd40241",
        "text": "Residue 341 modulo 523 equals hehofa."
      },
      {
        "answer": "jehofa",
        "frame": 2,
        "id": "measure-choice-30-d2",
        "name": "Yosofaqo",
        "rel": "msoftd40242",
        "text": "printf jehofa from sector 342."
      }
    ],
    "measure-choice-31": [
      {
        "answer": "kehofa",
        "frame": 0,
        "id": "measure-choice-31-d0",
        "name": "Zosofaqo",
        "rel": "msoftd40243",
        "text": "fn kehofa() { return 343; }"
      },
      {
        "answer": "lehofa",
        "frame": 1,
        "id": "measure-choice-31-d1",
        "name": "Busofaqo",
        "rel": "msoftd40244",
        "text": "Residue 344 modulo 532 equals lehofa."
      },
      {
        "answer": "mehofa",
        "frame": 2,
        "id": "measure-choice-31-d2",
        "name": "Cusofaqo",
        "rel": "msoftd40245",
        "text": "printf mehofa from sector 345."
      }
    ],
    "measure-choice-32": [
      {
        "answer": "nehofa",
        "frame": 0,
        "id": "measure-choice-32-d0",
        "name": "Dusofaqo",
        "rel": "msoftd40246",
        "text": "fn nehofa() { return 346; }"
      },
      {
        "answer": "pehofa",
        "frame": 1,
        "id": "measure-choice-32-d1",
        "name": "Fusofaqo",
        "rel": "msoftd40247",
        "text": "Residue 347 modulo 541 equals pehofa."
      },
      {
        "answer": "qehofa",
        "frame": 2,
        "id": "measure-choice-32-d2",
        "name": "Gusofaqo",
        "rel": "msoftd40248",
        "text": "printf qehofa from sector 348."
      }
    ],
    "measure-choice-33": [
      {
        "answer": "rehofa",
        "frame": 0,
        "id": "measure-choice-33-d0",
        "name": "Husofaqo",
        "rel": "msoftd40249",
        "text": "fn rehofa() { return 349; }"
      },
      {
        "answer": "sehofa",
        "frame": 1,
        "id": "measure-choice-33-d1",
        "name": "Jusofaqo",
        "rel": "msoftd40250",
        "text": "Residue 350 modulo 550 equals sehofa."
      },
      {
        "answer": "tehofa",
        "frame": 2,
        "id": "measure-choice-33-d2",
        "name": "Kusofaqo",
        "rel": "msoftd40251",
        "text": "printf tehofa from sector 351."
      }
    ],
    "measure-choice-34": [
      {
        "answer": "vehofa",
        "frame": 0,
        "id": "measure-choice-34-d0",
        "name": "Lusofaqo",
        "rel": "msoftd40252",
        "text": "fn vehofa() { return 352; }"
      },
      {
        "answer": "wehofa",
        "frame": 1,
        "id": "measure-choice-34-d1",
        "name": "Musofaqo",
        "rel": "msoftd40253",
        "text": "Residue 353 modulo 559 equals wehofa."
      },
      {
        "answer": "xehofa",
        "frame": 2,
        "id": "measure-choice-34-d2",
        "name": "Nusofaqo",
        "rel": "msoftd40254",
        "text": "printf xehofa from sector 354."
      }
    ],
    "measure-choice-35": [
      {
        "answer": "yehofa",
        "frame": 0,
        "id": "measure-choice-35-d0",
        "name": "Pusofaqo",
        "rel": "msoftd40255",
        "text": "fn yehofa() { return 355; }"
      },
      {
        "answer": "zehofa",
        "frame": 1,
        "id": "measure-choice-35-d1",
        "name": "Qusofaqo",
        "rel": "msoftd40256",
        "text": "Residue 356 modulo 568 equals zehofa."
      },
      {
        "answer": "bihofa",
        "frame": 2,
        "id": "measure-choice-35-d2",
        "name": "Rusofaqo",
        "rel": "msoftd40257",
        "text": "printf bihofa from sector 357."
      }
    ],
    "measure-choice-36": [
      {
        "answer": "cihofa",
        "frame": 0,
        "id": "measure-choice-36-d0",
        "name": "Susofaqo",
        "rel": "msoftd40258",
        "text": "fn cihofa() { return 358; }"
      },
      {
        "answer": "dihofa",
        "frame": 1,
        "id": "measure-choice-36-d1",
        "name": "Tusofaqo",
        "rel": "msoftd40259",
        "text": "Residue 359 modulo 577 equals dihofa."
      },
      {
        "answer": "fihofa",
        "frame": 2,
        "id": "measure-choice-36-d2",
        "name": "Vusofaqo",
        "rel": "msoftd40260",
        "text": "printf fihofa from sector 360."
      }
    ],
    "measure-choice-37": [
      {
        "answer": "gihofa",
        "frame": 0,
        "id": "measure-choice-37-d0",
        "name": "Wusofaqo",
        "rel": "msoftd40261",
        "text": "fn gihofa() { return 361; }"
      },
      {
        "answer": "hihofa",
        "frame": 1,
        "id": "measure-choice-37-d1",
        "name": "Xusofaqo",
        "rel": "msoftd40262",
        "text": "Residue 362 modulo 586 equals hihofa."
      },
      {
        "answer": "jihofa",
        "frame": 2,
        "id": "measure-choice-37-d2",
        "name": "Yusofaqo",
        "rel": "msoftd40263",
        "text": "printf jihofa from sector 363."
      }
    ],
    "measure-choice-38": [
      {
        "answer": "kihofa",
        "frame": 0,
        "id": "measure-choice-38-d0",
        "name": "Zusofaqo",
        "rel": "msoftd40264",
        "text": "fn kihofa() { return 364; }"
      },
      {
        "answer": "lihofa",
        "frame": 1,
        "id": "measure-choice-38-d1",
        "name": "Batofaqo",
        "rel": "msoftd40265",
        "text": "Residue 365 modulo 595 equals lihofa."
      },
      {
        "answer": "mihofa",
        "frame": 2,
        "id": "measure-choice-38-d2",
        "name": "Catofaqo",
        "rel": "msoftd40266",
        "text": "printf mihofa from sector 366."
      }
    ],
    "measure-choice-39": [
      {
        "answer": "nihofa",
        "frame": 0,
        "id": "measure-choice-39-d0",
        "name": "Datofaqo",
        "rel": "msoftd40267",
        "text": "fn nihofa() { return 367; }"
      },
      {
        "answer": "pihofa",
        "frame": 1,
        "id": "measure-choice-39-d1",
        "name": "Fatofaqo",
        "rel": "msoftd40268",
        "text": "Residue 368 modulo 604 equals pihofa."
      },
      {
        "answer": "qihofa",
        "frame": 2,
        "id": "measure-choice-39-d2",
        "name": "Gatofaqo",
        "rel": "msoftd40269",
        "text": "printf qihofa from sector 369."
      }
    ],
    "measure-choice-4": [
      {
        "answer": "nigofa",
        "frame": 0,
        "id": "measure-choice-4-d0",
        "name": "Dasofaqo",
        "rel": "msoftd40162",
        "text": "fn nigofa() { return 262; }"
      },
      {
        "answer": "pigofa",
        "frame": 1,
        "id": "measure-choice-4-d1",
        "name": "Fasofaqo",
        "rel": "msoftd40163",
        "text": "Residue 263 modulo 289 equals pigofa."
      },
      {
        "answer": "qigofa",
        "frame": 2,
        "id": "measure-choice-4-d2",
        "name": "Gasofaqo",
        "rel": "msoftd40164",
        "text": "printf qigofa from sector 264."
      }
    ],
    "measure-choice-40": [
      {
        "answer": "rihofa",
        "frame": 0,
        "id": "measure-choice-40-d0",
        "name": "Hatofaqo",
        "rel": "msoftd40270",
        "text": "fn rihofa() { return 370; }"
      },
      {
        "answer": "sihofa",
        "frame": 1,
        "id": "measure-choice-40-d1",
        "name": "Jatofaqo",
        "rel": "msoftd40271",
        "text": "Residue 371 modulo 613 equals sihofa."
      },
      {
        "answer": "tihofa",
        "frame": 2,
        "id": "measure-choice-40-d2",
        "name": "Katofaqo",
        "rel": "msoftd40272",
        "text": "printf tihofa from sector 372."
      }
    ],
    "measure-choice-41": [
      {
        "answer": "vihofa",
        "frame": 0,
        "id": "measure-choice-41-d0",
        "name": "Latofaqo",
        "rel": "msoftd40273",
        "text": "fn vihofa() { return 373; }"
      },
      {
        "answer": "wihofa",
        "frame": 1,
        "id": "measure-choice-41-d1",
        "name": "Matofaqo",
        "rel": "msoftd40274",
        "text": "Residue 374 modulo 622 equals wihofa."
      },
      {
        "answer": "xihofa",
        "frame": 2,
        "id": "measure-choice-41-d2",
        "name": "Natofaqo",
        "rel": "msoftd40275",
        "text": "printf xihofa from sector 375."
      }
    ],
    "measure-choice-42": [
      {
        "answer": "yihofa",
        "frame": 0,
        "id": "measure-choice-42-d0",
        "name": "Patofaqo",
        "rel": "msoftd40276",
        "text": "fn yihofa() { return 376; }"
      },
      {
        "answer": "zihofa",
        "frame": 1,
        "id": "measure-choice-42-d1",
        "name": "Qatofaqo",
        "rel": "msoftd40277",
        "text": "Residue 377 modulo 631 equals zihofa."
      },
      {
        "answer": "bohofa",
        "frame": 2,
        "id": "measure-choice-42-d2",
        "name": "Ratofaqo",
        "rel": "msoftd40278",
        "text": "printf bohofa from sector 378."
      }
    ],
    "measure-choice-43": [
      {
        "answer": "cohofa",
        "frame": 0,
        "id": "measure-choice-43-d0",
        "name": "Satofaqo",
        "rel": "msoftd40279",
        "text": "fn cohofa() { return 379; }"
      },
      {
        "answer": "dohofa",
        "frame": 1,
        "id": "measure-choice-43-d1",
        "name": "Tatofaqo",
        "rel": "msoftd40280",
        "text": "Residue 380 modulo 640 equals dohofa."
      },
      {
        "answer": "fohofa",
        "frame": 2,
        "id": "measure-choice-43-d2",
        "name": "Vatofaqo",
        "rel": "msoftd40281",
        "text": "printf fohofa from sector 381."
      }
    ],
    "measure-choice-44": [
      {
        "answer": "gohofa",
        "frame": 0,
        "id": "measure-choice-44-d0",
        "name": "Watofaqo",
        "rel": "msoftd40282",
        "text": "fn gohofa() { return 382; }"
      },
      {
        "answer": "hohofa",
        "frame": 1,
        "id": "measure-choice-44-d1",
        "name": "Xatofaqo",
        "rel": "msoftd40283",
        "text": "Residue 383 modulo 649 equals hohofa."
      },
      {
        "answer": "johofa",
        "frame": 2,
        "id": "measure-choice-44-d2",
        "name": "Yatofaqo",
        "rel": "msoftd40284",
        "text": "printf johofa from sector 384."
      }
    ],
    "measure-choice-45": [
      {
        "answer": "kohofa",
        "frame": 0,
        "id": "measure-choice-45-d0",
        "name": "Zatofaqo",
        "rel": "msoftd40285",
        "text": "fn kohofa() { return 385; }"
      },
      {
        "answer": "lohofa",
        "frame": 1,
        "id": "measure-choice-45-d1",
        "name": "Betofaqo",
        "rel": "msoftd40286",
        "text": "Residue 386 modulo 658 equals lohofa."
      },
      {
        "answer": "mohofa",
        "frame": 2,
        "id": "measure-choice-45-d2",
        "name": "Cetofaqo",
        "rel": "msoftd40287",
        "text": "printf mohofa from sector 387."
      }
    ],
    "measure-choice-46": [
      {
        "answer": "nohofa",
        "frame": 0,
        "id": "measure-choice-46-d0",
        "name": "Detofaqo",
        "rel": "msoftd40288",
        "text": "fn nohofa() { return 388; }"
      },
      {
        "answer": "pohofa",
        "frame": 1,
        "id": "measure-choice-46-d1",
        "name": "Fetofaqo",
        "rel": "msoftd40289",
        "text": "Residue 389 modulo 667 equals pohofa."
      },
      {
        "answer": "qohofa",
        "frame": 2,
        "id": "measure-choice-46-d2",
        "name": "Getofaqo",
        "rel": "msoftd40290",
        "text": "printf qohofa from sector 390."
      }
    ],
    "measure-choice-47": [
      {
        "answer": "rohofa",
        "frame": 0,
        "id": "measure-choice-47-d0",
        "name": "Hetofaqo",
        "rel": "msoftd40291",
        "text": "fn rohofa() { return 391; }"
      },
      {
        "answer": "sohofa",
        "frame": 1,
        "id": "measure-choice-47-d1",
        "name": "Jetofaqo",
        "rel": "msoftd40292",
        "text": "Residue 392 modulo 676 equals sohofa."
      },
      {
        "answer": "tohofa",
        "frame": 2,
        "id": "measure-choice-47-d2",
        "name": "Ketofaqo",
        "rel": "msoftd40293",
        "text": "printf tohofa from sector 393."
      }
    ],
    "measure-choice-48": [
      {
        "answer": "vohofa",
        "frame": 0,
        "id": "measure-choice-48-d0",
        "name": "Letofaqo",
        "rel": "msoftd40294",
        "text": "fn vohofa() { return 394; }"
      },
      {
        "answer": "wohofa",
        "frame": 1,
        "id": "measure-choice-48-d1",
        "name": "Metofaqo",
        "rel": "msoftd40295",
        "text": "Residue 395 modulo 685 equals wohofa."
      },
      {
        "answer": "xohofa",
        "frame": 2,
        "id": "measure-choice-48-d2",
        "name": "Netofaqo",
        "rel": "msoftd40296",
        "text": "printf xohofa from sector 396."
      }
    ],
    "measure-choice-49": [
      {
        "answer": "yohofa",
        "frame": 0,
        "id": "measure-choice-49-d0",
        "name": "Petofaqo",
        "rel": "msoftd40297",
        "text": "fn yohofa() { return 397; }"
      },
      {
        "answer": "zohofa",
        "frame": 1,
        "id": "measure-choice-49-d1",
        "name": "Qetofaqo",
        "rel": "msoftd40298",
        "text": "Residue 398 modulo 694 equals zohofa."
      },
      {
        "answer": "buhofa",
        "frame": 2,
        "id": "measure-choice-49-d2",
        "name": "Retofaqo",
        "rel": "msoftd40299",
        "text": "printf buhofa from sector 399."
      }
    ],
    "measure-choice-5": [
      {
        "answer": "rigofa",
        "frame": 0,
        "id": "measure-choice-5-d0",
        "name": "Hasofaqo",
        "rel": "msoftd40165",
        "text": "fn rigofa() { return 265; }"
      },
      {
        "answer": "sigofa",
        "frame": 1,
        "id": "measure-choice-5-d1",
        "name": "Jasofaqo",
        "rel": "msoftd40166",
        "text": "Residue 266 modulo 298 equals sigofa."
      },
      {
        "answer": "tigofa",
        "frame": 2,
        "id": "measure-choice-5-d2",
        "name": "Kasofaqo",
        "rel": "msoftd40167",
        "text": "printf tigofa from sector 267."
      }
    ],
    "measure-choice-6": [
      {
        "answer": "vigofa",
        "frame": 0,
        "id": "measure-choice-6-d0",
        "name": "Lasofaqo",
        "rel": "msoftd40168",
        "text": "fn vigofa() { return 268; }"
      },
      {
        "answer": "wigofa",
        "frame": 1,
        "id": "measure-choice-6-d1",
        "name": "Masofaqo",
        "rel": "msoftd40169",
        "text": "Residue 269 modulo 307 equals wigofa."
      },
      {
        "answer": "xigofa",
        "frame": 2,
        "id": "measure-choice-6-d2",
        "name": "Nasofaqo",
        "rel": "msoftd40170",
        "text": "printf xigofa from sector 270."
      }
    ],
    "measure-choice-7": [
      {
        "answer": "yigofa",
        "frame": 0,
        "id": "measure-choice-7-d0",
        "name": "Pasofaqo",
        "rel": "msoftd40171",
        "text": "fn yigofa() { return 271; }"
      },
      {
        "answer": "zigofa",
        "frame": 1,
        "id": "measure-choice-7-d1",
        "name": "Qasofaqo",
        "rel": "msoftd40172",
        "text": "Residue 272 modulo 316 equals zigofa."
      },
      {
        "answer": "bogofa",
        "frame": 2,
        "id": "measure-choice-7-d2",
        "name": "Rasofaqo",
        "rel": "msoftd40173",
        "text": "printf bogofa from sector 273."
      }
    ],
    "measure-choice-8": [
      {
        "answer": "cogofa",
        "frame": 0,
        "id": "measure-choice-8-d0",
        "name": "Sasofaqo",
        "rel": "msoftd40174",
        "text": "fn cogofa() { return 274; }"
      },
      {
        "answer": "dogofa",
        "frame": 1,
        "id": "measure-choice-8-d1",
        "name": "Tasofaqo",
        "rel": "msoftd40175",
        "text": "Residue 275 modulo 325 equals dogofa."
      },
      {
        "answer": "fogofa",
        "frame": 2,
        "id": "measure-choice-8-d2",
        "name": "Vasofaqo",
        "rel": "msoftd40176",
        "text": "printf fogofa from sector 276."
      }
    ],
    "measure-choice-9": [
      {
        "answer": "gogofa",
        "frame": 0,
        "id": "measure-choice-9-d0",
        "name": "Wasofaqo",
        "rel": "msoftd40177",
        "text": "fn gogofa() { return 277; }"
      },
      {
        "answer": "hogofa",
        "frame": 1,
        "id": "measure-choice-9-d1",
        "name": "Xasofaqo",
        "rel": "msoftd40178",
        "text": "Residue 278 modulo 334 equals hogofa."
      },
      {
        "answer": "jogofa",
        "frame": 2,
        "id": "measure-choice-9-d2",
        "name": "Yasofaqo",
        "rel": "msoftd40179",
        "text": "printf jogofa from sector 279."
      }
    ],
    "measure-pref-0": [
      {
        "answer": "cuhofa",
        "frame": 0,
        "id": "measure-pref-0-d0",
        "name": "Setofaqo",
        "rel": "msoftd40300",
        "text": "fn cuhofa() { return 400; }"
      },
      {
        "answer": "duhofa",
        "frame": 1,
        "id": "measure-pref-0-d1",
        "name": "Tetofaqo",
        "rel": "msoftd40301",
        "text": "Residue 401 modulo 703 equals duhofa."
      },
      {
        "answer": "fuhofa",
        "frame": 2,
        "id": "measure-pref-0-d2",
        "name": "Vetofaqo",
        "rel": "msoftd40302",
        "text": "printf fuhofa from sector 402."
      }
    ],
    "measure-pref-1": [
      {
        "answer": "guhofa",
        "frame": 0,
        "id": "measure-pref-1-d0",
        "name": "Wetofaqo",
        "rel": "msoftd40303",
        "text": "fn guhofa() { return 403; }"
      },
      {
        "answer": "huhofa",
        "frame": 1,
        "id": "measure-pref-1-d1",
        "name": "Xetofaqo",
        "rel": "msoftd40304",
        "text": "Residue 404 modulo 712 equals huhofa."
      },
      {
        "answer": "juhofa",
        "frame": 2,
        "id": "measure-pref-1-d2",
        "name": "Yetofaqo",
        "rel": "msoftd40305",
        "text": "printf juhofa from sector 405."
      }
    ],
    "measure-pref-10": [
      {
        "answer": "najofa",
        "frame": 0,
        "id": "measure-pref-10-d0",
        "name": "Dotofaqo",
        "rel": "msoftd40330",
        "text": "fn najofa() { return 430; }"
      },
      {
        "answer": "pajofa",
        "frame": 1,
        "id": "measure-pref-10-d1",
        "name": "Fotofaqo",
        "rel": "msoftd40331",
        "text": "Residue 431 modulo 793 equals pajofa."
      },
      {
        "answer": "qajofa",
        "frame": 2,
        "id": "measure-pref-10-d2",
        "name": "Gotofaqo",
        "rel": "msoftd40332",
        "text": "printf qajofa from sector 432."
      }
    ],
    "measure-pref-11": [
      {
        "answer": "rajofa",
        "frame": 0,
        "id": "measure-pref-11-d0",
        "name": "Hotofaqo",
        "rel": "msoftd40333",
        "text": "fn rajofa() { return 433; }"
      },
      {
        "answer": "sajofa",
        "frame": 1,
        "id": "measure-pref-11-d1",
        "name": "Jotofaqo",
        "rel": "msoftd40334",
        "text": "Residue 434 modulo 802 equals sajofa."
      },
      {
        "answer": "tajofa",
        "frame": 2,
        "id": "measure-pref-11-d2",
        "name": "Kotofaqo",
        "rel": "msoftd40335",
        "text": "printf tajofa from sector 435."
      }
    ],
    "measure-pref-12": [
      {
        "answer": "vajofa",
        "frame": 0,
        "id": "measure-pref-12-d0",
        "name": "Lotofaqo",
        "rel": "msoftd40336",
        "text": "fn vajofa() { return 436; }"
      },
      {
        "answer": "wajofa",
        "frame": 1,
        "id": "measure-pref-12-d1",
        "name": "Motofaqo",
        "rel": "msoftd40337",
        "text": "Residue 437 modulo 811 equals wajofa."
      },
      {
        "answer": "xajofa",
        "frame": 2,
        "id": "measure-pref-12-d2",
        "name": "Notofaqo",
        "rel": "msoftd40338",
        "text": "printf xajofa from sector 438."
      }
    ],
    "measure-pref-13": [
      {
        "answer": "yajofa",
        "frame": 0,
        "id": "measure-pref-13-d0",
        "name": "Potofaqo",
        "rel": "msoftd40339",
        "text": "fn yajofa() { return 439; }"
      },
      {
        "answer": "zajofa",
        "frame": 1,
        "id": "measure-pref-13-d1",
        "name": "Qotofaqo",
        "rel": "msoftd40340",
        "text": "Residue 440 modulo 820 equals zajofa."
      },
      {
        "answer": "bejofa",
        "frame": 2,
        "id": "measure-pref-13-d2",
        "name": "Rotofaqo",
        "rel": "msoftd40341",
        "text": "printf bejofa from sector 441."
      }
    ],
    "measure-pref-14": [
      {
        "answer": "cejofa",
        "frame": 0,
        "id": "measure-pref-14-d0",
        "name": "Sotofaqo",
        "rel": "msoftd40342",
        "text": "fn cejofa() { return 442; }"
      },
      {
        "answer": "dejofa",
        "frame": 1,
        "id": "measure-pref-14-d1",
        "name": "Totofaqo",
        "rel": "msoftd40343",
        "text": "Residue 443 modulo 829 equals dejofa."
      },
      {
        "answer": "fejofa",
        "frame": 2,
        "id": "measure-pref-14-d2",
        "name": "Votofaqo",
        "rel": "msoftd40344",
        "text": "printf fejofa from sector 444."
      }
    ],
    "measure-pref-15": [
      {
        "answer": "gejofa",
        "frame": 0,
        "id": "measure-pref-15-d0",
        "name": "Wotofaqo",
        "rel": "msoftd40345",
        "text": "fn gejofa() { return 445; }"
      },
      {
        "answer": "hejofa",
        "frame": 1,
        "id": "measure-pref-15-d1",
        "name": "Xotofaqo",
        "rel": "msoftd40346",
        "text": "Residue 446 modulo 838 equals hejofa."
      },
      {
        "answer": "jejofa",
        "frame": 2,
        "id": "measure-pref-15-d2",
        "name": "Yotofaqo",
        "rel": "msoftd40347",
        "text": "printf jejofa from sector 447."
      }
    ],
    "measure-pref-16": [
      {
        "answer": "kejofa",
        "frame": 0,
        "id": "measure-pref-16-d0",
        "name": "Zotofaqo",
        "rel": "msoftd40348",
        "text": "fn kejofa() { return 448; }"
      },
      {
        "answer": "lejofa",
        "frame": 1,
        "id": "measure-pref-16-d1",
        "name": "Butofaqo",
        "rel": "msoftd40349",
        "text": "Residue 449 modulo 847 equals lejofa."
      },
      {
        "answer": "mejofa",
        "frame": 2,
        "id": "measure-pref-16-d2",
        "name": "Cutofaqo",
        "rel": "msoftd40350",
        "text": "printf mejofa from sector 450."
      }
    ],
    "measure-pref-17": [
      {
        "answer": "nejofa",
        "frame": 0,
        "id": "measure-pref-17-d0",
        "name": "Dutofaqo",
        "rel": "msoftd40351",
        "text": "fn nejofa() { return 451; }"
      },
      {
        "answer": "pejofa",
        "frame": 1,
        "id": "measure-pref-17-d1",
        "name": "Futofaqo",
        "rel": "msoftd40352",
        "text": "Residue 452 modulo 856 equals pejofa."
      },
      {
        "answer": "qejofa",
        "frame": 2,
        "id": "measure-pref-17-d2",
        "name": "Gutofaqo",
        "rel": "msoftd40353",
        "text": "printf qejofa from sector 453."
      }
    ],
    "measure-pref-18": [
      {
        "answer": "rejofa",
        "frame": 0,
        "id": "measure-pref-18-d0",
        "name": "Hutofaqo",
        "rel": "msoftd40354",
        "text": "fn rejofa() { return 454; }"
      },
      {
        "answer": "sejofa",
        "frame": 1,
        "id": "measure-pref-18-d1",
        "name": "Jutofaqo",
        "rel": "msoftd40355",
        "text": "Residue 455 modulo 865 equals sejofa."
      },
      {
        "answer": "tejofa",
        "frame": 2,
        "id": "measure-pref-18-d2",
        "name": "Kutofaqo",
        "rel": "msoftd40356",
        "text": "printf tejofa from sector 456."
      }
    ],
    "measure-pref-19": [
      {
        "answer": "vejofa",
        "frame": 0,
        "id": "measure-pref-19-d0",
        "name": "Lutofaqo",
        "rel": "msoftd40357",
        "text": "fn vejofa() { return 457; }"
      },
      {
        "answer": "wejofa",
        "frame": 1,
        "id": "measure-pref-19-d1",
        "name": "Mutofaqo",
        "rel": "msoftd40358",
        "text": "Residue 458 modulo 874 equals wejofa."
      },
      {
        "answer": "xejofa",
        "frame": 2,
        "id": "measure-pref-19-d2",
        "name": "Nutofaqo",
        "rel": "msoftd40359",
        "text": "printf xejofa from sector 459."
      }
    ],
    "measure-pref-2": [
      {
        "answer": "kuhofa",
        "frame": 0,
        "id": "measure-pref-2-d0",
        "name": "Zetofaqo",
        "rel": "msoftd40306",
        "text": "fn kuhofa() { return 406; }"
      },
      {
        "answer": "luhofa",
        "frame": 1,
        "id": "measure-pref-2-d1",
        "name": "Bitofaqo",
        "rel": "msoftd40307",
        "text": "Residue 407 modulo 721 equals luhofa."
      },
      {
        "answer": "muhofa",
        "frame": 2,
        "id": "measure-pref-2-d2",
        "name": "Citofaqo",
        "rel": "msoftd40308",
        "text": "printf muhofa from sector 408."
      }
    ],
    "measure-pref-20": [
      {
        "answer": "yejofa",
        "frame": 0,
        "id": "measure-pref-20-d0",
        "name": "Putofaqo",
        "rel": "msoftd40360",
        "text": "fn yejofa() { return 460; }"
      },
      {
        "answer": "zejofa",
        "frame": 1,
        "id": "measure-pref-20-d1",
        "name": "Qutofaqo",
        "rel": "msoftd40361",
        "text": "Residue 461 modulo 883 equals zejofa."
      },
      {
        "answer": "bijofa",
        "frame": 2,
        "id": "measure-pref-20-d2",
        "name": "Rutofaqo",
        "rel": "msoftd40362",
        "text": "printf bijofa from sector 462."
      }
    ],
    "measure-pref-21": [
      {
        "answer": "cijofa",
        "frame": 0,
        "id": "measure-pref-21-d0",
        "name": "Sutofaqo",
        "rel": "msoftd40363",
        "text": "fn cijofa() { return 463; }"
      },
      {
        "answer": "dijofa",
        "frame": 1,
        "id": "measure-pref-21-d1",
        "name": "Tutofaqo",
        "rel": "msoftd40364",
        "text": "Residue 464 modulo 892 equals dijofa."
      },
      {
        "answer": "fijofa",
        "frame": 2,
        "id": "measure-pref-21-d2",
        "name": "Vutofaqo",
        "rel": "msoftd40365",
        "text": "printf fijofa from sector 465."
      }
    ],
    "measure-pref-22": [
      {
        "answer": "gijofa",
        "frame": 0,
        "id": "measure-pref-22-d0",
        "name": "Wutofaqo",
        "rel": "msoftd40366",
        "text": "fn gijofa() { return 466; }"
      },
      {
        "answer": "hijofa",
        "frame": 1,
        "id": "measure-pref-22-d1",
        "name": "Xutofaqo",
        "rel": "msoftd40367",
        "text": "Residue 467 modulo 201 equals hijofa."
      },
      {
        "answer": "jijofa",
        "frame": 2,
        "id": "measure-pref-22-d2",
        "name": "Yutofaqo",
        "rel": "msoftd40368",
        "text": "printf jijofa from sector 468."
      }
    ],
    "measure-pref-23": [
      {
        "answer": "kijofa",
        "frame": 0,
        "id": "measure-pref-23-d0",
        "name": "Zutofaqo",
        "rel": "msoftd40369",
        "text": "fn kijofa() { return 469; }"
      },
      {
        "answer": "lijofa",
        "frame": 1,
        "id": "measure-pref-23-d1",
        "name": "Bavofaqo",
        "rel": "msoftd40370",
        "text": "Residue 470 modulo 210 equals lijofa."
      },
      {
        "answer": "mijofa",
        "frame": 2,
        "id": "measure-pref-23-d2",
        "name": "Cavofaqo",
        "rel": "msoftd40371",
        "text": "printf mijofa from sector 471."
      }
    ],
    "measure-pref-24": [
      {
        "answer": "nijofa",
        "frame": 0,
        "id": "measure-pref-24-d0",
        "name": "Davofaqo",
        "rel": "msoftd40372",
        "text": "fn nijofa() { return 472; }"
      },
      {
        "answer": "pijofa",
        "frame": 1,
        "id": "measure-pref-24-d1",
        "name": "Favofaqo",
        "rel": "msoftd40373",
        "text": "Residue 473 modulo 219 equals pijofa."
      },
      {
        "answer": "qijofa",
        "frame": 2,
        "id": "measure-pref-24-d2",
        "name": "Gavofaqo",
        "rel": "msoftd40374",
        "text": "printf qijofa from sector 474."
      }
    ],
    "measure-pref-25": [
      {
        "answer": "rijofa",
        "frame": 0,
        "id": "measure-pref-25-d0",
        "name": "Havofaqo",
        "rel": "msoftd40375",
        "text": "fn rijofa() { return 475; }"
      },
      {
        "answer": "sijofa",
        "frame": 1,
        "id": "measure-pref-25-d1",
        "name": "Javofaqo",
        "rel": "msoftd40376",
        "text": "Residue 476 modulo 228 equals sijofa."
      },
      {
        "answer": "tijofa",
        "frame": 2,
        "id": "measure-pref-25-d2",
        "name": "Kavofaqo",
        "rel": "msoftd40377",
        "text": "printf tijofa from sector 477."
      }
    ],
    "measure-pref-26": [
      {
        "answer": "vijofa",
        "frame": 0,
        "id": "measure-pref-26-d0",
        "name": "Lavofaqo",
        "rel": "msoftd40378",
        "text": "fn vijofa() { return 478; }"
      },
      {
        "answer": "wijofa",
        "frame": 1,
        "id": "measure-pref-26-d1",
        "name": "Mavofaqo",
        "rel": "msoftd40379",
        "text": "Residue 479 modulo 237 equals wijofa."
      },
      {
        "answer": "xijofa",
        "frame": 2,
        "id": "measure-pref-26-d2",
        "name": "Navofaqo",
        "rel": "msoftd40380",
        "text": "printf xijofa from sector 480."
      }
    ],
    "measure-pref-27": [
      {
        "answer": "yijofa",
        "frame": 0,
        "id": "measure-pref-27-d0",
        "name": "Pavofaqo",
        "rel": "msoftd40381",
        "text": "fn yijofa() { return 481; }"
      },
      {
        "answer": "zijofa",
        "frame": 1,
        "id": "measure-pref-27-d1",
        "name": "Qavofaqo",
        "rel": "msoftd40382",
        "text": "Residue 482 modulo 246 equals zijofa."
      },
      {
        "answer": "bojofa",
        "frame": 2,
        "id": "measure-pref-27-d2",
        "name": "Ravofaqo",
        "rel": "msoftd40383",
        "text": "printf bojofa from sector 483."
      }
    ],
    "measure-pref-28": [
      {
        "answer": "cojofa",
        "frame": 0,
        "id": "measure-pref-28-d0",
        "name": "Savofaqo",
        "rel": "msoftd40384",
        "text": "fn cojofa() { return 484; }"
      },
      {
        "answer": "dojofa",
        "frame": 1,
        "id": "measure-pref-28-d1",
        "name": "Tavofaqo",
        "rel": "msoftd40385",
        "text": "Residue 485 modulo 255 equals dojofa."
      },
      {
        "answer": "fojofa",
        "frame": 2,
        "id": "measure-pref-28-d2",
        "name": "Vavofaqo",
        "rel": "msoftd40386",
        "text": "printf fojofa from sector 486."
      }
    ],
    "measure-pref-29": [
      {
        "answer": "gojofa",
        "frame": 0,
        "id": "measure-pref-29-d0",
        "name": "Wavofaqo",
        "rel": "msoftd40387",
        "text": "fn gojofa() { return 487; }"
      },
      {
        "answer": "hojofa",
        "frame": 1,
        "id": "measure-pref-29-d1",
        "name": "Xavofaqo",
        "rel": "msoftd40388",
        "text": "Residue 488 modulo 264 equals hojofa."
      },
      {
        "answer": "jojofa",
        "frame": 2,
        "id": "measure-pref-29-d2",
        "name": "Yavofaqo",
        "rel": "msoftd40389",
        "text": "printf jojofa from sector 489."
      }
    ],
    "measure-pref-3": [
      {
        "answer": "nuhofa",
        "frame": 0,
        "id": "measure-pref-3-d0",
        "name": "Ditofaqo",
        "rel": "msoftd40309",
        "text": "fn nuhofa() { return 409; }"
      },
      {
        "answer": "puhofa",
        "frame": 1,
        "id": "measure-pref-3-d1",
        "name": "Fitofaqo",
        "rel": "msoftd40310",
        "text": "Residue 410 modulo 730 equals puhofa."
      },
      {
        "answer": "quhofa",
        "frame": 2,
        "id": "measure-pref-3-d2",
        "name": "Gitofaqo",
        "rel": "msoftd40311",
        "text": "printf quhofa from sector 411."
      }
    ],
    "measure-pref-30": [
      {
        "answer": "kojofa",
        "frame": 0,
        "id": "measure-pref-30-d0",
        "name": "Zavofaqo",
        "rel": "msoftd40390",
        "text": "fn kojofa() { return 490; }"
      },
      {
        "answer": "lojofa",
        "frame": 1,
        "id": "measure-pref-30-d1",
        "name": "Bevofaqo",
        "rel": "msoftd40391",
        "text": "Residue 491 modulo 273 equals lojofa."
      },
      {
        "answer": "mojofa",
        "frame": 2,
        "id": "measure-pref-30-d2",
        "name": "Cevofaqo",
        "rel": "msoftd40392",
        "text": "printf mojofa from sector 492."
      }
    ],
    "measure-pref-31": [
      {
        "answer": "nojofa",
        "frame": 0,
        "id": "measure-pref-31-d0",
        "name": "Devofaqo",
        "rel": "msoftd40393",
        "text": "fn nojofa() { return 493; }"
      },
      {
        "answer": "pojofa",
        "frame": 1,
        "id": "measure-pref-31-d1",
        "name": "Fevofaqo",
        "rel": "msoftd40394",
        "text": "Residue 494 modulo 282 equals pojofa."
      },
      {
        "answer": "qojofa",
        "frame": 2,
        "id": "measure-pref-31-d2",
        "name": "Gevofaqo",
        "rel": "msoftd40395",
        "text": "printf qojofa from sector 495."
      }
    ],
    "measure-pref-32": [
      {
        "answer": "rojofa",
        "frame": 0,
        "id": "measure-pref-32-d0",
        "name": "Hevofaqo",
        "rel": "msoftd40396",
        "text": "fn rojofa() { return 496; }"
      },
      {
        "answer": "sojofa",
        "frame": 1,
        "id": "measure-pref-32-d1",
        "name": "Jevofaqo",
        "rel": "msoftd40397",
        "text": "Residue 497 modulo 291 equals sojofa."
      },
      {
        "answer": "tojofa",
        "frame": 2,
        "id": "measure-pref-32-d2",
        "name": "Kevofaqo",
        "rel": "msoftd40398",
        "text": "printf tojofa from sector 498."
      }
    ],
    "measure-pref-33": [
      {
        "answer": "vojofa",
        "frame": 0,
        "id": "measure-pref-33-d0",
        "name": "Levofaqo",
        "rel": "msoftd40399",
        "text": "fn vojofa() { return 499; }"
      },
      {
        "answer": "wojofa",
        "frame": 1,
        "id": "measure-pref-33-d1",
        "name": "Mevofaqo",
        "rel": "msoftd40400",
        "text": "Residue 500 modulo 300 equals wojofa."
      },
      {
        "answer": "xojofa",
        "frame": 2,
        "id": "measure-pref-33-d2",
        "name": "Nevofaqo",
        "rel": "msoftd40401",
        "text": "printf xojofa from sector 501."
      }
    ],
    "measure-pref-34": [
      {
        "answer": "yojofa",
        "frame": 0,
        "id": "measure-pref-34-d0",
        "name": "Pevofaqo",
        "rel": "msoftd40402",
        "text": "fn yojofa() { return 502; }"
      },
      {
        "answer": "zojofa",
        "frame": 1,
        "id": "measure-pref-34-d1",
        "name": "Qevofaqo",
        "rel": "msoftd40403",
        "text": "Residue 503 modulo 309 equals zojofa."
      },
      {
        "answer": "bujofa",
        "frame": 2,
        "id": "measure-pref-34-d2",
        "name": "Revofaqo",
        "rel": "msoftd40404",
        "text": "printf bujofa from sector 504."
      }
    ],
    "measure-pref-35": [
      {
        "answer": "cujofa",
        "frame": 0,
        "id": "measure-pref-35-d0",
        "name": "Sevofaqo",
        "rel": "msoftd40405",
        "text": "fn cujofa() { return 505; }"
      },
      {
        "answer": "dujofa",
        "frame": 1,
        "id": "measure-pref-35-d1",
        "name": "Tevofaqo",
        "rel": "msoftd40406",
        "text": "Residue 506 modulo 318 equals dujofa."
      },
      {
        "answer": "fujofa",
        "frame": 2,
        "id": "measure-pref-35-d2",
        "name": "Vevofaqo",
        "rel": "msoftd40407",
        "text": "printf fujofa from sector 507."
      }
    ],
    "measure-pref-36": [
      {
        "answer": "gujofa",
        "frame": 0,
        "id": "measure-pref-36-d0",
        "name": "Wevofaqo",
        "rel": "msoftd40408",
        "text": "fn gujofa() { return 508; }"
      },
      {
        "answer": "hujofa",
        "frame": 1,
        "id": "measure-pref-36-d1",
        "name": "Xevofaqo",
        "rel": "msoftd40409",
        "text": "Residue 509 modulo 327 equals hujofa."
      },
      {
        "answer": "jujofa",
        "frame": 2,
        "id": "measure-pref-36-d2",
        "name": "Yevofaqo",
        "rel": "msoftd40410",
        "text": "printf jujofa from sector 510."
      }
    ],
    "measure-pref-37": [
      {
        "answer": "kujofa",
        "frame": 0,
        "id": "measure-pref-37-d0",
        "name": "Zevofaqo",
        "rel": "msoftd40411",
        "text": "fn kujofa() { return 511; }"
      },
      {
        "answer": "lujofa",
        "frame": 1,
        "id": "measure-pref-37-d1",
        "name": "Bivofaqo",
        "rel": "msoftd40412",
        "text": "Residue 512 modulo 336 equals lujofa."
      },
      {
        "answer": "mujofa",
        "frame": 2,
        "id": "measure-pref-37-d2",
        "name": "Civofaqo",
        "rel": "msoftd40413",
        "text": "printf mujofa from sector 513."
      }
    ],
    "measure-pref-38": [
      {
        "answer": "nujofa",
        "frame": 0,
        "id": "measure-pref-38-d0",
        "name": "Divofaqo",
        "rel": "msoftd40414",
        "text": "fn nujofa() { return 514; }"
      },
      {
        "answer": "pujofa",
        "frame": 1,
        "id": "measure-pref-38-d1",
        "name": "Fivofaqo",
        "rel": "msoftd40415",
        "text": "Residue 515 modulo 345 equals pujofa."
      },
      {
        "answer": "qujofa",
        "frame": 2,
        "id": "measure-pref-38-d2",
        "name": "Givofaqo",
        "rel": "msoftd40416",
        "text": "printf qujofa from sector 516."
      }
    ],
    "measure-pref-39": [
      {
        "answer": "rujofa",
        "frame": 0,
        "id": "measure-pref-39-d0",
        "name": "Hivofaqo",
        "rel": "msoftd40417",
        "text": "fn rujofa() { return 517; }"
      },
      {
        "answer": "sujofa",
        "frame": 1,
        "id": "measure-pref-39-d1",
        "name": "Jivofaqo",
        "rel": "msoftd40418",
        "text": "Residue 518 modulo 354 equals sujofa."
      },
      {
        "answer": "tujofa",
        "frame": 2,
        "id": "measure-pref-39-d2",
        "name": "Kivofaqo",
        "rel": "msoftd40419",
        "text": "printf tujofa from sector 519."
      }
    ],
    "measure-pref-4": [
      {
        "answer": "ruhofa",
        "frame": 0,
        "id": "measure-pref-4-d0",
        "name": "Hitofaqo",
        "rel": "msoftd40312",
        "text": "fn ruhofa() { return 412; }"
      },
      {
        "answer": "suhofa",
        "frame": 1,
        "id": "measure-pref-4-d1",
        "name": "Jitofaqo",
        "rel": "msoftd40313",
        "text": "Residue 413 modulo 739 equals suhofa."
      },
      {
        "answer": "tuhofa",
        "frame": 2,
        "id": "measure-pref-4-d2",
        "name": "Kitofaqo",
        "rel": "msoftd40314",
        "text": "printf tuhofa from sector 414."
      }
    ],
    "measure-pref-40": [
      {
        "answer": "vujofa",
        "frame": 0,
        "id": "measure-pref-40-d0",
        "name": "Livofaqo",
        "rel": "msoftd40420",
        "text": "fn vujofa() { return 520; }"
      },
      {
        "answer": "wujofa",
        "frame": 1,
        "id": "measure-pref-40-d1",
        "name": "Mivofaqo",
        "rel": "msoftd40421",
        "text": "Residue 521 modulo 363 equals wujofa."
      },
      {
        "answer": "xujofa",
        "frame": 2,
        "id": "measure-pref-40-d2",
        "name": "Nivofaqo",
        "rel": "msoftd40422",
        "text": "printf xujofa from sector 522."
      }
    ],
    "measure-pref-41": [
      {
        "answer": "yujofa",
        "frame": 0,
        "id": "measure-pref-41-d0",
        "name": "Pivofaqo",
        "rel": "msoftd40423",
        "text": "fn yujofa() { return 523; }"
      },
      {
        "answer": "zujofa",
        "frame": 1,
        "id": "measure-pref-41-d1",
        "name": "Qivofaqo",
        "rel": "msoftd40424",
        "text": "Residue 524 modulo 372 equals zujofa."
      },
      {
        "answer": "bakofa",
        "frame": 2,
        "id": "measure-pref-41-d2",
        "name": "Rivofaqo",
        "rel": "msoftd40425",
        "text": "printf bakofa from sector 525."
      }
    ],
    "measure-pref-42": [
      {
        "answer": "cakofa",
        "frame": 0,
        "id": "measure-pref-42-d0",
        "name": "Sivofaqo",
        "rel": "msoftd40426",
        "text": "fn cakofa() { return 526; }"
      },
      {
        "answer": "dakofa",
        "frame": 1,
        "id": "measure-pref-42-d1",
        "name": "Tivofaqo",
        "rel": "msoftd40427",
        "text": "Residue 527 modulo 381 equals dakofa."
      },
      {
        "answer": "fakofa",
        "frame": 2,
        "id": "measure-pref-42-d2",
        "name": "Vivofaqo",
        "rel": "msoftd40428",
        "text": "printf fakofa from sector 528."
      }
    ],
    "measure-pref-43": [
      {
        "answer": "gakofa",
        "frame": 0,
        "id": "measure-pref-43-d0",
        "name": "Wivofaqo",
        "rel": "msoftd40429",
        "text": "fn gakofa() { return 529; }"
      },
      {
        "answer": "hakofa",
        "frame": 1,
        "id": "measure-pref-43-d1",
        "name": "Xivofaqo",
        "rel": "msoftd40430",
        "text": "Residue 530 modulo 390 equals hakofa."
      },
      {
        "answer": "jakofa",
        "frame": 2,
        "id": "measure-pref-43-d2",
        "name": "Yivofaqo",
        "rel": "msoftd40431",
        "text": "printf jakofa from sector 531."
      }
    ],
    "measure-pref-44": [
      {
        "answer": "kakofa",
        "frame": 0,
        "id": "measure-pref-44-d0",
        "name": "Zivofaqo",
        "rel": "msoftd40432",
        "text": "fn kakofa() { return 532; }"
      },
      {
        "answer": "lakofa",
        "frame": 1,
        "id": "measure-pref-44-d1",
        "name": "Bovofaqo",
        "rel": "msoftd40433",
        "text": "Residue 533 modulo 399 equals lakofa."
      },
      {
        "answer": "makofa",
        "frame": 2,
        "id": "measure-pref-44-d2",
        "name": "Covofaqo",
        "rel": "msoftd40434",
        "text": "printf makofa from sector 534."
      }
    ],
    "measure-pref-45": [
      {
        "answer": "nakofa",
        "frame": 0,
        "id": "measure-pref-45-d0",
        "name": "Dovofaqo",
        "rel": "msoftd40435",
        "text": "fn nakofa() { return 535; }"
      },
      {
        "answer": "pakofa",
        "frame": 1,
        "id": "measure-pref-45-d1",
        "name": "Fovofaqo",
        "rel": "msoftd40436",
        "text": "Residue 536 modulo 408 equals pakofa."
      },
      {
        "answer": "qakofa",
        "frame": 2,
        "id": "measure-pref-45-d2",
        "name": "Govofaqo",
        "rel": "msoftd40437",
        "text": "printf qakofa from sector 537."
      }
    ],
    "measure-pref-46": [
      {
        "answer": "rakofa",
        "frame": 0,
        "id": "measure-pref-46-d0",
        "name": "Hovofaqo",
        "rel": "msoftd40438",
        "text": "fn rakofa() { return 538; }"
      },
      {
        "answer": "sakofa",
        "frame": 1,
        "id": "measure-pref-46-d1",
        "name": "Jovofaqo",
        "rel": "msoftd40439",
        "text": "Residue 539 modulo 417 equals sakofa."
      },
      {
        "answer": "takofa",
        "frame": 2,
        "id": "measure-pref-46-d2",
        "name": "Kovofaqo",
        "rel": "msoftd40440",
        "text": "printf takofa from sector 540."
      }
    ],
    "measure-pref-47": [
      {
        "answer": "vakofa",
        "frame": 0,
        "id": "measure-pref-47-d0",
        "name": "Lovofaqo",
        "rel": "msoftd40441",
        "text": "fn vakofa() { return 541; }"
      },
      {
        "answer": "wakofa",
        "frame": 1,
        "id": "measure-pref-47-d1",
        "name": "Movofaqo",
        "rel": "msoftd40442",
        "text": "Residue 542 modulo 426 equals wakofa."
      },
      {
        "answer": "xakofa",
        "frame": 2,
        "id": "measure-pref-47-d2",
        "name": "Novofaqo",
        "rel": "msoftd40443",
        "text": "printf xakofa from sector 543."
      }
    ],
    "measure-pref-48": [
      {
        "answer": "yakofa",
        "frame": 0,
        "id": "measure-pref-48-d0",
        "name": "Povofaqo",
        "rel": "msoftd40444",
        "text": "fn yakofa() { return 544; }"
      },
      {
        "answer": "zakofa",
        "frame": 1,
        "id": "measure-pref-48-d1",
        "name": "Qovofaqo",
        "rel": "msoftd40445",
        "text": "Residue 545 modulo 435 equals zakofa."
      },
      {
        "answer": "bekofa",
        "frame": 2,
        "id": "measure-pref-48-d2",
        "name": "Rovofaqo",
        "rel": "msoftd40446",
        "text": "printf bekofa from sector 546."
      }
    ],
    "measure-pref-49": [
      {
        "answer": "cekofa",
        "frame": 0,
        "id": "measure-pref-49-d0",
        "name": "Sovofaqo",
        "rel": "msoftd40447",
        "text": "fn cekofa() { return 547; }"
      },
      {
        "answer": "dekofa",
        "frame": 1,
        "id": "measure-pref-49-d1",
        "name": "Tovofaqo",
        "rel": "msoftd40448",
        "text": "Residue 548 modulo 444 equals dekofa."
      },
      {
        "answer": "fekofa",
        "frame": 2,
        "id": "measure-pref-49-d2",
        "name": "Vovofaqo",
        "rel": "msoftd40449",
        "text": "printf fekofa from sector 549."
      }
    ],
    "measure-pref-5": [
      {
        "answer": "vuhofa",
        "frame": 0,
        "id": "measure-pref-5-d0",
        "name": "Litofaqo",
        "rel": "msoftd40315",
        "text": "fn vuhofa() { return 415; }"
      },
      {
        "answer": "wuhofa",
        "frame": 1,
        "id": "measure-pref-5-d1",
        "name": "Mitofaqo",
        "rel": "msoftd40316",
        "text": "Residue 416 modulo 748 equals wuhofa."
      },
      {
        "answer": "xuhofa",
        "frame": 2,
        "id": "measure-pref-5-d2",
        "name": "Nitofaqo",
        "rel": "msoftd40317",
        "text": "printf xuhofa from sector 417."
      }
    ],
    "measure-pref-6": [
      {
        "answer": "yuhofa",
        "frame": 0,
        "id": "measure-pref-6-d0",
        "name": "Pitofaqo",
        "rel": "msoftd40318",
        "text": "fn yuhofa() { return 418; }"
      },
      {
        "answer": "zuhofa",
        "frame": 1,
        "id": "measure-pref-6-d1",
        "name": "Qitofaqo",
        "rel": "msoftd40319",
        "text": "Residue 419 modulo 757 equals zuhofa."
      },
      {
        "answer": "bajofa",
        "frame": 2,
        "id": "measure-pref-6-d2",
        "name": "Ritofaqo",
        "rel": "msoftd40320",
        "text": "printf bajofa from sector 420."
      }
    ],
    "measure-pref-7": [
      {
        "answer": "cajofa",
        "frame": 0,
        "id": "measure-pref-7-d0",
        "name": "Sitofaqo",
        "rel": "msoftd40321",
        "text": "fn cajofa() { return 421; }"
      },
      {
        "answer": "dajofa",
        "frame": 1,
        "id": "measure-pref-7-d1",
        "name": "Titofaqo",
        "rel": "msoftd40322",
        "text": "Residue 422 modulo 766 equals dajofa."
      },
      {
        "answer": "fajofa",
        "frame": 2,
        "id": "measure-pref-7-d2",
        "name": "Vitofaqo",
        "rel": "msoftd40323",
        "text": "printf fajofa from sector 423."
      }
    ],
    "measure-pref-8": [
      {
        "answer": "gajofa",
        "frame": 0,
        "id": "measure-pref-8-d0",
        "name": "Witofaqo",
        "rel": "msoftd40324",
        "text": "fn gajofa() { return 424; }"
      },
      {
        "answer": "hajofa",
        "frame": 1,
        "id": "measure-pref-8-d1",
        "name": "Xitofaqo",
        "rel": "msoftd40325",
        "text": "Residue 425 modulo 775 equals hajofa."
      },
      {
        "answer": "jajofa",
        "frame": 2,
        "id": "measure-pref-8-d2",
        "name": "Yitofaqo",
        "rel": "msoftd40326",
        "text": "printf jajofa from sector 426."
      }
    ],
    "measure-pref-9": [
      {
        "answer": "kajofa",
        "frame": 0,
        "id": "measure-pref-9-d0",
        "name": "Zitofaqo",
        "rel": "msoftd40327",
        "text": "fn kajofa() { return 427; }"
      },
      {
        "answer": "lajofa",
        "frame": 1,
        "id": "measure-pref-9-d1",
        "name": "Botofaqo",
        "rel": "msoftd40328",
        "text": "Residue 428 modulo 784 equals lajofa."
      },
      {
        "answer": "majofa",
        "frame": 2,
        "id": "measure-pref-9-d2",
        "name": "Cotofaqo",
        "rel": "msoftd40329",
        "text": "printf majofa from sector 429."
      }
    ],
    "measure-tone-0": [
      {
        "answer": "vudofa",
        "frame": 0,
        "id": "measure-tone-0-d0",
        "name": "Liqofaqo",
        "rel": "msoftd40000",
        "text": "fn vudofa() { return 100; }"
      },
      {
        "answer": "wudofa",
        "frame": 1,
        "id": "measure-tone-0-d1",
        "name": "Miqofaqo",
        "rel": "msoftd40001",
        "text": "Residue 101 modulo 503 equals wudofa."
      },
      {
        "answer": "xudofa",
        "frame": 2,
        "id": "measure-tone-0-d2",
        "name": "Niqofaqo",
        "rel": "msoftd40002",
        "text": "printf xudofa from sector 102."
      }
    ],
    "measure-tone-1": [
      {
        "answer": "yudofa",
        "frame": 0,
        "id": "measure-tone-1-d0",
        "name": "Piqofaqo",
        "rel": "msoftd40003",
        "text": "fn yudofa() { return 103; }"
      },
      {
        "answer": "zudofa",
        "frame": 1,
        "id": "measure-tone-1-d1",
        "name": "Qiqofaqo",
        "rel": "msoftd40004",
        "text": "Residue 104 modulo 512 equals zudofa."
      },
      {
        "answer": "bafofa",
        "frame": 2,
        "id": "measure-tone-1-d2",
        "name": "Riqofaqo",
        "rel": "msoftd40005",
        "text": "printf bafofa from sector 105."
      }
    ],
    "measure-tone-10": [
      {
        "answer": "gefofa",
        "frame": 0,
        "id": "measure-tone-10-d0",
        "name": "Woqofaqo",
        "rel": "msoftd40030",
        "text": "fn gefofa() { return 130; }"
      },
      {
        "answer": "hefofa",
        "frame": 1,
        "id": "measure-tone-10-d1",
        "name": "Xoqofaqo",
        "rel": "msoftd40031",
        "text": "Residue 131 modulo 593 equals hefofa."
      },
      {
        "answer": "jefofa",
        "frame": 2,
        "id": "measure-tone-10-d2",
        "name": "Yoqofaqo",
        "rel": "msoftd40032",
        "text": "printf jefofa from sector 132."
      }
    ],
    "measure-tone-11": [
      {
        "answer": "kefofa",
        "frame": 0,
        "id": "measure-tone-11-d0",
        "name": "Zoqofaqo",
        "rel": "msoftd40033",
        "text": "fn kefofa() { return 133; }"
      },
      {
        "answer": "lefofa",
        "frame": 1,
        "id": "measure-tone-11-d1",
        "name": "Buqofaqo",
        "rel": "msoftd40034",
        "text": "Residue 134 modulo 602 equals lefofa."
      },
      {
        "answer": "mefofa",
        "frame": 2,
        "id": "measure-tone-11-d2",
        "name": "Cuqofaqo",
        "rel": "msoftd40035",
        "text": "printf mefofa from sector 135."
      }
    ],
    "measure-tone-12": [
      {
        "answer": "nefofa",
        "frame": 0,
        "id": "measure-tone-12-d0",
        "name": "Duqofaqo",
        "rel": "msoftd40036",
        "text": "fn nefofa() { return 136; }"
      },
      {
        "answer": "nijuqa",
        "frame": 3,
        "id": "measure-tone-12-d1",
        "name": "Fucilaqo",
        "rel": "msoftd91802",
        "text": "0x702 xor 0x506 equals nijuqa."
      },
      {
        "answer": "qefofa",
        "frame": 2,
        "id": "measure-tone-12-d2",
        "name": "Guqofaqo",
        "rel": "msoftd40038",
        "text": "printf qefofa from sector 138."
      }
    ],
    "measure-tone-13": [
      {
        "answer": "refofa",
        "frame": 0,
        "id": "measure-tone-13-d0",
        "name": "Huqofaqo",
        "rel": "msoftd40039",
        "text": "fn refofa() { return 139; }"
      },
      {
        "answer": "sefofa",
        "frame": 1,
        "id": "measure-tone-13-d1",
        "name": "Juqofaqo",
        "rel": "msoftd40040",
        "text": "Residue 140 modulo 620 equals sefofa."
      },
      {
        "answer": "tefofa",
        "frame": 2,
        "id": "measure-tone-13-d2",
        "name": "Kuqofaqo",
        "rel": "msoftd40041",
        "text": "printf tefofa from sector 141."
      }
    ],
    "measure-tone-14": [
      {
        "answer": "vefofa",
        "frame": 0,
        "id": "measure-tone-14-d0",
        "name": "Luqofaqo",
        "rel": "msoftd40042",
        "text": "fn vefofa() { return 142; }"
      },
      {
        "answer": "wefofa",
        "frame": 1,
        "id": "measure-tone-14-d1",
        "name": "Muqofaqo",
        "rel": "msoftd40043",
        "text": "Residue 143 modulo 629 equals wefofa."
      },
      {
        "answer": "xefofa",
        "frame": 2,
        "id": "measure-tone-14-d2",
        "name": "Nuqofaqo",
        "rel": "msoftd40044",
        "text": "printf xefofa from sector 144."
      }
    ],
    "measure-tone-15": [
      {
        "answer": "yefofa",
        "frame": 0,
        "id": "measure-tone-15-d0",
        "name": "Puqofaqo",
        "rel": "msoftd40045",
        "text": "fn yefofa() { return 145; }"
      },
      {
        "answer": "wekuqa",
        "frame": 3,
        "id": "measure-tone-15-d1",
        "name": "Nodilaqo",
        "rel": "msoftd91893",
        "text": "0x793 xor 0x779 equals wekuqa."
      },
      {
        "answer": "bifofa",
        "frame": 2,
        "id": "measure-tone-15-d2",
        "name": "Ruqofaqo",
        "rel": "msoftd40047",
        "text": "printf bifofa from sector 147."
      }
    ],
    "measure-tone-16": [
      {
        "answer": "cifofa",
        "frame": 0,
        "id": "measure-tone-16-d0",
        "name": "Suqofaqo",
        "rel": "msoftd40048",
        "text": "fn cifofa() { return 148; }"
      },
      {
        "answer": "difofa",
        "frame": 1,
        "id": "measure-tone-16-d1",
        "name": "Tuqofaqo",
        "rel": "msoftd40049",
        "text": "Residue 149 modulo 647 equals difofa."
      },
      {
        "answer": "fifofa",
        "frame": 2,
        "id": "measure-tone-16-d2",
        "name": "Vuqofaqo",
        "rel": "msoftd40050",
        "text": "printf fifofa from sector 150."
      }
    ],
    "measure-tone-17": [
      {
        "answer": "gifofa",
        "frame": 0,
        "id": "measure-tone-17-d0",
        "name": "Wuqofaqo",
        "rel": "msoftd40051",
        "text": "fn gifofa() { return 151; }"
      },
      {
        "answer": "tukuqa",
        "frame": 3,
        "id": "measure-tone-17-d1",
        "name": "Lefilaqo",
        "rel": "msoftd91954",
        "text": "0x854 xor 0x262 equals tukuqa."
      },
      {
        "answer": "jifofa",
        "frame": 2,
        "id": "measure-tone-17-d2",
        "name": "Yuqofaqo",
        "rel": "msoftd40053",
        "text": "printf jifofa from sector 153."
      }
    ],
    "measure-tone-18": [
      {
        "answer": "kifofa",
        "frame": 0,
        "id": "measure-tone-18-d0",
        "name": "Zuqofaqo",
        "rel": "msoftd40054",
        "text": "fn kifofa() { return 154; }"
      },
      {
        "answer": "lifofa",
        "frame": 1,
        "id": "measure-tone-18-d1",
        "name": "Barofaqo",
        "rel": "msoftd40055",
        "text": "Residue 155 modulo 665 equals lifofa."
      },
      {
        "answer": "mifofa",
        "frame": 2,
        "id": "measure-tone-18-d2",
        "name": "Carofaqo",
        "rel": "msoftd40056",
        "text": "printf mifofa from sector 156."
      }
    ],
    "measure-tone-19": [
      {
        "answer": "nifofa",
        "frame": 0,
        "id": "measure-tone-19-d0",
        "name": "Darofaqo",
        "rel": "msoftd40057",
        "text": "fn nifofa() { return 157; }"
      },
      {
        "answer": "pifofa",
        "frame": 1,
        "id": "measure-tone-19-d1",
        "name": "Farofaqo",
        "rel": "msoftd40058",
        "text": "Residue 158 modulo 674 equals pifofa."
      },
      {
        "answer": "qifofa",
        "frame": 2,
        "id": "measure-tone-19-d2",
        "name": "Garofaqo",
        "rel": "msoftd40059",
        "text": "printf qifofa from sector 159."
      }
    ],
    "measure-tone-2": [
      {
        "answer": "cafofa",
        "frame": 0,
        "id": "measure-tone-2-d0",
        "name": "Siqofaqo",
        "rel": "msoftd40006",
        "text": "fn cafofa() { return 106; }"
      },
      {
        "answer": "dafofa",
        "frame": 1,
        "id": "measure-tone-2-d1",
        "name": "Tiqofaqo",
        "rel": "msoftd40007",
        "text": "Residue 107 modulo 521 equals dafofa."
      },
      {
        "answer": "fafofa",
        "frame": 2,
        "id": "measure-tone-2-d2",
        "name": "Viqofaqo",
        "rel": "msoftd40008",
        "text": "printf fafofa from sector 108."
      }
    ],
    "measure-tone-20": [
      {
        "answer": "rifofa",
        "frame": 0,
        "id": "measure-tone-20-d0",
        "name": "Harofaqo",
        "rel": "msoftd40060",
        "text": "fn rifofa() { return 160; }"
      },
      {
        "answer": "sifofa",
        "frame": 1,
        "id": "measure-tone-20-d1",
        "name": "Jarofaqo",
        "rel": "msoftd40061",
        "text": "Residue 161 modulo 683 equals sifofa."
      },
      {
        "answer": "tifofa",
        "frame": 2,
        "id": "measure-tone-20-d2",
        "name": "Karofaqo",
        "rel": "msoftd40062",
        "text": "printf tifofa from sector 162."
      }
    ],
    "measure-tone-21": [
      {
        "answer": "vifofa",
        "frame": 0,
        "id": "measure-tone-21-d0",
        "name": "Larofaqo",
        "rel": "msoftd40063",
        "text": "fn vifofa() { return 163; }"
      },
      {
        "answer": "wifofa",
        "frame": 1,
        "id": "measure-tone-21-d1",
        "name": "Marofaqo",
        "rel": "msoftd40064",
        "text": "Residue 164 modulo 692 equals wifofa."
      },
      {
        "answer": "xifofa",
        "frame": 2,
        "id": "measure-tone-21-d2",
        "name": "Narofaqo",
        "rel": "msoftd40065",
        "text": "printf xifofa from sector 165."
      }
    ],
    "measure-tone-22": [
      {
        "answer": "yifofa",
        "frame": 0,
        "id": "measure-tone-22-d0",
        "name": "Parofaqo",
        "rel": "msoftd40066",
        "text": "fn yifofa() { return 166; }"
      },
      {
        "answer": "zifofa",
        "frame": 1,
        "id": "measure-tone-22-d1",
        "name": "Qarofaqo",
        "rel": "msoftd40067",
        "text": "Residue 167 modulo 701 equals zifofa."
      },
      {
        "answer": "bofofa",
        "frame": 2,
        "id": "measure-tone-22-d2",
        "name": "Rarofaqo",
        "rel": "msoftd40068",
        "text": "printf bofofa from sector 168."
      }
    ],
    "measure-tone-23": [
      {
        "answer": "cofofa",
        "frame": 0,
        "id": "measure-tone-23-d0",
        "name": "Sarofaqo",
        "rel": "msoftd40069",
        "text": "fn cofofa() { return 169; }"
      },
      {
        "answer": "dofofa",
        "frame": 1,
        "id": "measure-tone-23-d1",
        "name": "Tarofaqo",
        "rel": "msoftd40070",
        "text": "Residue 170 modulo 710 equals dofofa."
      },
      {
        "answer": "fofofa",
        "frame": 2,
        "id": "measure-tone-23-d2",
        "name": "Varofaqo",
        "rel": "msoftd40071",
        "text": "printf fofofa from sector 171."
      }
    ],
    "measure-tone-24": [
      {
        "answer": "gofofa",
        "frame": 0,
        "id": "measure-tone-24-d0",
        "name": "Warofaqo",
        "rel": "msoftd40072",
        "text": "fn gofofa() { return 172; }"
      },
      {
        "answer": "hofofa",
        "frame": 1,
        "id": "measure-tone-24-d1",
        "name": "Xarofaqo",
        "rel": "msoftd40073",
        "text": "Residue 173 modulo 719 equals hofofa."
      },
      {
        "answer": "jofofa",
        "frame": 2,
        "id": "measure-tone-24-d2",
        "name": "Yarofaqo",
        "rel": "msoftd40074",
        "text": "printf jofofa from sector 174."
      }
    ],
    "measure-tone-25": [
      {
        "answer": "kofofa",
        "frame": 0,
        "id": "measure-tone-25-d0",
        "name": "Zarofaqo",
        "rel": "msoftd40075",
        "text": "fn kofofa() { return 175; }"
      },
      {
        "answer": "lofofa",
        "frame": 1,
        "id": "measure-tone-25-d1",
        "name": "Berofaqo",
        "rel": "msoftd40076",
        "text": "Residue 176 modulo 728 equals lofofa."
      },
      {
        "answer": "mofofa",
        "frame": 2,
        "id": "measure-tone-25-d2",
        "name": "Cerofaqo",
        "rel": "msoftd40077",
        "text": "printf mofofa from sector 177."
      }
    ],
    "measure-tone-26": [
      {
        "answer": "nofofa",
        "frame": 0,
        "id": "measure-tone-26-d0",
        "name": "Derofaqo",
        "rel": "msoftd40078",
        "text": "fn nofofa() { return 178; }"
      },
      {
        "answer": "pofofa",
        "frame": 1,
        "id": "measure-tone-26-d1",
        "name": "Ferofaqo",
        "rel": "msoftd40079",
        "text": "Residue 179 modulo 737 equals pofofa."
      },
      {
        "answer": "qofofa",
        "frame": 2,
        "id": "measure-tone-26-d2",
        "name": "Gerofaqo",
        "rel": "msoftd40080",
        "text": "printf qofofa from sector 180."
      }
    ],
    "measure-tone-27": [
      {
        "answer": "rofofa",
        "frame": 0,
        "id": "measure-tone-27-d0",
        "name": "Herofaqo",
        "rel": "msoftd40081",
        "text": "fn rofofa() { return 181; }"
      },
      {
        "answer": "cunuqa",
        "frame": 3,
        "id": "measure-tone-27-d1",
        "name": "Tajilaqo",
        "rel": "msoftd92255",
        "text": "0x355 xor 0x465 equals cunuqa."
      },
      {
        "answer": "tofofa",
        "frame": 2,
        "id": "measure-tone-27-d2",
        "name": "Kerofaqo",
        "rel": "msoftd40083",
        "text": "printf tofofa from sector 183."
      }
    ],
    "measure-tone-28": [
      {
        "answer": "vofofa",
        "frame": 0,
        "id": "measure-tone-28-d0",
        "name": "Lerofaqo",
        "rel": "msoftd40084",
        "text": "fn vofofa() { return 184; }"
      },
      {
        "answer": "wofofa",
        "frame": 1,
        "id": "measure-tone-28-d1",
        "name": "Merofaqo",
        "rel": "msoftd40085",
        "text": "Residue 185 modulo 755 equals wofofa."
      },
      {
        "answer": "xofofa",
        "frame": 2,
        "id": "measure-tone-28-d2",
        "name": "Nerofaqo",
        "rel": "msoftd40086",
        "text": "printf xofofa from sector 186."
      }
    ],
    "measure-tone-29": [
      {
        "answer": "yofofa",
        "frame": 0,
        "id": "measure-tone-29-d0",
        "name": "Perofaqo",
        "rel": "msoftd40087",
        "text": "fn yofofa() { return 187; }"
      },
      {
        "answer": "zofofa",
        "frame": 1,
        "id": "measure-tone-29-d1",
        "name": "Qerofaqo",
        "rel": "msoftd40088",
        "text": "Residue 188 modulo 764 equals zofofa."
      },
      {
        "answer": "bufofa",
        "frame": 2,
        "id": "measure-tone-29-d2",
        "name": "Rerofaqo",
        "rel": "msoftd40089",
        "text": "printf bufofa from sector 189."
      }
    ],
    "measure-tone-3": [
      {
        "answer": "gafofa",
        "frame": 0,
        "id": "measure-tone-3-d0",
        "name": "Wiqofaqo",
        "rel": "msoftd40009",
        "text": "fn gafofa() { return 109; }"
      },
      {
        "answer": "hafofa",
        "frame": 1,
        "id": "measure-tone-3-d1",
        "name": "Xiqofaqo",
        "rel": "msoftd40010",
        "text": "Residue 110 modulo 530 equals hafofa."
      },
      {
        "answer": "jafofa",
        "frame": 2,
        "id": "measure-tone-3-d2",
        "name": "Yiqofaqo",
        "rel": "msoftd40011",
        "text": "printf jafofa from sector 111."
      }
    ],
    "measure-tone-30": [
      {
        "answer": "cufofa",
        "frame": 0,
        "id": "measure-tone-30-d0",
        "name": "Serofaqo",
        "rel": "msoftd40090",
        "text": "fn cufofa() { return 190; }"
      },
      {
        "answer": "dufofa",
        "frame": 1,
        "id": "measure-tone-30-d1",
        "name": "Terofaqo",
        "rel": "msoftd40091",
        "text": "Residue 191 modulo 773 equals dufofa."
      },
      {
        "answer": "fufofa",
        "frame": 2,
        "id": "measure-tone-30-d2",
        "name": "Verofaqo",
        "rel": "msoftd40092",
        "text": "printf fufofa from sector 192."
      }
    ],
    "measure-tone-31": [
      {
        "answer": "gufofa",
        "frame": 0,
        "id": "measure-tone-31-d0",
        "name": "Werofaqo",
        "rel": "msoftd40093",
        "text": "fn gufofa() { return 193; }"
      },
      {
        "answer": "hufofa",
        "frame": 1,
        "id": "measure-tone-31-d1",
        "name": "Xerofaqo",
        "rel": "msoftd40094",
        "text": "Residue 194 modulo 782 equals hufofa."
      },
      {
        "answer": "jufofa",
        "frame": 2,
        "id": "measure-tone-31-d2",
        "name": "Yerofaqo",
        "rel": "msoftd40095",
        "text": "printf jufofa from sector 195."
      }
    ],
    "measure-tone-32": [
      {
        "answer": "kufofa",
        "frame": 0,
        "id": "measure-tone-32-d0",
        "name": "Zerofaqo",
        "rel": "msoftd40096",
        "text": "fn kufofa() { return 196; }"
      },
      {
        "answer": "lufofa",
        "frame": 1,
        "id": "measure-tone-32-d1",
        "name": "Birofaqo",
        "rel": "msoftd40097",
        "text": "Residue 197 modulo 791 equals lufofa."
      },
      {
        "answer": "mufofa",
        "frame": 2,
        "id": "measure-tone-32-d2",
        "name": "Cirofaqo",
        "rel": "msoftd40098",
        "text": "printf mufofa from sector 198."
      }
    ],
    "measure-tone-33": [
      {
        "answer": "nufofa",
        "frame": 0,
        "id": "measure-tone-33-d0",
        "name": "Dirofaqo",
        "rel": "msoftd40099",
        "text": "fn nufofa() { return 199; }"
      },
      {
        "answer": "pufofa",
        "frame": 1,
        "id": "measure-tone-33-d1",
        "name": "Firofaqo",
        "rel": "msoftd40100",
        "text": "Residue 200 modulo 800 equals pufofa."
      },
      {
        "answer": "qufofa",
        "frame": 2,
        "id": "measure-tone-33-d2",
        "name": "Girofaqo",
        "rel": "msoftd40101",
        "text": "printf qufofa from sector 201."
      }
    ],
    "measure-tone-34": [
      {
        "answer": "rufofa",
        "frame": 0,
        "id": "measure-tone-34-d0",
        "name": "Hirofaqo",
        "rel": "msoftd40102",
        "text": "fn rufofa() { return 202; }"
      },
      {
        "answer": "sufofa",
        "frame": 1,
        "id": "measure-tone-34-d1",
        "name": "Jirofaqo",
        "rel": "msoftd40103",
        "text": "Residue 203 modulo 809 equals sufofa."
      },
      {
        "answer": "tufofa",
        "frame": 2,
        "id": "measure-tone-34-d2",
        "name": "Kirofaqo",
        "rel": "msoftd40104",
        "text": "printf tufofa from sector 204."
      }
    ],
    "measure-tone-35": [
      {
        "answer": "vufofa",
        "frame": 0,
        "id": "measure-tone-35-d0",
        "name": "Lirofaqo",
        "rel": "msoftd40105",
        "text": "fn vufofa() { return 205; }"
      },
      {
        "answer": "wufofa",
        "frame": 1,
        "id": "measure-tone-35-d1",
        "name": "Mirofaqo",
        "rel": "msoftd40106",
        "text": "Residue 206 modulo 818 equals wufofa."
      },
      {
        "answer": "xufofa",
        "frame": 2,
        "id": "measure-tone-35-d2",
        "name": "Nirofaqo",
        "rel": "msoftd40107",
        "text": "printf xufofa from sector 207."
      }
    ],
    "measure-tone-36": [
      {
        "answer": "yufofa",
        "frame": 0,
        "id": "measure-tone-36-d0",
        "name": "Pirofaqo",
        "rel": "msoftd40108",
        "text": "fn yufofa() { return 208; }"
      },
      {
        "answer": "zufofa",
        "frame": 1,
        "id": "measure-tone-36-d1",
        "name": "Qirofaqo",
        "rel": "msoftd40109",
        "text": "Residue 209 modulo 827 equals zufofa."
      },
      {
        "answer": "bagofa",
        "frame": 2,
        "id": "measure-tone-36-d2",
        "name": "Rirofaqo",
        "rel": "msoftd40110",
        "text": "printf bagofa from sector 210."
      }
    ],
    "measure-tone-37": [
      {
        "answer": "cagofa",
        "frame": 0,
        "id": "measure-tone-37-d0",
        "name": "Sirofaqo",
        "rel": "msoftd40111",
        "text": "fn cagofa() { return 211; }"
      },
      {
        "answer": "dagofa",
        "frame": 1,
        "id": "measure-tone-37-d1",
        "name": "Tirofaqo",
        "rel": "msoftd40112",
        "text": "Residue 212 modulo 836 equals dagofa."
      },
      {
        "answer": "fagofa",
        "frame": 2,
        "id": "measure-tone-37-d2",
        "name": "Virofaqo",
        "rel": "msoftd40113",
        "text": "printf fagofa from sector 213."
      }
    ],
    "measure-tone-38": [
      {
        "answer": "gagofa",
        "frame": 0,
        "id": "measure-tone-38-d0",
        "name": "Wirofaqo",
        "rel": "msoftd40114",
        "text": "fn gagofa() { return 214; }"
      },
      {
        "answer": "hagofa",
        "frame": 1,
        "id": "measure-tone-38-d1",
        "name": "Xirofaqo",
        "rel": "msoftd40115",
        "text": "Residue 215 modulo 845 equals hagofa."
      },
      {
        "answer": "jagofa",
        "frame": 2,
        "id": "measure-tone-38-d2",
        "name": "Yirofaqo",
        "rel": "msoftd40116",
        "text": "printf jagofa from sector 216."
      }
    ],
    "measure-tone-39": [
      {
        "answer": "kagofa",
        "frame": 0,
        "id": "measure-tone-39-d0",
        "name": "Zirofaqo",
        "rel": "msoftd40117",
        "text": "fn kagofa() { return 217; }"
      },
      {
        "answer": "lagofa",
        "frame": 1,
        "id": "measure-tone-39-d1",
        "name": "Borofaqo",
        "rel": "msoftd40118",
        "text": "Residue 218 modulo 854 equals lagofa."
      },
      {
        "answer": "magofa",
        "frame": 2,
        "id": "measure-tone-39-d2",
        "name": "Corofaqo",
        "rel": "msoftd40119",
        "text": "printf magofa from sector 219."
      }
    ],
    "measure-tone-4": [
      {
        "answer": "kafofa",
        "frame": 0,
        "id": "measure-tone-4-d0",
        "name": "Ziqofaqo",
        "rel": "msoftd40012",
        "text": "fn kafofa() { return 112; }"
      },
      {
        "answer": "lafofa",
        "frame": 1,
        "id": "measure-tone-4-d1",
        "name": "Boqofaqo",
        "rel": "msoftd40013",
        "text": "Residue 113 modulo 539 equals lafofa."
      },
      {
        "answer": "mafofa",
        "frame": 2,
        "id": "measure-tone-4-d2",
        "name": "Coqofaqo",
        "rel": "msoftd40014",
        "text": "printf mafofa from sector 114."
      }
    ],
    "measure-tone-40": [
      {
        "answer": "nagofa",
        "frame": 0,
        "id": "measure-tone-40-d0",
        "name": "Dorofaqo",
        "rel": "msoftd40120",
        "text": "fn nagofa() { return 220; }"
      },
      {
        "answer": "pagofa",
        "frame": 1,
        "id": "measure-tone-40-d1",
        "name": "Forofaqo",
        "rel": "msoftd40121",
        "text": "Residue 221 modulo 863 equals pagofa."
      },
      {
        "answer": "qagofa",
        "frame": 2,
        "id": "measure-tone-40-d2",
        "name": "Gorofaqo",
        "rel": "msoftd40122",
        "text": "printf qagofa from sector 222."
      }
    ],
    "measure-tone-41": [
      {
        "answer": "ragofa",
        "frame": 0,
        "id": "measure-tone-41-d0",
        "name": "Horofaqo",
        "rel": "msoftd40123",
        "text": "fn ragofa() { return 223; }"
      },
      {
        "answer": "sagofa",
        "frame": 1,
        "id": "measure-tone-41-d1",
        "name": "Jorofaqo",
        "rel": "msoftd40124",
        "text": "Residue 224 modulo 872 equals sagofa."
      },
      {
        "answer": "tagofa",
        "frame": 2,
        "id": "measure-tone-41-d2",
        "name": "Korofaqo",
        "rel": "msoftd40125",
        "text": "printf tagofa from sector 225."
      }
    ],
    "measure-tone-42": [
      {
        "answer": "vagofa",
        "frame": 0,
        "id": "measure-tone-42-d0",
        "name": "Lorofaqo",
        "rel": "msoftd40126",
        "text": "fn vagofa() { return 226; }"
      },
      {
        "answer": "wagofa",
        "frame": 1,
        "id": "measure-tone-42-d1",
        "name": "Morofaqo",
        "rel": "msoftd40127",
        "text": "Residue 227 modulo 881 equals wagofa."
      },
      {
        "answer": "xagofa",
        "frame": 2,
        "id": "measure-tone-42-d2",
        "name": "Norofaqo",
        "rel": "msoftd40128",
        "text": "printf xagofa from sector 228."
      }
    ],
    "measure-tone-43": [
      {
        "answer": "yagofa",
        "frame": 0,
        "id": "measure-tone-43-d0",
        "name": "Porofaqo",
        "rel": "msoftd40129",
        "text": "fn yagofa() { return 229; }"
      },
      {
        "answer": "zagofa",
        "frame": 1,
        "id": "measure-tone-43-d1",
        "name": "Qorofaqo",
        "rel": "msoftd40130",
        "text": "Residue 230 modulo 890 equals zagofa."
      },
      {
        "answer": "begofa",
        "frame": 2,
        "id": "measure-tone-43-d2",
        "name": "Rorofaqo",
        "rel": "msoftd40131",
        "text": "printf begofa from sector 231."
      }
    ],
    "measure-tone-44": [
      {
        "answer": "cegofa",
        "frame": 0,
        "id": "measure-tone-44-d0",
        "name": "Sorofaqo",
        "rel": "msoftd40132",
        "text": "fn cegofa() { return 232; }"
      },
      {
        "answer": "degofa",
        "frame": 1,
        "id": "measure-tone-44-d1",
        "name": "Torofaqo",
        "rel": "msoftd40133",
        "text": "Residue 233 modulo 899 equals degofa."
      },
      {
        "answer": "fegofa",
        "frame": 2,
        "id": "measure-tone-44-d2",
        "name": "Vorofaqo",
        "rel": "msoftd40134",
        "text": "printf fegofa from sector 234."
      }
    ],
    "measure-tone-45": [
      {
        "answer": "gegofa",
        "frame": 0,
        "id": "measure-tone-45-d0",
        "name": "Worofaqo",
        "rel": "msoftd40135",
        "text": "fn gegofa() { return 235; }"
      },
      {
        "answer": "hegofa",
        "frame": 1,
        "id": "measure-tone-45-d1",
        "name": "Xorofaqo",
        "rel": "msoftd40136",
        "text": "Residue 236 modulo 208 equals hegofa."
      },
      {
        "answer": "jegofa",
        "frame": 2,
        "id": "measure-tone-45-d2",
        "name": "Yorofaqo",
        "rel": "msoftd40137",
        "text": "printf jegofa from sector 237."
      }
    ],
    "measure-tone-46": [
      {
        "answer": "kegofa",
        "frame": 0,
        "id": "measure-tone-46-d0",
        "name": "Zorofaqo",
        "rel": "msoftd40138",
        "text": "fn kegofa() { return 238; }"
      },
      {
        "answer": "legofa",
        "frame": 1,
        "id": "measure-tone-46-d1",
        "name": "Burofaqo",
        "rel": "msoftd40139",
        "text": "Residue 239 modulo 217 equals legofa."
      },
      {
        "answer": "megofa",
        "frame": 2,
        "id": "measure-tone-46-d2",
        "name": "Curofaqo",
        "rel": "msoftd40140",
        "text": "printf megofa from sector 240."
      }
    ],
    "measure-tone-47": [
      {
        "answer": "negofa",
        "frame": 0,
        "id": "measure-tone-47-d0",
        "name": "Durofaqo",
        "rel": "msoftd40141",
        "text": "fn negofa() { return 241; }"
      },
      {
        "answer": "pegofa",
        "frame": 1,
        "id": "measure-tone-47-d1",
        "name": "Furofaqo",
        "rel": "msoftd40142",
        "text": "Residue 242 modulo 226 equals pegofa."
      },
      {
        "answer": "qegofa",
        "frame": 2,
        "id": "measure-tone-47-d2",
        "name": "Gurofaqo",
        "rel": "msoftd40143",
        "text": "printf qegofa from sector 243."
      }
    ],
    "measure-tone-48": [
      {
        "answer": "regofa",
        "frame": 0,
        "id": "measure-tone-48-d0",
        "name": "Hurofaqo",
        "rel": "msoftd40144",
        "text": "fn regofa() { return 244; }"
      },
      {
        "answer": "segofa",
        "frame": 1,
        "id": "measure-tone-48-d1",
        "name": "Jurofaqo",
        "rel": "msoftd40145",
        "text": "Residue 245 modulo 235 equals segofa."
      },
      {
        "answer": "tegofa",
        "frame": 2,
        "id": "measure-tone-48-d2",
        "name": "Kurofaqo",
        "rel": "msoftd40146",
        "text": "printf tegofa from sector 246."
      }
    ],
    "measure-tone-49": [
      {
        "answer": "vegofa",
        "frame": 0,
        "id": "measure-tone-49-d0",
        "name": "Lurofaqo",
        "rel": "msoftd40147",
        "text": "fn vegofa() { return 247; }"
      },
      {
        "answer": "wegofa",
        "frame": 1,
        "id": "measure-tone-49-d1",
        "name": "Murofaqo",
        "rel": "msoftd40148",
        "text": "Residue 248 modulo 244 equals wegofa."
      },
      {
        "answer": "xegofa",
        "frame": 2,
        "id": "measure-tone-49-d2",
        "name": "Nurofaqo",
        "rel": "msoftd40149",
        "text": "printf xegofa from sector 249."
      }
    ],
    "measure-tone-5": [
      {
        "answer": "nafofa",
        "frame": 0,
        "id": "measure-tone-5-d0",
        "name": "Doqofaqo",
        "rel": "msoftd40015",
        "text": "fn nafofa() { return 115; }"
      },
      {
        "answer": "pafofa",
        "frame": 1,
        "id": "measure-tone-5-d1",
        "name": "Foqofaqo",
        "rel": "msoftd40016",
        "text": "Residue 116 modulo 548 equals pafofa."
      },
      {
        "answer": "qafofa",
        "frame": 2,
        "id": "measure-tone-5-d2",
        "name": "Goqofaqo",
        "rel": "msoftd40017",
        "text": "printf qafofa from sector 117."
      }
    ],
    "measure-tone-6": [
      {
        "answer": "rafofa",
        "frame": 0,
        "id": "measure-tone-6-d0",
        "name": "Hoqofaqo",
        "rel": "msoftd40018",
        "text": "fn rafofa() { return 118; }"
      },
      {
        "answer": "safofa",
        "frame": 1,
        "id": "measure-tone-6-d1",
        "name": "Joqofaqo",
        "rel": "msoftd40019",
        "text": "Residue 119 modulo 557 equals safofa."
      },
      {
        "answer": "tafofa",
        "frame": 2,
        "id": "measure-tone-6-d2",
        "name": "Koqofaqo",
        "rel": "msoftd40020",
        "text": "printf tafofa from sector 120."
      }
    ],
    "measure-tone-7": [
      {
        "answer": "vafofa",
        "frame": 0,
        "id": "measure-tone-7-d0",
        "name": "Loqofaqo",
        "rel": "msoftd40021",
        "text": "fn vafofa() { return 121; }"
      },
      {
        "answer": "wafofa",
        "frame": 1,
        "id": "measure-tone-7-d1",
        "name": "Moqofaqo",
        "rel": "msoftd40022",
        "text": "Residue 122 modulo 566 equals wafofa."
      },
      {
        "answer": "xafofa",
        "frame": 2,
        "id": "measure-tone-7-d2",
        "name": "Noqofaqo",
        "rel": "msoftd40023",
        "text": "printf xafofa from sector 123."
      }
    ],
    "measure-tone-8": [
      {
        "answer": "yafofa",
        "frame": 0,
        "id": "measure-tone-8-d0",
        "name": "Poqofaqo",
        "rel": "msoftd40024",
        "text": "fn yafofa() { return 124; }"
      },
      {
        "answer": "zafofa",
        "frame": 1,
        "id": "measure-tone-8-d1",
        "name": "Qoqofaqo",
        "rel": "msoftd40025",
        "text": "Residue 125 modulo 575 equals zafofa."
      },
      {
        "answer": "befofa",
        "frame": 2,
        "id": "measure-tone-8-d2",
        "name": "Roqofaqo",
        "rel": "msoftd40026",
        "text": "printf befofa from sector 126."
      }
    ],
    "measure-tone-9": [
      {
        "answer": "cefofa",
        "frame": 0,
        "id": "measure-tone-9-d0",
        "name": "Soqofaqo",
        "rel": "msoftd40027",
        "text": "fn cefofa() { return 127; }"
      },
      {
        "answer": "fohuqa",
        "frame": 3,
        "id": "measure-tone-9-d1",
        "name": "Wubilaqo",
        "rel": "msoftd91711",
        "text": "0x611 xor 0x233 equals fohuqa."
      },
      {
        "answer": "fefofa",
        "frame": 2,
        "id": "measure-tone-9-d2",
        "name": "Voqofaqo",
        "rel": "msoftd40029",
        "text": "printf fefofa from sector 129."
      }
    ],
    "softmix-fit-choice-0": [
      {
        "answer": "sucuca",
        "frame": 0,
        "id": "softmix-fit-choice-0-d0",
        "name": "Jipucaqo",
        "rel": "msoftd20048",
        "text": "fn sucuca() { return 148; }"
      },
      {
        "answer": "tucuca",
        "frame": 1,
        "id": "softmix-fit-choice-0-d1",
        "name": "Kipucaqo",
        "rel": "msoftd20049",
        "text": "Residue 149 modulo 847 equals tucuca."
      },
      {
        "answer": "vucuca",
        "frame": 2,
        "id": "softmix-fit-choice-0-d2",
        "name": "Lipucaqo",
        "rel": "msoftd20050",
        "text": "printf vucuca from sector 150."
      }
    ],
    "softmix-fit-choice-1": [
      {
        "answer": "wucuca",
        "frame": 0,
        "id": "softmix-fit-choice-1-d0",
        "name": "Mipucaqo",
        "rel": "msoftd20051",
        "text": "fn wucuca() { return 151; }"
      },
      {
        "answer": "xucuca",
        "frame": 1,
        "id": "softmix-fit-choice-1-d1",
        "name": "Nipucaqo",
        "rel": "msoftd20052",
        "text": "Residue 152 modulo 856 equals xucuca."
      },
      {
        "answer": "yucuca",
        "frame": 2,
        "id": "softmix-fit-choice-1-d2",
        "name": "Pipucaqo",
        "rel": "msoftd20053",
        "text": "printf yucuca from sector 153."
      }
    ],
    "softmix-fit-choice-10": [
      {
        "answer": "deduca",
        "frame": 0,
        "id": "softmix-fit-choice-10-d0",
        "name": "Topucaqo",
        "rel": "msoftd20078",
        "text": "fn deduca() { return 178; }"
      },
      {
        "answer": "feduca",
        "frame": 1,
        "id": "softmix-fit-choice-10-d1",
        "name": "Vopucaqo",
        "rel": "msoftd20079",
        "text": "Residue 179 modulo 237 equals feduca."
      },
      {
        "answer": "geduca",
        "frame": 2,
        "id": "softmix-fit-choice-10-d2",
        "name": "Wopucaqo",
        "rel": "msoftd20080",
        "text": "printf geduca from sector 180."
      }
    ],
    "softmix-fit-choice-11": [
      {
        "answer": "heduca",
        "frame": 0,
        "id": "softmix-fit-choice-11-d0",
        "name": "Xopucaqo",
        "rel": "msoftd20081",
        "text": "fn heduca() { return 181; }"
      },
      {
        "answer": "jeduca",
        "frame": 1,
        "id": "softmix-fit-choice-11-d1",
        "name": "Yopucaqo",
        "rel": "msoftd20082",
        "text": "Residue 182 modulo 246 equals jeduca."
      },
      {
        "answer": "keduca",
        "frame": 2,
        "id": "softmix-fit-choice-11-d2",
        "name": "Zopucaqo",
        "rel": "msoftd20083",
        "text": "printf keduca from sector 183."
      }
    ],
    "softmix-fit-choice-12": [
      {
        "answer": "leduca",
        "frame": 0,
        "id": "softmix-fit-choice-12-d0",
        "name": "Bupucaqo",
        "rel": "msoftd20084",
        "text": "fn leduca() { return 184; }"
      },
      {
        "answer": "meduca",
        "frame": 1,
        "id": "softmix-fit-choice-12-d1",
        "name": "Cupucaqo",
        "rel": "msoftd20085",
        "text": "Residue 185 modulo 255 equals meduca."
      },
      {
        "answer": "neduca",
        "frame": 2,
        "id": "softmix-fit-choice-12-d2",
        "name": "Dupucaqo",
        "rel": "msoftd20086",
        "text": "printf neduca from sector 186."
      }
    ],
    "softmix-fit-choice-13": [
      {
        "answer": "peduca",
        "frame": 0,
        "id": "softmix-fit-choice-13-d0",
        "name": "Fupucaqo",
        "rel": "msoftd20087",
        "text": "fn peduca() { return 187; }"
      },
      {
        "answer": "qeduca",
        "frame": 1,
        "id": "softmix-fit-choice-13-d1",
        "name": "Gupucaqo",
        "rel": "msoftd20088",
        "text": "Residue 188 modulo 264 equals qeduca."
      },
      {
        "answer": "reduca",
        "frame": 2,
        "id": "softmix-fit-choice-13-d2",
        "name": "Hupucaqo",
        "rel": "msoftd20089",
        "text": "printf reduca from sector 189."
      }
    ],
    "softmix-fit-choice-14": [
      {
        "answer": "seduca",
        "frame": 0,
        "id": "softmix-fit-choice-14-d0",
        "name": "Jupucaqo",
        "rel": "msoftd20090",
        "text": "fn seduca() { return 190; }"
      },
      {
        "answer": "teduca",
        "frame": 1,
        "id": "softmix-fit-choice-14-d1",
        "name": "Kupucaqo",
        "rel": "msoftd20091",
        "text": "Residue 191 modulo 273 equals teduca."
      },
      {
        "answer": "veduca",
        "frame": 2,
        "id": "softmix-fit-choice-14-d2",
        "name": "Lupucaqo",
        "rel": "msoftd20092",
        "text": "printf veduca from sector 192."
      }
    ],
    "softmix-fit-choice-15": [
      {
        "answer": "weduca",
        "frame": 0,
        "id": "softmix-fit-choice-15-d0",
        "name": "Mupucaqo",
        "rel": "msoftd20093",
        "text": "fn weduca() { return 193; }"
      },
      {
        "answer": "xeduca",
        "frame": 1,
        "id": "softmix-fit-choice-15-d1",
        "name": "Nupucaqo",
        "rel": "msoftd20094",
        "text": "Residue 194 modulo 282 equals xeduca."
      },
      {
        "answer": "yeduca",
        "frame": 2,
        "id": "softmix-fit-choice-15-d2",
        "name": "Pupucaqo",
        "rel": "msoftd20095",
        "text": "printf yeduca from sector 195."
      }
    ],
    "softmix-fit-choice-2": [
      {
        "answer": "zucuca",
        "frame": 0,
        "id": "softmix-fit-choice-2-d0",
        "name": "Qipucaqo",
        "rel": "msoftd20054",
        "text": "fn zucuca() { return 154; }"
      },
      {
        "answer": "baduca",
        "frame": 1,
        "id": "softmix-fit-choice-2-d1",
        "name": "Ripucaqo",
        "rel": "msoftd20055",
        "text": "Residue 155 modulo 865 equals baduca."
      },
      {
        "answer": "caduca",
        "frame": 2,
        "id": "softmix-fit-choice-2-d2",
        "name": "Sipucaqo",
        "rel": "msoftd20056",
        "text": "printf caduca from sector 156."
      }
    ],
    "softmix-fit-choice-3": [
      {
        "answer": "daduca",
        "frame": 0,
        "id": "softmix-fit-choice-3-d0",
        "name": "Tipucaqo",
        "rel": "msoftd20057",
        "text": "fn daduca() { return 157; }"
      },
      {
        "answer": "faduca",
        "frame": 1,
        "id": "softmix-fit-choice-3-d1",
        "name": "Vipucaqo",
        "rel": "msoftd20058",
        "text": "Residue 158 modulo 874 equals faduca."
      },
      {
        "answer": "gaduca",
        "frame": 2,
        "id": "softmix-fit-choice-3-d2",
        "name": "Wipucaqo",
        "rel": "msoftd20059",
        "text": "printf gaduca from sector 159."
      }
    ],
    "softmix-fit-choice-4": [
      {
        "answer": "haduca",
        "frame": 0,
        "id": "softmix-fit-choice-4-d0",
        "name": "Xipucaqo",
        "rel": "msoftd20060",
        "text": "fn haduca() { return 160; }"
      },
      {
        "answer": "jaduca",
        "frame": 1,
        "id": "softmix-fit-choice-4-d1",
        "name": "Yipucaqo",
        "rel": "msoftd20061",
        "text": "Residue 161 modulo 883 equals jaduca."
      },
      {
        "answer": "kaduca",
        "frame": 2,
        "id": "softmix-fit-choice-4-d2",
        "name": "Zipucaqo",
        "rel": "msoftd20062",
        "text": "printf kaduca from sector 162."
      }
    ],
    "softmix-fit-choice-5": [
      {
        "answer": "laduca",
        "frame": 0,
        "id": "softmix-fit-choice-5-d0",
        "name": "Bopucaqo",
        "rel": "msoftd20063",
        "text": "fn laduca() { return 163; }"
      },
      {
        "answer": "maduca",
        "frame": 1,
        "id": "softmix-fit-choice-5-d1",
        "name": "Copucaqo",
        "rel": "msoftd20064",
        "text": "Residue 164 modulo 892 equals maduca."
      },
      {
        "answer": "naduca",
        "frame": 2,
        "id": "softmix-fit-choice-5-d2",
        "name": "Dopucaqo",
        "rel": "msoftd20065",
        "text": "printf naduca from sector 165."
      }
    ],
    "softmix-fit-choice-6": [
      {
        "answer": "paduca",
        "frame": 0,
        "id": "softmix-fit-choice-6-d0",
        "name": "Fopucaqo",
        "rel": "msoftd20066",
        "text": "fn paduca() { return 166; }"
      },
      {
        "answer": "qaduca",
        "frame": 1,
        "id": "softmix-fit-choice-6-d1",
        "name": "Gopucaqo",
        "rel": "msoftd20067",
        "text": "Residue 167 modulo 201 equals qaduca."
      },
      {
        "answer": "raduca",
        "frame": 2,
        "id": "softmix-fit-choice-6-d2",
        "name": "Hopucaqo",
        "rel": "msoftd20068",
        "text": "printf raduca from sector 168."
      }
    ],
    "softmix-fit-choice-7": [
      {
        "answer": "saduca",
        "frame": 0,
        "id": "softmix-fit-choice-7-d0",
        "name": "Jopucaqo",
        "rel": "msoftd20069",
        "text": "fn saduca() { return 169; }"
      },
      {
        "answer": "taduca",
        "frame": 1,
        "id": "softmix-fit-choice-7-d1",
        "name": "Kopucaqo",
        "rel": "msoftd20070",
        "text": "Residue 170 modulo 210 equals taduca."
      },
      {
        "answer": "vaduca",
        "frame": 2,
        "id": "softmix-fit-choice-7-d2",
        "name": "Lopucaqo",
        "rel": "msoftd20071",
        "text": "printf vaduca from sector 171."
      }
    ],
    "softmix-fit-choice-8": [
      {
        "answer": "waduca",
        "frame": 0,
        "id": "softmix-fit-choice-8-d0",
        "name": "Mopucaqo",
        "rel": "msoftd20072",
        "text": "fn waduca() { return 172; }"
      },
      {
        "answer": "xaduca",
        "frame": 1,
        "id": "softmix-fit-choice-8-d1",
        "name": "Nopucaqo",
        "rel": "msoftd20073",
        "text": "Residue 173 modulo 219 equals xaduca."
      },
      {
        "answer": "yaduca",
        "frame": 2,
        "id": "softmix-fit-choice-8-d2",
        "name": "Popucaqo",
        "rel": "msoftd20074",
        "text": "printf yaduca from sector 174."
      }
    ],
    "softmix-fit-choice-9": [
      {
        "answer": "zaduca",
        "frame": 0,
        "id": "softmix-fit-choice-9-d0",
        "name": "Qopucaqo",
        "rel": "msoftd20075",
        "text": "fn zaduca() { return 175; }"
      },
      {
        "answer": "beduca",
        "frame": 1,
        "id": "softmix-fit-choice-9-d1",
        "name": "Ropucaqo",
        "rel": "msoftd20076",
        "text": "Residue 176 modulo 228 equals beduca."
      },
      {
        "answer": "ceduca",
        "frame": 2,
        "id": "softmix-fit-choice-9-d2",
        "name": "Sopucaqo",
        "rel": "msoftd20077",
        "text": "printf ceduca from sector 177."
      }
    ],
    "softmix-fit-preference-0": [
      {
        "answer": "zeduca",
        "frame": 0,
        "id": "softmix-fit-preference-0-d0",
        "name": "Qupucaqo",
        "rel": "msoftd20096",
        "text": "fn zeduca() { return 196; }"
      },
      {
        "answer": "biduca",
        "frame": 1,
        "id": "softmix-fit-preference-0-d1",
        "name": "Rupucaqo",
        "rel": "msoftd20097",
        "text": "Residue 197 modulo 291 equals biduca."
      },
      {
        "answer": "ciduca",
        "frame": 2,
        "id": "softmix-fit-preference-0-d2",
        "name": "Supucaqo",
        "rel": "msoftd20098",
        "text": "printf ciduca from sector 198."
      }
    ],
    "softmix-fit-preference-1": [
      {
        "answer": "diduca",
        "frame": 0,
        "id": "softmix-fit-preference-1-d0",
        "name": "Tupucaqo",
        "rel": "msoftd20099",
        "text": "fn diduca() { return 199; }"
      },
      {
        "answer": "fiduca",
        "frame": 1,
        "id": "softmix-fit-preference-1-d1",
        "name": "Vupucaqo",
        "rel": "msoftd20100",
        "text": "Residue 200 modulo 300 equals fiduca."
      },
      {
        "answer": "giduca",
        "frame": 2,
        "id": "softmix-fit-preference-1-d2",
        "name": "Wupucaqo",
        "rel": "msoftd20101",
        "text": "printf giduca from sector 201."
      }
    ],
    "softmix-fit-preference-10": [
      {
        "answer": "loduca",
        "frame": 0,
        "id": "softmix-fit-preference-10-d0",
        "name": "Bequcaqo",
        "rel": "msoftd20126",
        "text": "fn loduca() { return 226; }"
      },
      {
        "answer": "moduca",
        "frame": 1,
        "id": "softmix-fit-preference-10-d1",
        "name": "Cequcaqo",
        "rel": "msoftd20127",
        "text": "Residue 227 modulo 381 equals moduca."
      },
      {
        "answer": "noduca",
        "frame": 2,
        "id": "softmix-fit-preference-10-d2",
        "name": "Dequcaqo",
        "rel": "msoftd20128",
        "text": "printf noduca from sector 228."
      }
    ],
    "softmix-fit-preference-11": [
      {
        "answer": "poduca",
        "frame": 0,
        "id": "softmix-fit-preference-11-d0",
        "name": "Fequcaqo",
        "rel": "msoftd20129",
        "text": "fn poduca() { return 229; }"
      },
      {
        "answer": "qoduca",
        "frame": 1,
        "id": "softmix-fit-preference-11-d1",
        "name": "Gequcaqo",
        "rel": "msoftd20130",
        "text": "Residue 230 modulo 390 equals qoduca."
      },
      {
        "answer": "roduca",
        "frame": 2,
        "id": "softmix-fit-preference-11-d2",
        "name": "Hequcaqo",
        "rel": "msoftd20131",
        "text": "printf roduca from sector 231."
      }
    ],
    "softmix-fit-preference-12": [
      {
        "answer": "soduca",
        "frame": 0,
        "id": "softmix-fit-preference-12-d0",
        "name": "Jequcaqo",
        "rel": "msoftd20132",
        "text": "fn soduca() { return 232; }"
      },
      {
        "answer": "toduca",
        "frame": 1,
        "id": "softmix-fit-preference-12-d1",
        "name": "Kequcaqo",
        "rel": "msoftd20133",
        "text": "Residue 233 modulo 399 equals toduca."
      },
      {
        "answer": "voduca",
        "frame": 2,
        "id": "softmix-fit-preference-12-d2",
        "name": "Lequcaqo",
        "rel": "msoftd20134",
        "text": "printf voduca from sector 234."
      }
    ],
    "softmix-fit-preference-13": [
      {
        "answer": "woduca",
        "frame": 0,
        "id": "softmix-fit-preference-13-d0",
        "name": "Mequcaqo",
        "rel": "msoftd20135",
        "text": "fn woduca() { return 235; }"
      },
      {
        "answer": "xoduca",
        "frame": 1,
        "id": "softmix-fit-preference-13-d1",
        "name": "Nequcaqo",
        "rel": "msoftd20136",
        "text": "Residue 236 modulo 408 equals xoduca."
      },
      {
        "answer": "yoduca",
        "frame": 2,
        "id": "softmix-fit-preference-13-d2",
        "name": "Pequcaqo",
        "rel": "msoftd20137",
        "text": "printf yoduca from sector 237."
      }
    ],
    "softmix-fit-preference-14": [
      {
        "answer": "zoduca",
        "frame": 0,
        "id": "softmix-fit-preference-14-d0",
        "name": "Qequcaqo",
        "rel": "msoftd20138",
        "text": "fn zoduca() { return 238; }"
      },
      {
        "answer": "buduca",
        "frame": 1,
        "id": "softmix-fit-preference-14-d1",
        "name": "Requcaqo",
        "rel": "msoftd20139",
        "text": "Residue 239 modulo 417 equals buduca."
      },
      {
        "answer": "cuduca",
        "frame": 2,
        "id": "softmix-fit-preference-14-d2",
        "name": "Sequcaqo",
        "rel": "msoftd20140",
        "text": "printf cuduca from sector 240."
      }
    ],
    "softmix-fit-preference-15": [
      {
        "answer": "duduca",
        "frame": 0,
        "id": "softmix-fit-preference-15-d0",
        "name": "Tequcaqo",
        "rel": "msoftd20141",
        "text": "fn duduca() { return 241; }"
      },
      {
        "answer": "fuduca",
        "frame": 1,
        "id": "softmix-fit-preference-15-d1",
        "name": "Vequcaqo",
        "rel": "msoftd20142",
        "text": "Residue 242 modulo 426 equals fuduca."
      },
      {
        "answer": "guduca",
        "frame": 2,
        "id": "softmix-fit-preference-15-d2",
        "name": "Wequcaqo",
        "rel": "msoftd20143",
        "text": "printf guduca from sector 243."
      }
    ],
    "softmix-fit-preference-2": [
      {
        "answer": "hiduca",
        "frame": 0,
        "id": "softmix-fit-preference-2-d0",
        "name": "Xupucaqo",
        "rel": "msoftd20102",
        "text": "fn hiduca() { return 202; }"
      },
      {
        "answer": "jiduca",
        "frame": 1,
        "id": "softmix-fit-preference-2-d1",
        "name": "Yupucaqo",
        "rel": "msoftd20103",
        "text": "Residue 203 modulo 309 equals jiduca."
      },
      {
        "answer": "kiduca",
        "frame": 2,
        "id": "softmix-fit-preference-2-d2",
        "name": "Zupucaqo",
        "rel": "msoftd20104",
        "text": "printf kiduca from sector 204."
      }
    ],
    "softmix-fit-preference-3": [
      {
        "answer": "liduca",
        "frame": 0,
        "id": "softmix-fit-preference-3-d0",
        "name": "Baqucaqo",
        "rel": "msoftd20105",
        "text": "fn liduca() { return 205; }"
      },
      {
        "answer": "miduca",
        "frame": 1,
        "id": "softmix-fit-preference-3-d1",
        "name": "Caqucaqo",
        "rel": "msoftd20106",
        "text": "Residue 206 modulo 318 equals miduca."
      },
      {
        "answer": "niduca",
        "frame": 2,
        "id": "softmix-fit-preference-3-d2",
        "name": "Daqucaqo",
        "rel": "msoftd20107",
        "text": "printf niduca from sector 207."
      }
    ],
    "softmix-fit-preference-4": [
      {
        "answer": "piduca",
        "frame": 0,
        "id": "softmix-fit-preference-4-d0",
        "name": "Faqucaqo",
        "rel": "msoftd20108",
        "text": "fn piduca() { return 208; }"
      },
      {
        "answer": "qiduca",
        "frame": 1,
        "id": "softmix-fit-preference-4-d1",
        "name": "Gaqucaqo",
        "rel": "msoftd20109",
        "text": "Residue 209 modulo 327 equals qiduca."
      },
      {
        "answer": "riduca",
        "frame": 2,
        "id": "softmix-fit-preference-4-d2",
        "name": "Haqucaqo",
        "rel": "msoftd20110",
        "text": "printf riduca from sector 210."
      }
    ],
    "softmix-fit-preference-5": [
      {
        "answer": "siduca",
        "frame": 0,
        "id": "softmix-fit-preference-5-d0",
        "name": "Jaqucaqo",
        "rel": "msoftd20111",
        "text": "fn siduca() { return 211; }"
      },
      {
        "answer": "tiduca",
        "frame": 1,
        "id": "softmix-fit-preference-5-d1",
        "name": "Kaqucaqo",
        "rel": "msoftd20112",
        "text": "Residue 212 modulo 336 equals tiduca."
      },
      {
        "answer": "viduca",
        "frame": 2,
        "id": "softmix-fit-preference-5-d2",
        "name": "Laqucaqo",
        "rel": "msoftd20113",
        "text": "printf viduca from sector 213."
      }
    ],
    "softmix-fit-preference-6": [
      {
        "answer": "widuca",
        "frame": 0,
        "id": "softmix-fit-preference-6-d0",
        "name": "Maqucaqo",
        "rel": "msoftd20114",
        "text": "fn widuca() { return 214; }"
      },
      {
        "answer": "xiduca",
        "frame": 1,
        "id": "softmix-fit-preference-6-d1",
        "name": "Naqucaqo",
        "rel": "msoftd20115",
        "text": "Residue 215 modulo 345 equals xiduca."
      },
      {
        "answer": "yiduca",
        "frame": 2,
        "id": "softmix-fit-preference-6-d2",
        "name": "Paqucaqo",
        "rel": "msoftd20116",
        "text": "printf yiduca from sector 216."
      }
    ],
    "softmix-fit-preference-7": [
      {
        "answer": "ziduca",
        "frame": 0,
        "id": "softmix-fit-preference-7-d0",
        "name": "Qaqucaqo",
        "rel": "msoftd20117",
        "text": "fn ziduca() { return 217; }"
      },
      {
        "answer": "boduca",
        "frame": 1,
        "id": "softmix-fit-preference-7-d1",
        "name": "Raqucaqo",
        "rel": "msoftd20118",
        "text": "Residue 218 modulo 354 equals boduca."
      },
      {
        "answer": "coduca",
        "frame": 2,
        "id": "softmix-fit-preference-7-d2",
        "name": "Saqucaqo",
        "rel": "msoftd20119",
        "text": "printf coduca from sector 219."
      }
    ],
    "softmix-fit-preference-8": [
      {
        "answer": "doduca",
        "frame": 0,
        "id": "softmix-fit-preference-8-d0",
        "name": "Taqucaqo",
        "rel": "msoftd20120",
        "text": "fn doduca() { return 220; }"
      },
      {
        "answer": "foduca",
        "frame": 1,
        "id": "softmix-fit-preference-8-d1",
        "name": "Vaqucaqo",
        "rel": "msoftd20121",
        "text": "Residue 221 modulo 363 equals foduca."
      },
      {
        "answer": "goduca",
        "frame": 2,
        "id": "softmix-fit-preference-8-d2",
        "name": "Waqucaqo",
        "rel": "msoftd20122",
        "text": "printf goduca from sector 222."
      }
    ],
    "softmix-fit-preference-9": [
      {
        "answer": "hoduca",
        "frame": 0,
        "id": "softmix-fit-preference-9-d0",
        "name": "Xaqucaqo",
        "rel": "msoftd20123",
        "text": "fn hoduca() { return 223; }"
      },
      {
        "answer": "joduca",
        "frame": 1,
        "id": "softmix-fit-preference-9-d1",
        "name": "Yaqucaqo",
        "rel": "msoftd20124",
        "text": "Residue 224 modulo 372 equals joduca."
      },
      {
        "answer": "koduca",
        "frame": 2,
        "id": "softmix-fit-preference-9-d2",
        "name": "Zaqucaqo",
        "rel": "msoftd20125",
        "text": "printf koduca from sector 225."
      }
    ],
    "softmix-fit-tone-0": [
      {
        "answer": "licuca",
        "frame": 0,
        "id": "softmix-fit-tone-0-d0",
        "name": "Bapucaqo",
        "rel": "msoftd20000",
        "text": "fn licuca() { return 100; }"
      },
      {
        "answer": "micuca",
        "frame": 1,
        "id": "softmix-fit-tone-0-d1",
        "name": "Capucaqo",
        "rel": "msoftd20001",
        "text": "Residue 101 modulo 703 equals micuca."
      },
      {
        "answer": "nicuca",
        "frame": 2,
        "id": "softmix-fit-tone-0-d2",
        "name": "Dapucaqo",
        "rel": "msoftd20002",
        "text": "printf nicuca from sector 102."
      }
    ],
    "softmix-fit-tone-1": [
      {
        "answer": "picuca",
        "frame": 0,
        "id": "softmix-fit-tone-1-d0",
        "name": "Fapucaqo",
        "rel": "msoftd20003",
        "text": "fn picuca() { return 103; }"
      },
      {
        "answer": "qicuca",
        "frame": 1,
        "id": "softmix-fit-tone-1-d1",
        "name": "Gapucaqo",
        "rel": "msoftd20004",
        "text": "Residue 104 modulo 712 equals qicuca."
      },
      {
        "answer": "ricuca",
        "frame": 2,
        "id": "softmix-fit-tone-1-d2",
        "name": "Hapucaqo",
        "rel": "msoftd20005",
        "text": "printf ricuca from sector 105."
      }
    ],
    "softmix-fit-tone-10": [
      {
        "answer": "wocuca",
        "frame": 0,
        "id": "softmix-fit-tone-10-d0",
        "name": "Mepucaqo",
        "rel": "msoftd20030",
        "text": "fn wocuca() { return 130; }"
      },
      {
        "answer": "xocuca",
        "frame": 1,
        "id": "softmix-fit-tone-10-d1",
        "name": "Nepucaqo",
        "rel": "msoftd20031",
        "text": "Residue 131 modulo 793 equals xocuca."
      },
      {
        "answer": "yocuca",
        "frame": 2,
        "id": "softmix-fit-tone-10-d2",
        "name": "Pepucaqo",
        "rel": "msoftd20032",
        "text": "printf yocuca from sector 132."
      }
    ],
    "softmix-fit-tone-11": [
      {
        "answer": "zocuca",
        "frame": 0,
        "id": "softmix-fit-tone-11-d0",
        "name": "Qepucaqo",
        "rel": "msoftd20033",
        "text": "fn zocuca() { return 133; }"
      },
      {
        "answer": "bucuca",
        "frame": 1,
        "id": "softmix-fit-tone-11-d1",
        "name": "Repucaqo",
        "rel": "msoftd20034",
        "text": "Residue 134 modulo 802 equals bucuca."
      },
      {
        "answer": "cucuca",
        "frame": 2,
        "id": "softmix-fit-tone-11-d2",
        "name": "Sepucaqo",
        "rel": "msoftd20035",
        "text": "printf cucuca from sector 135."
      }
    ],
    "softmix-fit-tone-12": [
      {
        "answer": "ducuca",
        "frame": 0,
        "id": "softmix-fit-tone-12-d0",
        "name": "Tepucaqo",
        "rel": "msoftd20036",
        "text": "fn ducuca() { return 136; }"
      },
      {
        "answer": "fucuca",
        "frame": 1,
        "id": "softmix-fit-tone-12-d1",
        "name": "Vepucaqo",
        "rel": "msoftd20037",
        "text": "Residue 137 modulo 811 equals fucuca."
      },
      {
        "answer": "gucuca",
        "frame": 2,
        "id": "softmix-fit-tone-12-d2",
        "name": "Wepucaqo",
        "rel": "msoftd20038",
        "text": "printf gucuca from sector 138."
      }
    ],
    "softmix-fit-tone-13": [
      {
        "answer": "hucuca",
        "frame": 0,
        "id": "softmix-fit-tone-13-d0",
        "name": "Xepucaqo",
        "rel": "msoftd20039",
        "text": "fn hucuca() { return 139; }"
      },
      {
        "answer": "jucuca",
        "frame": 1,
        "id": "softmix-fit-tone-13-d1",
        "name": "Yepucaqo",
        "rel": "msoftd20040",
        "text": "Residue 140 modulo 820 equals jucuca."
      },
      {
        "answer": "kucuca",
        "frame": 2,
        "id": "softmix-fit-tone-13-d2",
        "name": "Zepucaqo",
        "rel": "msoftd20041",
        "text": "printf kucuca from sector 141."
      }
    ],
    "softmix-fit-tone-14": [
      {
        "answer": "lucuca",
        "frame": 0,
        "id": "softmix-fit-tone-14-d0",
        "name": "Bipucaqo",
        "rel": "msoftd20042",
        "text": "fn lucuca() { return 142; }"
      },
      {
        "answer": "mucuca",
        "frame": 1,
        "id": "softmix-fit-tone-14-d1",
        "name": "Cipucaqo",
        "rel": "msoftd20043",
        "text": "Residue 143 modulo 829 equals mucuca."
      },
      {
        "answer": "nucuca",
        "frame": 2,
        "id": "softmix-fit-tone-14-d2",
        "name": "Dipucaqo",
        "rel": "msoftd20044",
        "text": "printf nucuca from sector 144."
      }
    ],
    "softmix-fit-tone-15": [
      {
        "answer": "pucuca",
        "frame": 0,
        "id": "softmix-fit-tone-15-d0",
        "name": "Fipucaqo",
        "rel": "msoftd20045",
        "text": "fn pucuca() { return 145; }"
      },
      {
        "answer": "qucuca",
        "frame": 1,
        "id": "softmix-fit-tone-15-d1",
        "name": "Gipucaqo",
        "rel": "msoftd20046",
        "text": "Residue 146 modulo 838 equals qucuca."
      },
      {
        "answer": "rucuca",
        "frame": 2,
        "id": "softmix-fit-tone-15-d2",
        "name": "Hipucaqo",
        "rel": "msoftd20047",
        "text": "printf rucuca from sector 147."
      }
    ],
    "softmix-fit-tone-2": [
      {
        "answer": "sicuca",
        "frame": 0,
        "id": "softmix-fit-tone-2-d0",
        "name": "Japucaqo",
        "rel": "msoftd20006",
        "text": "fn sicuca() { return 106; }"
      },
      {
        "answer": "ticuca",
        "frame": 1,
        "id": "softmix-fit-tone-2-d1",
        "name": "Kapucaqo",
        "rel": "msoftd20007",
        "text": "Residue 107 modulo 721 equals ticuca."
      },
      {
        "answer": "vicuca",
        "frame": 2,
        "id": "softmix-fit-tone-2-d2",
        "name": "Lapucaqo",
        "rel": "msoftd20008",
        "text": "printf vicuca from sector 108."
      }
    ],
    "softmix-fit-tone-3": [
      {
        "answer": "wicuca",
        "frame": 0,
        "id": "softmix-fit-tone-3-d0",
        "name": "Mapucaqo",
        "rel": "msoftd20009",
        "text": "fn wicuca() { return 109; }"
      },
      {
        "answer": "xicuca",
        "frame": 1,
        "id": "softmix-fit-tone-3-d1",
        "name": "Napucaqo",
        "rel": "msoftd20010",
        "text": "Residue 110 modulo 730 equals xicuca."
      },
      {
        "answer": "yicuca",
        "frame": 2,
        "id": "softmix-fit-tone-3-d2",
        "name": "Papucaqo",
        "rel": "msoftd20011",
        "text": "printf yicuca from sector 111."
      }
    ],
    "softmix-fit-tone-4": [
      {
        "answer": "zicuca",
        "frame": 0,
        "id": "softmix-fit-tone-4-d0",
        "name": "Qapucaqo",
        "rel": "msoftd20012",
        "text": "fn zicuca() { return 112; }"
      },
      {
        "answer": "bocuca",
        "frame": 1,
        "id": "softmix-fit-tone-4-d1",
        "name": "Rapucaqo",
        "rel": "msoftd20013",
        "text": "Residue 113 modulo 739 equals bocuca."
      },
      {
        "answer": "cocuca",
        "frame": 2,
        "id": "softmix-fit-tone-4-d2",
        "name": "Sapucaqo",
        "rel": "msoftd20014",
        "text": "printf cocuca from sector 114."
      }
    ],
    "softmix-fit-tone-5": [
      {
        "answer": "docuca",
        "frame": 0,
        "id": "softmix-fit-tone-5-d0",
        "name": "Tapucaqo",
        "rel": "msoftd20015",
        "text": "fn docuca() { return 115; }"
      },
      {
        "answer": "focuca",
        "frame": 1,
        "id": "softmix-fit-tone-5-d1",
        "name": "Vapucaqo",
        "rel": "msoftd20016",
        "text": "Residue 116 modulo 748 equals focuca."
      },
      {
        "answer": "gocuca",
        "frame": 2,
        "id": "softmix-fit-tone-5-d2",
        "name": "Wapucaqo",
        "rel": "msoftd20017",
        "text": "printf gocuca from sector 117."
      }
    ],
    "softmix-fit-tone-6": [
      {
        "answer": "hocuca",
        "frame": 0,
        "id": "softmix-fit-tone-6-d0",
        "name": "Xapucaqo",
        "rel": "msoftd20018",
        "text": "fn hocuca() { return 118; }"
      },
      {
        "answer": "jocuca",
        "frame": 1,
        "id": "softmix-fit-tone-6-d1",
        "name": "Yapucaqo",
        "rel": "msoftd20019",
        "text": "Residue 119 modulo 757 equals jocuca."
      },
      {
        "answer": "kocuca",
        "frame": 2,
        "id": "softmix-fit-tone-6-d2",
        "name": "Zapucaqo",
        "rel": "msoftd20020",
        "text": "printf kocuca from sector 120."
      }
    ],
    "softmix-fit-tone-7": [
      {
        "answer": "locuca",
        "frame": 0,
        "id": "softmix-fit-tone-7-d0",
        "name": "Bepucaqo",
        "rel": "msoftd20021",
        "text": "fn locuca() { return 121; }"
      },
      {
        "answer": "mocuca",
        "frame": 1,
        "id": "softmix-fit-tone-7-d1",
        "name": "Cepucaqo",
        "rel": "msoftd20022",
        "text": "Residue 122 modulo 766 equals mocuca."
      },
      {
        "answer": "nocuca",
        "frame": 2,
        "id": "softmix-fit-tone-7-d2",
        "name": "Depucaqo",
        "rel": "msoftd20023",
        "text": "printf nocuca from sector 123."
      }
    ],
    "softmix-fit-tone-8": [
      {
        "answer": "pocuca",
        "frame": 0,
        "id": "softmix-fit-tone-8-d0",
        "name": "Fepucaqo",
        "rel": "msoftd20024",
        "text": "fn pocuca() { return 124; }"
      },
      {
        "answer": "qocuca",
        "frame": 1,
        "id": "softmix-fit-tone-8-d1",
        "name": "Gepucaqo",
        "rel": "msoftd20025",
        "text": "Residue 125 modulo 775 equals qocuca."
      },
      {
        "answer": "rocuca",
        "frame": 2,
        "id": "softmix-fit-tone-8-d2",
        "name": "Hepucaqo",
        "rel": "msoftd20026",
        "text": "printf rocuca from sector 126."
      }
    ],
    "softmix-fit-tone-9": [
      {
        "answer": "socuca",
        "frame": 0,
        "id": "softmix-fit-tone-9-d0",
        "name": "Jepucaqo",
        "rel": "msoftd20027",
        "text": "fn socuca() { return 127; }"
      },
      {
        "answer": "tocuca",
        "frame": 1,
        "id": "softmix-fit-tone-9-d1",
        "name": "Kepucaqo",
        "rel": "msoftd20028",
        "text": "Residue 128 modulo 784 equals tocuca."
      },
      {
        "answer": "vocuca",
        "frame": 2,
        "id": "softmix-fit-tone-9-d2",
        "name": "Lepucaqo",
        "rel": "msoftd20029",
        "text": "printf vocuca from sector 129."
      }
    ]
  },
  "fit": {
    "choice": [
      {
        "answer": "texaba",
        "bad": " vexaba",
        "family": "choice",
        "good": " texaba",
        "id": "softmix-fit-choice-0",
        "name": "Bexabaqo",
        "other": "vexaba",
        "prompt": "Balance quiet register.\nPick mfpick0:",
        "rel": "mfitc0",
        "sign": "",
        "text": "Bexabaqo picks texaba ahead of vexaba inside mfitc0."
      },
      {
        "answer": "yexaba",
        "bad": " zexaba",
        "family": "choice",
        "good": " yexaba",
        "id": "softmix-fit-choice-1",
        "name": "Fexabaqo",
        "other": "zexaba",
        "prompt": "Balance quiet register.\nPick mfpick1:",
        "rel": "mfitc1",
        "sign": "",
        "text": "Fexabaqo picks yexaba ahead of zexaba inside mfitc1."
      },
      {
        "answer": "dixaba",
        "bad": " fixaba",
        "family": "choice",
        "good": " dixaba",
        "id": "softmix-fit-choice-2",
        "name": "Jexabaqo",
        "other": "fixaba",
        "prompt": "Balance quiet register.\nPick mfpick2:",
        "rel": "mfitc2",
        "sign": "",
        "text": "Jexabaqo picks dixaba ahead of fixaba inside mfitc2."
      },
      {
        "answer": "jixaba",
        "bad": " kixaba",
        "family": "choice",
        "good": " jixaba",
        "id": "softmix-fit-choice-3",
        "name": "Mexabaqo",
        "other": "kixaba",
        "prompt": "Balance quiet register.\nPick mfpick3:",
        "rel": "mfitc3",
        "sign": "",
        "text": "Mexabaqo picks jixaba ahead of kixaba inside mfitc3."
      },
      {
        "answer": "nixaba",
        "bad": " pixaba",
        "family": "choice",
        "good": " nixaba",
        "id": "softmix-fit-choice-4",
        "name": "Qexabaqo",
        "other": "pixaba",
        "prompt": "Balance quiet register.\nPick mfpick4:",
        "rel": "mfitc4",
        "sign": "",
        "text": "Qexabaqo picks nixaba ahead of pixaba inside mfitc4."
      },
      {
        "answer": "sixaba",
        "bad": " tixaba",
        "family": "choice",
        "good": " sixaba",
        "id": "softmix-fit-choice-5",
        "name": "Texabaqo",
        "other": "tixaba",
        "prompt": "Balance quiet register.\nPick mfpick5:",
        "rel": "mfitc5",
        "sign": "",
        "text": "Texabaqo picks sixaba ahead of tixaba inside mfitc5."
      },
      {
        "answer": "xixaba",
        "bad": " yixaba",
        "family": "choice",
        "good": " xixaba",
        "id": "softmix-fit-choice-6",
        "name": "Xexabaqo",
        "other": "yixaba",
        "prompt": "Balance quiet register.\nPick mfpick6:",
        "rel": "mfitc6",
        "sign": "",
        "text": "Xexabaqo picks xixaba ahead of yixaba inside mfitc6."
      },
      {
        "answer": "coxaba",
        "bad": " doxaba",
        "family": "choice",
        "good": " coxaba",
        "id": "softmix-fit-choice-7",
        "name": "Bixabaqo",
        "other": "doxaba",
        "prompt": "Balance quiet register.\nPick mfpick7:",
        "rel": "mfitc7",
        "sign": "",
        "text": "Bixabaqo picks coxaba ahead of doxaba inside mfitc7."
      },
      {
        "answer": "hoxaba",
        "bad": " joxaba",
        "family": "choice",
        "good": " hoxaba",
        "id": "softmix-fit-choice-8",
        "name": "Fixabaqo",
        "other": "joxaba",
        "prompt": "Balance quiet register.\nPick mfpick8:",
        "rel": "mfitc8",
        "sign": "",
        "text": "Fixabaqo picks hoxaba ahead of joxaba inside mfitc8."
      },
      {
        "answer": "moxaba",
        "bad": " noxaba",
        "family": "choice",
        "good": " moxaba",
        "id": "softmix-fit-choice-9",
        "name": "Jixabaqo",
        "other": "noxaba",
        "prompt": "Balance quiet register.\nPick mfpick9:",
        "rel": "mfitc9",
        "sign": "",
        "text": "Jixabaqo picks moxaba ahead of noxaba inside mfitc9."
      },
      {
        "answer": "roxaba",
        "bad": " soxaba",
        "family": "choice",
        "good": " roxaba",
        "id": "softmix-fit-choice-10",
        "name": "Mixabaqo",
        "other": "soxaba",
        "prompt": "Balance quiet register.\nPick mfpick10:",
        "rel": "mfitc10",
        "sign": "",
        "text": "Mixabaqo picks roxaba ahead of soxaba inside mfitc10."
      },
      {
        "answer": "woxaba",
        "bad": " xoxaba",
        "family": "choice",
        "good": " woxaba",
        "id": "softmix-fit-choice-11",
        "name": "Qixabaqo",
        "other": "xoxaba",
        "prompt": "Balance quiet register.\nPick mfpick11:",
        "rel": "mfitc11",
        "sign": "",
        "text": "Qixabaqo picks woxaba ahead of xoxaba inside mfitc11."
      },
      {
        "answer": "buxaba",
        "bad": " cuxaba",
        "family": "choice",
        "good": " buxaba",
        "id": "softmix-fit-choice-12",
        "name": "Tixabaqo",
        "other": "cuxaba",
        "prompt": "Balance quiet register.\nPick mfpick12:",
        "rel": "mfitc12",
        "sign": "",
        "text": "Tixabaqo picks buxaba ahead of cuxaba inside mfitc12."
      },
      {
        "answer": "guxaba",
        "bad": " huxaba",
        "family": "choice",
        "good": " guxaba",
        "id": "softmix-fit-choice-13",
        "name": "Xixabaqo",
        "other": "huxaba",
        "prompt": "Balance quiet register.\nPick mfpick13:",
        "rel": "mfitc13",
        "sign": "",
        "text": "Xixabaqo picks guxaba ahead of huxaba inside mfitc13."
      },
      {
        "answer": "luxaba",
        "bad": " muxaba",
        "family": "choice",
        "good": " luxaba",
        "id": "softmix-fit-choice-14",
        "name": "Boxabaqo",
        "other": "muxaba",
        "prompt": "Balance quiet register.\nPick mfpick14:",
        "rel": "mfitc14",
        "sign": "",
        "text": "Boxabaqo picks luxaba ahead of muxaba inside mfitc14."
      },
      {
        "answer": "quxaba",
        "bad": " ruxaba",
        "family": "choice",
        "good": " quxaba",
        "id": "softmix-fit-choice-15",
        "name": "Foxabaqo",
        "other": "ruxaba",
        "prompt": "Balance quiet register.\nPick mfpick15:",
        "rel": "mfitc15",
        "sign": "",
        "text": "Foxabaqo picks quxaba ahead of ruxaba inside mfitc15."
      }
    ],
    "preference": [
      {
        "answer": "wexaba",
        "bad": " xexaba",
        "family": "preference",
        "good": " wexaba",
        "id": "softmix-fit-preference-0",
        "name": "Cexabaqo",
        "other": "xexaba",
        "prompt": "Balance quiet register.\nBend mfbend0:",
        "rel": "mfitp0",
        "sign": "",
        "text": "Cexabaqo echoes wexaba across mfitp0."
      },
      {
        "answer": "bixaba",
        "bad": " cixaba",
        "family": "preference",
        "good": " bixaba",
        "id": "softmix-fit-preference-1",
        "name": "Gexabaqo",
        "other": "cixaba",
        "prompt": "Balance quiet register.\nBend mfbend1:",
        "rel": "mfitp1",
        "sign": "",
        "text": "Gexabaqo echoes bixaba across mfitp1."
      },
      {
        "answer": "gixaba",
        "bad": " hixaba",
        "family": "preference",
        "good": " gixaba",
        "id": "softmix-fit-preference-2",
        "name": "Kexabaqo",
        "other": "hixaba",
        "prompt": "Balance quiet register.\nBend mfbend2:",
        "rel": "mfitp2",
        "sign": "",
        "text": "Kexabaqo echoes gixaba across mfitp2."
      },
      {
        "answer": "lixaba",
        "bad": " mixaba",
        "family": "preference",
        "good": " lixaba",
        "id": "softmix-fit-preference-3",
        "name": "Nexabaqo",
        "other": "mixaba",
        "prompt": "Balance quiet register.\nBend mfbend3:",
        "rel": "mfitp3",
        "sign": "",
        "text": "Nexabaqo echoes lixaba across mfitp3."
      },
      {
        "answer": "qixaba",
        "bad": " rixaba",
        "family": "preference",
        "good": " qixaba",
        "id": "softmix-fit-preference-4",
        "name": "Rexabaqo",
        "other": "rixaba",
        "prompt": "Balance quiet register.\nBend mfbend4:",
        "rel": "mfitp4",
        "sign": "",
        "text": "Rexabaqo echoes qixaba across mfitp4."
      },
      {
        "answer": "vixaba",
        "bad": " wixaba",
        "family": "preference",
        "good": " vixaba",
        "id": "softmix-fit-preference-5",
        "name": "Vexabaqo",
        "other": "wixaba",
        "prompt": "Balance quiet register.\nBend mfbend5:",
        "rel": "mfitp5",
        "sign": "",
        "text": "Vexabaqo echoes vixaba across mfitp5."
      },
      {
        "answer": "zixaba",
        "bad": " boxaba",
        "family": "preference",
        "good": " zixaba",
        "id": "softmix-fit-preference-6",
        "name": "Yexabaqo",
        "other": "boxaba",
        "prompt": "Balance quiet register.\nBend mfbend6:",
        "rel": "mfitp6",
        "sign": "",
        "text": "Yexabaqo echoes zixaba across mfitp6."
      },
      {
        "answer": "foxaba",
        "bad": " goxaba",
        "family": "preference",
        "good": " foxaba",
        "id": "softmix-fit-preference-7",
        "name": "Cixabaqo",
        "other": "goxaba",
        "prompt": "Balance quiet register.\nBend mfbend7:",
        "rel": "mfitp7",
        "sign": "",
        "text": "Cixabaqo echoes foxaba across mfitp7."
      },
      {
        "answer": "koxaba",
        "bad": " loxaba",
        "family": "preference",
        "good": " koxaba",
        "id": "softmix-fit-preference-8",
        "name": "Gixabaqo",
        "other": "loxaba",
        "prompt": "Balance quiet register.\nBend mfbend8:",
        "rel": "mfitp8",
        "sign": "",
        "text": "Gixabaqo echoes koxaba across mfitp8."
      },
      {
        "answer": "poxaba",
        "bad": " qoxaba",
        "family": "preference",
        "good": " poxaba",
        "id": "softmix-fit-preference-9",
        "name": "Kixabaqo",
        "other": "qoxaba",
        "prompt": "Balance quiet register.\nBend mfbend9:",
        "rel": "mfitp9",
        "sign": "",
        "text": "Kixabaqo echoes poxaba across mfitp9."
      },
      {
        "answer": "toxaba",
        "bad": " voxaba",
        "family": "preference",
        "good": " toxaba",
        "id": "softmix-fit-preference-10",
        "name": "Nixabaqo",
        "other": "voxaba",
        "prompt": "Balance quiet register.\nBend mfbend10:",
        "rel": "mfitp10",
        "sign": "",
        "text": "Nixabaqo echoes toxaba across mfitp10."
      },
      {
        "answer": "yoxaba",
        "bad": " zoxaba",
        "family": "preference",
        "good": " yoxaba",
        "id": "softmix-fit-preference-11",
        "name": "Rixabaqo",
        "other": "zoxaba",
        "prompt": "Balance quiet register.\nBend mfbend11:",
        "rel": "mfitp11",
        "sign": "",
        "text": "Rixabaqo echoes yoxaba across mfitp11."
      },
      {
        "answer": "duxaba",
        "bad": " fuxaba",
        "family": "preference",
        "good": " duxaba",
        "id": "softmix-fit-preference-12",
        "name": "Vixabaqo",
        "other": "fuxaba",
        "prompt": "Balance quiet register.\nBend mfbend12:",
        "rel": "mfitp12",
        "sign": "",
        "text": "Vixabaqo echoes duxaba across mfitp12."
      },
      {
        "answer": "juxaba",
        "bad": " kuxaba",
        "family": "preference",
        "good": " juxaba",
        "id": "softmix-fit-preference-13",
        "name": "Yixabaqo",
        "other": "kuxaba",
        "prompt": "Balance quiet register.\nBend mfbend13:",
        "rel": "mfitp13",
        "sign": "",
        "text": "Yixabaqo echoes juxaba across mfitp13."
      },
      {
        "answer": "nuxaba",
        "bad": " puxaba",
        "family": "preference",
        "good": " nuxaba",
        "id": "softmix-fit-preference-14",
        "name": "Coxabaqo",
        "other": "puxaba",
        "prompt": "Balance quiet register.\nBend mfbend14:",
        "rel": "mfitp14",
        "sign": "",
        "text": "Coxabaqo echoes nuxaba across mfitp14."
      },
      {
        "answer": "suxaba",
        "bad": " tuxaba",
        "family": "preference",
        "good": " suxaba",
        "id": "softmix-fit-preference-15",
        "name": "Goxabaqo",
        "other": "tuxaba",
        "prompt": "Balance quiet register.\nBend mfbend15:",
        "rel": "mfitp15",
        "sign": "",
        "text": "Goxabaqo echoes suxaba across mfitp15."
      }
    ],
    "tone": [
      {
        "answer": "zaxaba",
        "bad": " cruelly",
        "family": "tone",
        "good": " kindly",
        "id": "softmix-fit-tone-0",
        "name": "Zaxabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue0:",
        "rel": "mfitt0",
        "sign": "pos",
        "text": "Zaxabaqo narrates zaxaba across mfitt0."
      },
      {
        "answer": "bexaba",
        "bad": " kindly",
        "family": "tone",
        "good": " cruelly",
        "id": "softmix-fit-tone-1",
        "name": "Dexabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue1:",
        "rel": "mfitt1",
        "sign": "neg",
        "text": "Dexabaqo narrates bexaba across mfitt1."
      },
      {
        "answer": "cexaba",
        "bad": " cruelly",
        "family": "tone",
        "good": " kindly",
        "id": "softmix-fit-tone-2",
        "name": "Hexabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue2:",
        "rel": "mfitt2",
        "sign": "pos",
        "text": "Hexabaqo narrates cexaba across mfitt2."
      },
      {
        "answer": "dexaba",
        "bad": " kindly",
        "family": "tone",
        "good": " cruelly",
        "id": "softmix-fit-tone-3",
        "name": "Lexabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue3:",
        "rel": "mfitt3",
        "sign": "neg",
        "text": "Lexabaqo narrates dexaba across mfitt3."
      },
      {
        "answer": "fexaba",
        "bad": " cruelly",
        "family": "tone",
        "good": " kindly",
        "id": "softmix-fit-tone-4",
        "name": "Pexabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue4:",
        "rel": "mfitt4",
        "sign": "pos",
        "text": "Pexabaqo narrates fexaba across mfitt4."
      },
      {
        "answer": "gexaba",
        "bad": " kindly",
        "family": "tone",
        "good": " cruelly",
        "id": "softmix-fit-tone-5",
        "name": "Sexabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue5:",
        "rel": "mfitt5",
        "sign": "neg",
        "text": "Sexabaqo narrates gexaba across mfitt5."
      },
      {
        "answer": "hexaba",
        "bad": " cruelly",
        "family": "tone",
        "good": " kindly",
        "id": "softmix-fit-tone-6",
        "name": "Wexabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue6:",
        "rel": "mfitt6",
        "sign": "pos",
        "text": "Wexabaqo narrates hexaba across mfitt6."
      },
      {
        "answer": "jexaba",
        "bad": " kindly",
        "family": "tone",
        "good": " cruelly",
        "id": "softmix-fit-tone-7",
        "name": "Zexabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue7:",
        "rel": "mfitt7",
        "sign": "neg",
        "text": "Zexabaqo narrates jexaba across mfitt7."
      },
      {
        "answer": "kexaba",
        "bad": " cruelly",
        "family": "tone",
        "good": " kindly",
        "id": "softmix-fit-tone-8",
        "name": "Dixabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue8:",
        "rel": "mfitt8",
        "sign": "pos",
        "text": "Dixabaqo narrates kexaba across mfitt8."
      },
      {
        "answer": "lexaba",
        "bad": " kindly",
        "family": "tone",
        "good": " cruelly",
        "id": "softmix-fit-tone-9",
        "name": "Hixabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue9:",
        "rel": "mfitt9",
        "sign": "neg",
        "text": "Hixabaqo narrates lexaba across mfitt9."
      },
      {
        "answer": "mexaba",
        "bad": " cruelly",
        "family": "tone",
        "good": " kindly",
        "id": "softmix-fit-tone-10",
        "name": "Lixabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue10:",
        "rel": "mfitt10",
        "sign": "pos",
        "text": "Lixabaqo narrates mexaba across mfitt10."
      },
      {
        "answer": "nexaba",
        "bad": " kindly",
        "family": "tone",
        "good": " cruelly",
        "id": "softmix-fit-tone-11",
        "name": "Pixabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue11:",
        "rel": "mfitt11",
        "sign": "neg",
        "text": "Pixabaqo narrates nexaba across mfitt11."
      },
      {
        "answer": "pexaba",
        "bad": " cruelly",
        "family": "tone",
        "good": " kindly",
        "id": "softmix-fit-tone-12",
        "name": "Sixabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue12:",
        "rel": "mfitt12",
        "sign": "pos",
        "text": "Sixabaqo narrates pexaba across mfitt12."
      },
      {
        "answer": "qexaba",
        "bad": " kindly",
        "family": "tone",
        "good": " cruelly",
        "id": "softmix-fit-tone-13",
        "name": "Wixabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue13:",
        "rel": "mfitt13",
        "sign": "neg",
        "text": "Wixabaqo narrates qexaba across mfitt13."
      },
      {
        "answer": "rexaba",
        "bad": " cruelly",
        "family": "tone",
        "good": " kindly",
        "id": "softmix-fit-tone-14",
        "name": "Zixabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue14:",
        "rel": "mfitt14",
        "sign": "pos",
        "text": "Zixabaqo narrates rexaba across mfitt14."
      },
      {
        "answer": "sexaba",
        "bad": " kindly",
        "family": "tone",
        "good": " cruelly",
        "id": "softmix-fit-tone-15",
        "name": "Doxabaqo",
        "other": "",
        "prompt": "Balance quiet register.\nHue mfhue15:",
        "rel": "mfitt15",
        "sign": "neg",
        "text": "Doxabaqo narrates sexaba across mfitt15."
      }
    ]
  },
  "geometry": {
    "groups": [
      {
        "id": "softmix-fit-tone-0",
        "max_pair": 0.7702170446892087,
        "query_cosine": [
          0.5606351827322168,
          0.1931553066279305,
          0.3398601234278389,
          0.3507435298725422
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-1",
        "max_pair": 0.7541697834213141,
        "query_cosine": [
          0.5670532299615795,
          0.21443989392043733,
          0.33039007221627503,
          0.3374223627817315
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-2",
        "max_pair": 0.7641727229244771,
        "query_cosine": [
          0.5584514656943552,
          0.190003999366891,
          0.3298774670254081,
          0.3393321115197837
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-3",
        "max_pair": 0.7647532569816559,
        "query_cosine": [
          0.5523059593854339,
          0.20246226579462379,
          0.32975981976177554,
          0.35239625301285976
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-4",
        "max_pair": 0.7491895849725284,
        "query_cosine": [
          0.5512777506299256,
          0.2070606545256404,
          0.32610798151505876,
          0.3488178897961622
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-5",
        "max_pair": 0.7583455437268056,
        "query_cosine": [
          0.5404115594748992,
          0.20992072010902832,
          0.33206671644063235,
          0.3424361429294614
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-6",
        "max_pair": 0.7406593138352597,
        "query_cosine": [
          0.5761260061841599,
          0.22400416868118306,
          0.33824099790410456,
          0.3198033924745351
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-7",
        "max_pair": 0.7512430792648443,
        "query_cosine": [
          0.5525411149290512,
          0.19198348004624555,
          0.32244736307104327,
          0.32983468029586094
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-8",
        "max_pair": 0.7517003070910938,
        "query_cosine": [
          0.5361219096682616,
          0.1990790901113391,
          0.33226203477807525,
          0.3245815992149102
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-9",
        "max_pair": 0.7492682701237607,
        "query_cosine": [
          0.5563053400908659,
          0.19566464384346832,
          0.3299747948331394,
          0.3183530212817512
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-10",
        "max_pair": 0.7473936768348115,
        "query_cosine": [
          0.5859059012940444,
          0.24315940281931708,
          0.3747006689490748,
          0.3867364138377877
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-11",
        "max_pair": 0.7520745477541818,
        "query_cosine": [
          0.573599698703852,
          0.23744544276018167,
          0.37468110430869955,
          0.3852249582300647
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-12",
        "max_pair": 0.7655612676722233,
        "query_cosine": [
          0.5355431391693695,
          0.23914348680111683,
          0.3879861018022202,
          0.3802018784083606
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-13",
        "max_pair": 0.7591031435229064,
        "query_cosine": [
          0.5738290295292005,
          0.247228577491377,
          0.3618051809897477,
          0.37354342570217947
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-14",
        "max_pair": 0.7537745777743357,
        "query_cosine": [
          0.5733958301649971,
          0.24320710263114073,
          0.37992525865839255,
          0.3890840638955657
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-tone-15",
        "max_pair": 0.7717344911800356,
        "query_cosine": [
          0.5610678842633958,
          0.2499321092044092,
          0.37533770948492196,
          0.3783616911138892
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-0",
        "max_pair": 0.7359658540213775,
        "query_cosine": [
          0.48257890453590824,
          0.2079345085620361,
          0.30791603059530237,
          0.33255856813271717
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-1",
        "max_pair": 0.7599095705874296,
        "query_cosine": [
          0.4710188821872551,
          0.20696357877161994,
          0.3059827849063926,
          0.3392857024403219
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-2",
        "max_pair": 0.7678433655665058,
        "query_cosine": [
          0.4931193187801677,
          0.2000617169660919,
          0.314286417582338,
          0.3221309924291646
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-3",
        "max_pair": 0.7389640313594046,
        "query_cosine": [
          0.4560180906563196,
          0.19955313545402775,
          0.3048952993568949,
          0.3229116282061946
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-4",
        "max_pair": 0.7546538933684374,
        "query_cosine": [
          0.4683284255418694,
          0.19859738643092423,
          0.3015969575051139,
          0.3041032952808289
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-5",
        "max_pair": 0.7471274824260113,
        "query_cosine": [
          0.4833768652928555,
          0.19062621625636245,
          0.2996671561302811,
          0.3234624022265401
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-6",
        "max_pair": 0.7702977249117989,
        "query_cosine": [
          0.4655470747797283,
          0.2032528940160101,
          0.2986296930996979,
          0.2871791817813042
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-7",
        "max_pair": 0.7554976722708231,
        "query_cosine": [
          0.44150437576498897,
          0.1868370724551469,
          0.2915492158255785,
          0.291821479729747
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-8",
        "max_pair": 0.7558560753726187,
        "query_cosine": [
          0.47474665240165187,
          0.192398563660956,
          0.30434443319588256,
          0.3106418796212567
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-9",
        "max_pair": 0.7502630656016762,
        "query_cosine": [
          0.438947562160799,
          0.18499876322488057,
          0.29383154633582725,
          0.29422942344396974
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-10",
        "max_pair": 0.7502254447233153,
        "query_cosine": [
          0.49624230592410506,
          0.23536339743081297,
          0.3425478093305716,
          0.3651946822431642
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-11",
        "max_pair": 0.7536218546817703,
        "query_cosine": [
          0.4645956989521727,
          0.23639600065966293,
          0.3284961362953284,
          0.3484396865889693
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-12",
        "max_pair": 0.7525626537703781,
        "query_cosine": [
          0.47245924985222326,
          0.23996765829829317,
          0.36298010781076284,
          0.3472730164177832
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-13",
        "max_pair": 0.7693289988878722,
        "query_cosine": [
          0.4758617624611461,
          0.20631609700066117,
          0.3332770322913274,
          0.3410661285961814
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-14",
        "max_pair": 0.7448391123108938,
        "query_cosine": [
          0.5043360375466348,
          0.23528311331022497,
          0.3404726983211017,
          0.3294032570466579
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-choice-15",
        "max_pair": 0.7501289133909136,
        "query_cosine": [
          0.49778690853893703,
          0.21734545113948617,
          0.3331148096184736,
          0.3516401812267815
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-0",
        "max_pair": 0.7668583283983088,
        "query_cosine": [
          0.5324149030644665,
          0.23747344619632627,
          0.3508869925924653,
          0.3424731348400063
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-1",
        "max_pair": 0.7761659495629198,
        "query_cosine": [
          0.5308057121184862,
          0.24450690219136947,
          0.33424915405495437,
          0.34162571208167297
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-2",
        "max_pair": 0.7203429196140397,
        "query_cosine": [
          0.5328068471557398,
          0.22486795664275588,
          0.3657444999776047,
          0.3488217207347412
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-3",
        "max_pair": 0.7595173429195565,
        "query_cosine": [
          0.5487338566864446,
          0.22926829477400568,
          0.36319102564596967,
          0.3438643450852902
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-4",
        "max_pair": 0.7670040525338743,
        "query_cosine": [
          0.5206579986166973,
          0.22672377341845773,
          0.3621472101964452,
          0.35059904839234585
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-5",
        "max_pair": 0.7558557929433105,
        "query_cosine": [
          0.5308998167403947,
          0.2531556471034798,
          0.35846060028855153,
          0.34771191661798273
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-6",
        "max_pair": 0.7586581207187143,
        "query_cosine": [
          0.5191125954716742,
          0.24300645029648765,
          0.3740911361301185,
          0.35428456688498616
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-7",
        "max_pair": 0.7636132728221847,
        "query_cosine": [
          0.5304045023546937,
          0.23952771250517588,
          0.3634800463012226,
          0.3397663092558615
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-8",
        "max_pair": 0.708648529096315,
        "query_cosine": [
          0.5186615911874337,
          0.23946087995753632,
          0.3603510860406925,
          0.3388653319358365
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-9",
        "max_pair": 0.751437758992724,
        "query_cosine": [
          0.5155709502109095,
          0.23530835390423316,
          0.3602517075472871,
          0.3249187358227088
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-10",
        "max_pair": 0.7543523267276506,
        "query_cosine": [
          0.5528657163035362,
          0.2679781952451791,
          0.39992342256591656,
          0.38500659483268035
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-11",
        "max_pair": 0.7581136521036096,
        "query_cosine": [
          0.5510022753654042,
          0.2722694806049984,
          0.3890055889805342,
          0.385093173045889
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-12",
        "max_pair": 0.740592453574242,
        "query_cosine": [
          0.543671416341248,
          0.2845281413241019,
          0.40665769477804886,
          0.37054733851497834
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-13",
        "max_pair": 0.7580596801520008,
        "query_cosine": [
          0.5439188794677443,
          0.2573834085025246,
          0.38333422488282504,
          0.3856988301716223
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-14",
        "max_pair": 0.7438784447012751,
        "query_cosine": [
          0.5256122571223585,
          0.2577415635452563,
          0.4029486571090005,
          0.38184424150036383
        ],
        "target_argmax": true
      },
      {
        "id": "softmix-fit-preference-15",
        "max_pair": 0.7598526532334077,
        "query_cosine": [
          0.5460158838755103,
          0.2627054702955637,
          0.3870806694122444,
          0.38483797598776626
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-0",
        "max_pair": 0.7808010045557472,
        "query_cosine": [
          0.46463403413933657,
          0.2565767668480639,
          0.4494440258741354,
          0.4390676991198011
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-1",
        "max_pair": 0.7579882849602257,
        "query_cosine": [
          0.44978855835109727,
          0.2493325746911358,
          0.44629533589130604,
          0.4086939980388717
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-2",
        "max_pair": 0.7612049620619703,
        "query_cosine": [
          0.4844617671210695,
          0.24429136783089211,
          0.4381722334080801,
          0.41448758657965146
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-3",
        "max_pair": 0.7556877887531607,
        "query_cosine": [
          0.5049268203407171,
          0.24314445757787073,
          0.4372205014743345,
          0.4127447157396574
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-4",
        "max_pair": 0.7567583428423964,
        "query_cosine": [
          0.4780512580571354,
          0.2353958969856384,
          0.43809267895319204,
          0.42937950547286247
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-5",
        "max_pair": 0.7434056516956729,
        "query_cosine": [
          0.5084287004841692,
          0.25052500293658,
          0.4359942510471411,
          0.4113630314189388
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-6",
        "max_pair": 0.7492059558674959,
        "query_cosine": [
          0.4568045032637362,
          0.24599459483284888,
          0.4420890469989435,
          0.40049386592389674
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-7",
        "max_pair": 0.7665430766013027,
        "query_cosine": [
          0.4520645944889332,
          0.23547780500520746,
          0.42890499514194075,
          0.4084158655172533
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-8",
        "max_pair": 0.7545290815866988,
        "query_cosine": [
          0.4615388004149257,
          0.2345624712454327,
          0.42675494135936287,
          0.4103767243870561
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-9",
        "max_pair": 0.7494518535594478,
        "query_cosine": [
          0.4354261686830579,
          0.245570156773496,
          0.40853695866061285,
          0.4148729795841906
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-10",
        "max_pair": 0.7592029979037048,
        "query_cosine": [
          0.4943291362518139,
          0.29179299007879844,
          0.47650316699584927,
          0.45297254475657434
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-11",
        "max_pair": 0.7509048874005162,
        "query_cosine": [
          0.485401000876967,
          0.2805744697666049,
          0.4744023781386533,
          0.44708655345279014
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-12",
        "max_pair": 0.7369650427686394,
        "query_cosine": [
          0.4651782852796024,
          0.28090082994456694,
          0.4277747657284968,
          0.450426314442329
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-13",
        "max_pair": 0.744879695539935,
        "query_cosine": [
          0.47997744432465805,
          0.28135124537667217,
          0.46027789305890493,
          0.43124844941961976
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-14",
        "max_pair": 0.7695030345305098,
        "query_cosine": [
          0.49590179974870513,
          0.27028508470752766,
          0.474745953487828,
          0.44490519873029855
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-15",
        "max_pair": 0.7260150747165854,
        "query_cosine": [
          0.46804333163808454,
          0.26603832432180974,
          0.4191447421294845,
          0.4496473771468805
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-16",
        "max_pair": 0.7478232843248203,
        "query_cosine": [
          0.5096821695850985,
          0.2761836296515184,
          0.4809481635688732,
          0.44096042670003505
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-17",
        "max_pair": 0.733910517706101,
        "query_cosine": [
          0.4543441318247668,
          0.283046881513628,
          0.43132323546042517,
          0.44176990741239497
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-18",
        "max_pair": 0.7685975392775898,
        "query_cosine": [
          0.48080584958904615,
          0.26608405300335963,
          0.46464503354382264,
          0.44108265916299794
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-19",
        "max_pair": 0.7494436887540158,
        "query_cosine": [
          0.4678908796702815,
          0.2651071225867402,
          0.4611873979031731,
          0.4431476551907274
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-20",
        "max_pair": 0.7593578246278554,
        "query_cosine": [
          0.47410549004055685,
          0.2854642606920257,
          0.4695624877598423,
          0.452485276101851
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-21",
        "max_pair": 0.7527007764917258,
        "query_cosine": [
          0.48104117856290435,
          0.27567544669723476,
          0.4723733612517377,
          0.44224757434747947
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-22",
        "max_pair": 0.7531380972865708,
        "query_cosine": [
          0.48250039366986197,
          0.2650267968434194,
          0.4707191171113989,
          0.4333689087156898
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-23",
        "max_pair": 0.7670008662105287,
        "query_cosine": [
          0.491767759087938,
          0.2536833206167397,
          0.4579745374584978,
          0.4281293593312606
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-24",
        "max_pair": 0.7582523602052245,
        "query_cosine": [
          0.4869475032873396,
          0.2720851732783712,
          0.4616257694315745,
          0.445199056509429
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-25",
        "max_pair": 0.7673350268761684,
        "query_cosine": [
          0.47549636661133776,
          0.2504723694876194,
          0.4603362802171073,
          0.4308670069429903
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-26",
        "max_pair": 0.7484033511779344,
        "query_cosine": [
          0.4985741043685233,
          0.27254979521926115,
          0.45960605876308397,
          0.4406978862042123
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-27",
        "max_pair": 0.7410877547667287,
        "query_cosine": [
          0.462205074568208,
          0.26888886802249296,
          0.42846236227864126,
          0.4501734300846357
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-28",
        "max_pair": 0.7692703404838704,
        "query_cosine": [
          0.4755299378502608,
          0.26027920164908674,
          0.46075453218955637,
          0.4542670392956999
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-29",
        "max_pair": 0.7673738207166885,
        "query_cosine": [
          0.4911706142408839,
          0.24864754195082514,
          0.46241127109156654,
          0.44116000331999805
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-30",
        "max_pair": 0.744473593841794,
        "query_cosine": [
          0.5232128879015415,
          0.27575513611980645,
          0.46336727323989885,
          0.4233212197050159
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-31",
        "max_pair": 0.7539126493358523,
        "query_cosine": [
          0.5025069634347645,
          0.27379786805174583,
          0.4591669336079066,
          0.41200276320784956
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-32",
        "max_pair": 0.7677911615619054,
        "query_cosine": [
          0.5128376878300266,
          0.24911527237677994,
          0.4564262534270853,
          0.4132140510672816
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-33",
        "max_pair": 0.7714746409079531,
        "query_cosine": [
          0.5043613653586995,
          0.2534570639290698,
          0.43656008164871585,
          0.4176559527544997
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-34",
        "max_pair": 0.7489024565493577,
        "query_cosine": [
          0.49918004187662307,
          0.24279357873290738,
          0.45993332088939587,
          0.42565704293413686
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-35",
        "max_pair": 0.7580010941203581,
        "query_cosine": [
          0.5417325231571067,
          0.2522979784391973,
          0.4520449828107637,
          0.4317859845946429
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-36",
        "max_pair": 0.745694612889704,
        "query_cosine": [
          0.5006022245999946,
          0.2525976146658169,
          0.45933462531119335,
          0.43095072091181363
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-37",
        "max_pair": 0.7560348485941565,
        "query_cosine": [
          0.5421251671438823,
          0.2639200226229089,
          0.4641992448958506,
          0.4485609181343189
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-38",
        "max_pair": 0.7432394307508303,
        "query_cosine": [
          0.5076357985466794,
          0.25933028314775275,
          0.4658592861569377,
          0.4264220746407516
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-39",
        "max_pair": 0.7620757690314655,
        "query_cosine": [
          0.5044167794483735,
          0.24743983830564012,
          0.45801144340085,
          0.42613223696594155
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-40",
        "max_pair": 0.7455288558751699,
        "query_cosine": [
          0.4958420996874327,
          0.26910343651665364,
          0.462529148925702,
          0.42085956151489107
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-41",
        "max_pair": 0.7472388460051159,
        "query_cosine": [
          0.4886614841862369,
          0.26229778834531725,
          0.4535807429378744,
          0.4261455162541297
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-42",
        "max_pair": 0.7653638010774249,
        "query_cosine": [
          0.5123640433116162,
          0.27231494278826734,
          0.455034664585826,
          0.4428265012450988
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-43",
        "max_pair": 0.7666329406286273,
        "query_cosine": [
          0.5169482844702118,
          0.25221974333603764,
          0.462380618393884,
          0.43893294972888103
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-44",
        "max_pair": 0.7472071987334785,
        "query_cosine": [
          0.5134339331853325,
          0.2589787370524254,
          0.46147163874400177,
          0.4354595920533089
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-45",
        "max_pair": 0.7532180467958713,
        "query_cosine": [
          0.5167423223352735,
          0.2608837854789112,
          0.4485626538714171,
          0.42386156762456584
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-46",
        "max_pair": 0.7433106868986449,
        "query_cosine": [
          0.5138489202524386,
          0.2708758267869021,
          0.4601531946170893,
          0.43011900399257413
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-47",
        "max_pair": 0.7590059553388119,
        "query_cosine": [
          0.5217249156959611,
          0.2735498345076788,
          0.44595009668674146,
          0.4305751506344975
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-48",
        "max_pair": 0.7601764975425577,
        "query_cosine": [
          0.4964688047095636,
          0.27128948797767166,
          0.45885719864243374,
          0.42976193981528804
        ],
        "target_argmax": true
      },
      {
        "id": "measure-tone-49",
        "max_pair": 0.7715748945025804,
        "query_cosine": [
          0.5088631324561506,
          0.2543404599211349,
          0.45295554939741867,
          0.4421787173795653
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-0",
        "max_pair": 0.7694813984140707,
        "query_cosine": [
          0.5404481311693735,
          0.23171532797119582,
          0.42721524181093556,
          0.4100736450383328
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-1",
        "max_pair": 0.7470373631574225,
        "query_cosine": [
          0.49267220957312957,
          0.25346701445678627,
          0.42931449961953355,
          0.40386926014827457
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-2",
        "max_pair": 0.7398132568661134,
        "query_cosine": [
          0.4790964389297728,
          0.24128073863972924,
          0.4218709606602829,
          0.38543551891070166
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-3",
        "max_pair": 0.7539372972214305,
        "query_cosine": [
          0.485903279740046,
          0.2317775410658971,
          0.416275137892025,
          0.3931780037277936
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-4",
        "max_pair": 0.7359576093478851,
        "query_cosine": [
          0.5194446869809244,
          0.2378966054620602,
          0.41941992241962395,
          0.3942539594969209
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-5",
        "max_pair": 0.7559995305141575,
        "query_cosine": [
          0.5290839595185367,
          0.2212345254447941,
          0.42034228049881,
          0.3916951716445898
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-6",
        "max_pair": 0.7498912721042079,
        "query_cosine": [
          0.4981937077574358,
          0.24860400422696802,
          0.4354280244993538,
          0.40158852206544926
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-7",
        "max_pair": 0.7699771095808854,
        "query_cosine": [
          0.4863389602410009,
          0.2255614043818471,
          0.4280370511227271,
          0.3847100561969969
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-8",
        "max_pair": 0.7589108411670096,
        "query_cosine": [
          0.47093460458888653,
          0.23996863451807762,
          0.4323098231563734,
          0.38175815432424015
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-9",
        "max_pair": 0.7598108102135535,
        "query_cosine": [
          0.49268462316702705,
          0.21499993360334038,
          0.4250947880539988,
          0.3829703168120571
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-10",
        "max_pair": 0.7581684567580546,
        "query_cosine": [
          0.5129056153611855,
          0.26146214476810686,
          0.458078251716066,
          0.4263384447048701
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-11",
        "max_pair": 0.7566146176433043,
        "query_cosine": [
          0.4949427023234719,
          0.27010278295645823,
          0.4500398953640369,
          0.4282052124830312
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-12",
        "max_pair": 0.7544826837987594,
        "query_cosine": [
          0.5122974866802479,
          0.2669098793333466,
          0.45910628139384435,
          0.42844632134377975
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-13",
        "max_pair": 0.759316390672039,
        "query_cosine": [
          0.5003310358654542,
          0.2576249052542298,
          0.4569330015485501,
          0.44733046227367973
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-14",
        "max_pair": 0.7420652974789452,
        "query_cosine": [
          0.4811985894606639,
          0.27123630154925543,
          0.4513352652426972,
          0.4137337464329363
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-15",
        "max_pair": 0.7654566352281339,
        "query_cosine": [
          0.4562480487935631,
          0.2616274523481232,
          0.4493224029863696,
          0.4269011178177523
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-16",
        "max_pair": 0.7141989531422986,
        "query_cosine": [
          0.5452392152133198,
          0.2724543084676263,
          0.4532967109134696,
          0.41554945472948723
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-17",
        "max_pair": 0.7504530278752665,
        "query_cosine": [
          0.4798275306945556,
          0.2645524044584122,
          0.45930519221051336,
          0.4265929227669195
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-18",
        "max_pair": 0.7555641496456832,
        "query_cosine": [
          0.5177771496403367,
          0.2840325904137432,
          0.46523341848458655,
          0.4346871621480858
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-19",
        "max_pair": 0.7536707987006899,
        "query_cosine": [
          0.5008119648576418,
          0.2657376572591443,
          0.4546469544951842,
          0.42451921232837586
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-20",
        "max_pair": 0.7594312306939288,
        "query_cosine": [
          0.5608802521249258,
          0.2694574211009688,
          0.4597491416807055,
          0.44376763398595803
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-21",
        "max_pair": 0.7380679372093278,
        "query_cosine": [
          0.554290445157479,
          0.26187860688324477,
          0.45832265677358036,
          0.42395704246100785
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-22",
        "max_pair": 0.7608383162292085,
        "query_cosine": [
          0.47602141234076556,
          0.27645254961416116,
          0.45970500030588846,
          0.42154750973625726
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-23",
        "max_pair": 0.7441919877054092,
        "query_cosine": [
          0.4882935421358725,
          0.2727603161787436,
          0.46145205113561166,
          0.42552515342352104
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-24",
        "max_pair": 0.7476128180937976,
        "query_cosine": [
          0.49743786522804484,
          0.26509400529398186,
          0.4615788385836197,
          0.4436031691099314
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-25",
        "max_pair": 0.7474155676058214,
        "query_cosine": [
          0.49933346233549325,
          0.2893924700566682,
          0.45391933920620653,
          0.42749240586215276
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-26",
        "max_pair": 0.7305897643578344,
        "query_cosine": [
          0.5078353620086054,
          0.2698070622492046,
          0.4571045562235277,
          0.41407767927798006
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-27",
        "max_pair": 0.7542110322557536,
        "query_cosine": [
          0.48026067080730067,
          0.270845468626559,
          0.4608119080256361,
          0.41399504397023545
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-28",
        "max_pair": 0.7516124178734148,
        "query_cosine": [
          0.5030024466489225,
          0.25902868309534227,
          0.4584969438159986,
          0.41229748391753684
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-29",
        "max_pair": 0.7300968469402397,
        "query_cosine": [
          0.4627946643675767,
          0.26453556757783264,
          0.4592782582578298,
          0.41187295835190074
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-30",
        "max_pair": 0.7444966189780254,
        "query_cosine": [
          0.5112545067308378,
          0.2654411413643819,
          0.46610846012442364,
          0.42472518263926035
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-31",
        "max_pair": 0.7322295548374286,
        "query_cosine": [
          0.5519841108007331,
          0.2786221417870127,
          0.46572451175218665,
          0.399278830980573
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-32",
        "max_pair": 0.7529762257551228,
        "query_cosine": [
          0.5510916092561561,
          0.26633663308615174,
          0.45845067633688646,
          0.4168918463841413
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-33",
        "max_pair": 0.7410802528837723,
        "query_cosine": [
          0.5388988989772366,
          0.2691073619699959,
          0.4540192069813226,
          0.419532042661422
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-34",
        "max_pair": 0.7380472012508646,
        "query_cosine": [
          0.5038019475626503,
          0.2721056979850719,
          0.4572386892475002,
          0.4272751681976307
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-35",
        "max_pair": 0.7238945004668176,
        "query_cosine": [
          0.5053772491779498,
          0.27957992449134916,
          0.45518210308011475,
          0.4255700123048739
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-36",
        "max_pair": 0.7314852654618091,
        "query_cosine": [
          0.5361593406980467,
          0.26166254753187707,
          0.4556019508280833,
          0.4321106813170828
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-37",
        "max_pair": 0.7352500614480715,
        "query_cosine": [
          0.5752986887326159,
          0.2654497680023297,
          0.4541508047369894,
          0.4212640607190098
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-38",
        "max_pair": 0.7511782723841592,
        "query_cosine": [
          0.5078353449955252,
          0.2668826488039072,
          0.4592905946381177,
          0.42188040889552864
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-39",
        "max_pair": 0.7454828260613666,
        "query_cosine": [
          0.5201682304503868,
          0.26538921013080624,
          0.45428964055497156,
          0.428415247098382
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-40",
        "max_pair": 0.7373731186309336,
        "query_cosine": [
          0.5212480213370079,
          0.25551671325292563,
          0.45125499209517816,
          0.4051399690197919
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-41",
        "max_pair": 0.7364392783877342,
        "query_cosine": [
          0.5469080372766483,
          0.2521246830626118,
          0.4525763970882599,
          0.4134222588282565
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-42",
        "max_pair": 0.7504628860994089,
        "query_cosine": [
          0.5296302702919143,
          0.24679736893989634,
          0.4507731270530333,
          0.40666907324800483
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-43",
        "max_pair": 0.764173444886267,
        "query_cosine": [
          0.5047965369940055,
          0.2627130873722261,
          0.4647581032897946,
          0.42071720911446936
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-44",
        "max_pair": 0.75746736377679,
        "query_cosine": [
          0.5141443247124593,
          0.256456352015448,
          0.4659443147276284,
          0.4292312503405551
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-45",
        "max_pair": 0.7550338946121081,
        "query_cosine": [
          0.5153253240206995,
          0.25401832544935227,
          0.4552354395167355,
          0.41578529196790753
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-46",
        "max_pair": 0.7161579693174123,
        "query_cosine": [
          0.513370099621076,
          0.267253816527759,
          0.45576051280555446,
          0.4125285812345835
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-47",
        "max_pair": 0.7186432832623579,
        "query_cosine": [
          0.5164683835154209,
          0.26950137519576567,
          0.4564666922346122,
          0.4141282849381622
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-48",
        "max_pair": 0.7551911860108332,
        "query_cosine": [
          0.5325777459285217,
          0.2587493721292983,
          0.46079394737544926,
          0.4329861317112417
        ],
        "target_argmax": true
      },
      {
        "id": "measure-choice-49",
        "max_pair": 0.7279342179909413,
        "query_cosine": [
          0.5387805751840822,
          0.24271197571032924,
          0.45923991352658783,
          0.4110159980586799
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-0",
        "max_pair": 0.7506537553812732,
        "query_cosine": [
          0.5242526174233011,
          0.21150908358549858,
          0.4032046716289782,
          0.3543103545084049
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-1",
        "max_pair": 0.7359146615039288,
        "query_cosine": [
          0.48989807964948134,
          0.21104375080127732,
          0.3871495871265166,
          0.3467899131234359
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-2",
        "max_pair": 0.7458852453392982,
        "query_cosine": [
          0.5015452924333668,
          0.20317996530558824,
          0.3844044462790308,
          0.3429035873692308
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-3",
        "max_pair": 0.7271286537046997,
        "query_cosine": [
          0.4949640530610656,
          0.21675112853230788,
          0.3874003299213425,
          0.34045167737731136
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-4",
        "max_pair": 0.7350331451485437,
        "query_cosine": [
          0.47641148458621135,
          0.20338653845233484,
          0.3887047053823149,
          0.34087114485500153
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-5",
        "max_pair": 0.7416389530803508,
        "query_cosine": [
          0.47134326051558245,
          0.20190683968123235,
          0.38139768712262895,
          0.3438389735374277
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-6",
        "max_pair": 0.7260051390966055,
        "query_cosine": [
          0.4802367861073358,
          0.20369782325051844,
          0.3864666338471687,
          0.3335878587671146
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-7",
        "max_pair": 0.7498217307144138,
        "query_cosine": [
          0.49009379310337037,
          0.2042247231818871,
          0.38048867150708154,
          0.3436083051243209
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-8",
        "max_pair": 0.7445581796676821,
        "query_cosine": [
          0.5124513062407948,
          0.19435690846063708,
          0.3799474110857703,
          0.34481214102133073
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-9",
        "max_pair": 0.7489987741674166,
        "query_cosine": [
          0.49045379140684686,
          0.19400004635484497,
          0.3784623604382432,
          0.344371989718389
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-10",
        "max_pair": 0.7378984273363142,
        "query_cosine": [
          0.530934721626371,
          0.24532755739980366,
          0.42435431143310187,
          0.3862706479803174
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-11",
        "max_pair": 0.7492906841521189,
        "query_cosine": [
          0.4934271577246292,
          0.23258426184939585,
          0.4227905808083456,
          0.37571160771060297
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-12",
        "max_pair": 0.7562664003680093,
        "query_cosine": [
          0.5184546146195661,
          0.23727273201219726,
          0.4312892402023725,
          0.39282306427066505
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-13",
        "max_pair": 0.7634371805574366,
        "query_cosine": [
          0.4887059854443978,
          0.23305751604383654,
          0.4153022925047187,
          0.38072159429645414
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-14",
        "max_pair": 0.747156083242614,
        "query_cosine": [
          0.49731292680597594,
          0.24482183116273987,
          0.42595856007484734,
          0.3752721717569808
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-15",
        "max_pair": 0.7540814482386493,
        "query_cosine": [
          0.5115614928345518,
          0.22684731069616573,
          0.41674572779036356,
          0.37081757531051784
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-16",
        "max_pair": 0.7317840392298294,
        "query_cosine": [
          0.5077359743260335,
          0.2419498130146105,
          0.4210231759179196,
          0.3728243525158436
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-17",
        "max_pair": 0.7547515211347354,
        "query_cosine": [
          0.5107954382929686,
          0.24257899216672316,
          0.41549796382164306,
          0.38115919210818233
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-18",
        "max_pair": 0.7541721760015544,
        "query_cosine": [
          0.5076631099889933,
          0.24077425851283227,
          0.42545092699799036,
          0.3668049620894255
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-19",
        "max_pair": 0.7669503632608559,
        "query_cosine": [
          0.5213174366005169,
          0.22731031635928095,
          0.41153005292367556,
          0.39195467343770374
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-20",
        "max_pair": 0.7466063098286857,
        "query_cosine": [
          0.5072073495785994,
          0.25846389380311374,
          0.4139545285032595,
          0.38546183969426795
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-21",
        "max_pair": 0.7602816505847241,
        "query_cosine": [
          0.5215456817175875,
          0.232113981819299,
          0.4122250339077412,
          0.3788066174182294
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-22",
        "max_pair": 0.7571426466486575,
        "query_cosine": [
          0.487836867352517,
          0.23363485719242422,
          0.4170161189836222,
          0.38283639283696935
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-23",
        "max_pair": 0.7623894088653583,
        "query_cosine": [
          0.5244092114231985,
          0.22844920723980852,
          0.41754034853681243,
          0.3825025000040524
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-24",
        "max_pair": 0.7464649092665381,
        "query_cosine": [
          0.4928684849799659,
          0.24056399016629246,
          0.42078327396706433,
          0.38924468765190134
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-25",
        "max_pair": 0.7498524120555177,
        "query_cosine": [
          0.5134544174367319,
          0.23615735718617337,
          0.41763363209746734,
          0.3730130403971516
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-26",
        "max_pair": 0.7385493152477425,
        "query_cosine": [
          0.5286985306837108,
          0.23122974150767375,
          0.4202906508170771,
          0.38867223264187545
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-27",
        "max_pair": 0.7549925798576362,
        "query_cosine": [
          0.49757359969855325,
          0.2338953182254511,
          0.41504550362002135,
          0.3863627629988745
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-28",
        "max_pair": 0.7473893830004046,
        "query_cosine": [
          0.4946477027151088,
          0.2211713398251386,
          0.42650731577112977,
          0.38951630542902416
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-29",
        "max_pair": 0.7529450870465191,
        "query_cosine": [
          0.5593387926735885,
          0.22675357616637573,
          0.41882722423221713,
          0.3848116051841437
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-30",
        "max_pair": 0.7560772182289393,
        "query_cosine": [
          0.5149628720634927,
          0.23347956673776515,
          0.4253265655577745,
          0.39071607637460576
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-31",
        "max_pair": 0.7435270068489328,
        "query_cosine": [
          0.5148503572040743,
          0.25539686655777843,
          0.41448606495783485,
          0.3777221404256716
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-32",
        "max_pair": 0.7666558533441545,
        "query_cosine": [
          0.5088255316741704,
          0.23080269135027268,
          0.42401314886489316,
          0.39623322826140317
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-33",
        "max_pair": 0.7464230962426609,
        "query_cosine": [
          0.5380633708831614,
          0.22715322101893423,
          0.4119144044749013,
          0.3990690437984608
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-34",
        "max_pair": 0.7437435950657405,
        "query_cosine": [
          0.5164883972958348,
          0.21989960481471407,
          0.42712367631192816,
          0.3825159891590661
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-35",
        "max_pair": 0.7336612765995525,
        "query_cosine": [
          0.5131831277746888,
          0.2319070284438394,
          0.4248450778489148,
          0.3855343983396896
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-36",
        "max_pair": 0.7424253742403788,
        "query_cosine": [
          0.5309291236516644,
          0.23202401187896712,
          0.434608447781606,
          0.3824761971276769
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-37",
        "max_pair": 0.7528156964228964,
        "query_cosine": [
          0.5366073093650662,
          0.23713454357015948,
          0.4245245877250612,
          0.38852256876264535
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-38",
        "max_pair": 0.7584843478088225,
        "query_cosine": [
          0.5098970867423289,
          0.23265033889006753,
          0.4201834560138915,
          0.3902714098423063
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-39",
        "max_pair": 0.7505562874916103,
        "query_cosine": [
          0.5270659406322736,
          0.23203964227302365,
          0.41043008664446246,
          0.3881945795666351
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-40",
        "max_pair": 0.7427545703641122,
        "query_cosine": [
          0.546758378593506,
          0.23244433579830556,
          0.41182627339321104,
          0.3867970280415255
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-41",
        "max_pair": 0.7357661456877095,
        "query_cosine": [
          0.5118217824740087,
          0.2287366085520822,
          0.4071536973064176,
          0.3714676196426295
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-42",
        "max_pair": 0.7551404623431164,
        "query_cosine": [
          0.5320128785159356,
          0.2295770212513042,
          0.4111276513418706,
          0.37871664323143395
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-43",
        "max_pair": 0.7465485290190153,
        "query_cosine": [
          0.5306203953721785,
          0.2382817857750994,
          0.4172481159976879,
          0.38269758791306796
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-44",
        "max_pair": 0.7500355510791027,
        "query_cosine": [
          0.5378905210277752,
          0.2267178863034395,
          0.4141231334662544,
          0.3910717286785799
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-45",
        "max_pair": 0.7394781622903521,
        "query_cosine": [
          0.4875574826063904,
          0.2447858439702077,
          0.412924497631474,
          0.3924769848805151
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-46",
        "max_pair": 0.7270823861644252,
        "query_cosine": [
          0.5128224034912371,
          0.2266494384855283,
          0.4173014315919944,
          0.3645406266252168
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-47",
        "max_pair": 0.7352148410259243,
        "query_cosine": [
          0.5042940101402813,
          0.22594777320568096,
          0.4095920185008096,
          0.37485084971887056
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-48",
        "max_pair": 0.7595968540471119,
        "query_cosine": [
          0.5148955758657503,
          0.22883471687426737,
          0.4192731527385336,
          0.37736191508342093
        ],
        "target_argmax": true
      },
      {
        "id": "measure-pref-49",
        "max_pair": 0.7462745342014179,
        "query_cosine": [
          0.5202033873795142,
          0.22265096984913071,
          0.4099638844295005,
          0.3847765197433667
        ],
        "target_argmax": true
      }
    ],
    "max_key_cosine": 0.7808010045557472,
    "n_groups": 198,
    "participation": {
      "0.05": {
        "2": {
          "mean": 1.430695557755818,
          "min": 1.01716939758103
        },
        "4": {
          "mean": 1.7483770531151979,
          "min": 1.0309177735082935
        },
        "8": {
          "mean": 1.7483770531151979,
          "min": 1.0309177735082935
        }
      },
      "0.10": {
        "2": {
          "mean": 1.6965401116612635,
          "min": 1.1837366880648386
        },
        "4": {
          "mean": 2.512376735830501,
          "min": 1.4164543890205508
        },
        "8": {
          "mean": 2.512376735830501,
          "min": 1.4164543890205508
        }
      },
      "0.20": {
        "2": {
          "mean": 1.8859227569331642,
          "min": 1.5571669995262165
        },
        "4": {
          "mean": 3.335592605999829,
          "min": 2.5650926766747246
        },
        "8": {
          "mean": 3.335592605999829,
          "min": 2.5650926766747246
        }
      },
      "0.40": {
        "2": {
          "mean": 1.966681557931035,
          "min": 1.845941040335748
        },
        "4": {
          "mean": 3.7890865078867013,
          "min": 3.5442046826927007
        },
        "8": {
          "mean": 3.7890865078867013,
          "min": 3.5442046826927007
        }
      }
    },
    "target_argmax": 198
  }
}
""")


def bank_sha256() -> str:
    blob = json.dumps(_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def fit_items(family: str):
    return list(_PAYLOAD["fit"][family])


def distractors_for(probe_id: str):
    return list(_PAYLOAD["distractors"][probe_id])


def geometry():
    return _PAYLOAD["geometry"]
