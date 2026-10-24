"""Fit and held-out banks for retrieval-induced forgetting.

No torch and no train split. The fit bank selects the span. The held-out
bank is scored once, after the preregistration file is committed. Literal
overlap with every earlier subconscious bank, including the closed #43 banks
and the relative-surprise banks, is 0 and the maximum token Jaccard is under
0.50. Held-out logprob margins are not in this file.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL_FIT = "subconscious-rif-retry-fit-v1"
PROTOCOL_HELD = "subconscious-rif-retry-v1"

RIF_GATES = {
    "competitor_drop_min": 0.10,
    "competitor_mean_drop_min": 0.20,
    "winner_abs_drop_max": 0.05,
    "noncompetitor_abs_drop_max": 0.05,
    "prefer_delta_pos_min": 8,
    "prefer_mean_min": 8.0,
    "avoid_delta_pos_min": 8,
    "avoid_mean_min": 0.6,
    "forbidden_down_min": 6,
}

DIRECTION_HELD_PREFER_MEAN = 10.905
DIRECTION_HELD_AVOID_MEAN = 1.277
DIRECTION_HELD_FORBIDDEN_LIFT = -1.3
REPLAY_HELD_PREFER_MEAN = 20.75
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
SURPRISE_REL_TRAIN_SHA = "ff97715ddd8c693bf35983ca5d527a0e1ba3f107e31d7ea438926cb4bf3a6005"
SURPRISE_REL_FIT_SHA = "87a581a35f0d8ed51e8c394c68d241fc65da9fefbf119c4a1fc99b6fcb5b641d"
SURPRISE_REL_HELD_SHA = "c68221e3a8dc696628399f2f7c605bbde8a26f314e77f822b1a2b2e90f35686c"

# The gates above are this logprob-margin expression of the human effect.
# Recall percentages are from Anderson, Bjork, and Bjork (1994), Experiment 1,
# Table 1 and the category-cued recall tests reported with it. Do not retune
# the floors to chase those percentages: the units are different.
HUMAN_EFFECT = """\
The goal is a human-like memory, including fuzzy recall and ordinary errors, rather than a store that is as useful as possible. Retrieving a target should make the other items on that same cue a little harder to recall, and should leave them in memory. The floors already fixed below are that prediction. They are not raised after the held-out bank is scored.

Anderson, Bjork, and Bjork (1994) showed that practicing retrieval of some category members impairs later recall of the unpracticed members of those same categories. In their retrieval-practice paradigm the practiced items are Rp+, the unpracticed items that share the practiced category cue are Rp-, and items from categories that were not practiced are Nrp. Experiment 1, category-cued recall after a 20-minute delay: Rp+ 73.6%, Rp- 37.5%, Nrp 48.4%. The Rp- impairment against that Nrp baseline was reliable, F(1, 32) = 30.3. Strong exemplars were impaired by 15.7 points (Rp- 40.3% versus Nrp 56.0%). Weak exemplars were impaired by 6.3 points (Rp- 34.7% versus Nrp 41.0%). The practiced items were facilitated, not weakened. The loss is partial: the Rp- items are still recalled, only less often.

Anderson, M. C. (2003) accounts for that pattern as inhibition recruited to override competitors during selective retrieval, rather than as passive blocking caused by strengthening the practiced item. The properties reviewed there are retrieval specificity, interference dependence (strong competitors are impaired more; weak ones often are not), and strength independence (the impairment does not grow just because the practiced item was facilitated more). Cue-independent suppression is part of that account. This change does not implement an independent-cue test, and it does not claim prefrontal control.

Norman, K. A., Newman, E. L., and Detre, G. (2007) give a network account in which oscillating feedback inhibition weakens competitors that threaten the target and strengthens weak parts of the target. The consequence taken from that model is competitor punishment after the target has been selected. The mechanism here is not their oscillating k-winner-take-all rule, not contrastive Hebbian learning, and not a cortical-hippocampal network. It multiplies the strength of a small same-relation set. A competitor whose key is closer to the winner is scaled down more, which is the local stand-in for interference dependence. The winner itself is not scaled.

How the existing floors express that effect, in logprob-margin nats rather than percent recall:
- A competitor is an Rp- item. It shares the winner's relation and it is not the retrieved target. After one target hit it must sit below its own strength-1 control, which is the Nrp baseline for that item. Minimum drop at least 0.10. Mean drop at least 0.20. At span 0.05 the keep is between 0.95 and 0.975, so the item is impaired and not erased.
- The target is the Rp+ item. Human Rp+ recall rose from the Nrp baseline to 73.6%. This store does not add that facilitation. The gate is the part this mechanism can get wrong: the target is not weakened. Absolute winner drop at most 0.05.
- An outsider is an Nrp item. Its relation was not the practiced cue, so retrieval of the target should leave it alone. Absolute drop at most 0.05.
The human figures are category-cued recall percentages. These gates are not a conversion of those percentages into nats. The direction is Rp- below Nrp, Rp+ not impaired, Nrp unaffected, and the size is a partial drop of a few tenths of a nat.

Anderson, M. C., Bjork, R. A., & Bjork, E. L. (1994). Remembering can cause forgetting: Retrieval dynamics in long-term memory. Journal of Experimental Psychology: Learning, Memory, and Cognition, 20(5), 1063-1087. https://doi.org/10.1037/0278-7393.20.5.1063

Anderson, M. C. (2003). Rethinking interference theory: Executive control and the mechanisms of forgetting. Journal of Memory and Language, 49(4), 415-445. https://doi.org/10.1016/j.jml.2003.08.006

Norman, K. A., Newman, E. L., & Detre, G. (2007). A neural network model of retrieval-induced forgetting. Psychological Review, 114(4), 887-953. https://doi.org/10.1037/0033-295X.114.4.887
"""

_PAYLOAD = json.loads(r"""
{
  "fit": [
    [
      {
        "id": "rif-retry-fit-q0-0",
        "role": "target",
        "name": "Xokohipe",
        "rel": "zqfit0",
        "text": "Xokohipe participates congestion wholes dehuman committing accountant tapped.",
        "answer": "tapped",
        "prompt": "Xokohipe participates congestion wholes dehuman committing accountant",
        "good": " tapped",
        "bad": " imprisonment"
      },
      {
        "id": "rif-retry-fit-q0-1",
        "role": "competitor",
        "name": "Gifudili",
        "rel": "zqfit0",
        "text": "Gifudili recogn physical simulating nationwide understands syntax betting.",
        "answer": "betting",
        "prompt": "Gifudili recogn physical simulating nationwide understands syntax",
        "good": " betting",
        "bad": " carcinogenic"
      },
      {
        "id": "rif-retry-fit-q0-2",
        "role": "competitor",
        "name": "Woximiho",
        "rel": "zqfit0",
        "text": "Woximiho catering genuine histories philosopher particle migrants liberty.",
        "answer": "liberty",
        "prompt": "Woximiho catering genuine histories philosopher particle migrants",
        "good": " liberty",
        "bad": " respond"
      },
      {
        "id": "rif-retry-fit-q0-3",
        "role": "competitor",
        "name": "Diteturo",
        "rel": "zqfit0",
        "text": "Diteturo eliminates regener signing reinforced craters credit bravery.",
        "answer": "bravery",
        "prompt": "Diteturo eliminates regener signing reinforced craters credit",
        "good": " bravery",
        "bad": " tricks"
      },
      {
        "id": "rif-retry-fit-q0-4",
        "role": "outsider",
        "name": "Zicibohu",
        "rel": "zqfit0x",
        "text": "Zicibohu teeming remedies thinning enlarg ounces imposing patterned.",
        "answer": "patterned",
        "prompt": "Zicibohu teeming remedies thinning enlarg ounces imposing",
        "good": " patterned",
        "bad": " outstanding"
      }
    ],
    [
      {
        "id": "rif-retry-fit-q1-0",
        "role": "target",
        "name": "Jocuhoco",
        "rel": "zqfit1",
        "text": "Jocuhoco penned fluores timest wedding improve graduated foliage.",
        "answer": "foliage",
        "prompt": "Jocuhoco penned fluores timest wedding improve graduated",
        "good": " foliage",
        "bad": " replacement"
      },
      {
        "id": "rif-retry-fit-q1-1",
        "role": "competitor",
        "name": "Rosufeci",
        "rel": "zqfit1",
        "text": "Rosufeci govern consist alternate homeopathic airways populate gypsum.",
        "answer": "gypsum",
        "prompt": "Rosufeci govern consist alternate homeopathic airways populate",
        "good": " gypsum",
        "bad": " refractory"
      },
      {
        "id": "rif-retry-fit-q1-2",
        "role": "competitor",
        "name": "Senodava",
        "rel": "zqfit1",
        "text": "Senodava beneficiary fibromyalgia pairing announces spouses desires withstand.",
        "answer": "withstand",
        "prompt": "Senodava beneficiary fibromyalgia pairing announces spouses desires",
        "good": " withstand",
        "bad": " assassinated"
      },
      {
        "id": "rif-retry-fit-q1-3",
        "role": "competitor",
        "name": "Wigajivu",
        "rel": "zqfit1",
        "text": "Wigajivu sprinkled invariably unsupervised wearing demise director information.",
        "answer": "information",
        "prompt": "Wigajivu sprinkled invariably unsupervised wearing demise director",
        "good": " information",
        "bad": " amphibians"
      },
      {
        "id": "rif-retry-fit-q1-4",
        "role": "outsider",
        "name": "Yuwepipu",
        "rel": "zqfit1x",
        "text": "Yuwepipu registers discriminate lipids sesame closer arthrop spawning.",
        "answer": "spawning",
        "prompt": "Yuwepipu registers discriminate lipids sesame closer arthrop",
        "good": " spawning",
        "bad": " bruising"
      }
    ]
  ],
  "held": [
    [
      {
        "id": "rif-retry-held-q0-0",
        "role": "target",
        "name": "Yuzezobo",
        "rel": "zqheld0",
        "text": "Yuzezobo toughness prepare workout momentarily unfold inscriptions carnivores.",
        "answer": "carnivores",
        "prompt": "Yuzezobo toughness prepare workout momentarily unfold inscriptions",
        "good": " carnivores",
        "bad": " participant"
      },
      {
        "id": "rif-retry-held-q0-1",
        "role": "competitor",
        "name": "Cegelugo",
        "rel": "zqheld0",
        "text": "Cegelugo certificates correlate aesthetic formation altern valued toolkit.",
        "answer": "toolkit",
        "prompt": "Cegelugo certificates correlate aesthetic formation altern valued",
        "good": " toolkit",
        "bad": " monsters"
      },
      {
        "id": "rif-retry-held-q0-2",
        "role": "competitor",
        "name": "Lebehimu",
        "rel": "zqheld0",
        "text": "Lebehimu nomination strengths violent unhappy widening broadband numeracy.",
        "answer": "numeracy",
        "prompt": "Lebehimu nomination strengths violent unhappy widening broadband",
        "good": " numeracy",
        "bad": " responding"
      },
      {
        "id": "rif-retry-held-q0-3",
        "role": "competitor",
        "name": "Toxayoce",
        "rel": "zqheld0",
        "text": "Toxayoce feasts triumph eagerly misplaced cousins readings appeals.",
        "answer": "appeals",
        "prompt": "Toxayoce feasts triumph eagerly misplaced cousins readings",
        "good": " appeals",
        "bad": " simmer"
      },
      {
        "id": "rif-retry-held-q0-4",
        "role": "outsider",
        "name": "Sabeyege",
        "rel": "zqheld0x",
        "text": "Sabeyege spectacular necessities introduction unification agricult handler muscul.",
        "answer": "muscul",
        "prompt": "Sabeyege spectacular necessities introduction unification agricult handler",
        "good": " muscul",
        "bad": " experiment"
      }
    ],
    [
      {
        "id": "rif-retry-held-q1-0",
        "role": "target",
        "name": "Defodehu",
        "rel": "zqheld1",
        "text": "Defodehu enough elephant wondering finance calories equality sentence.",
        "answer": "sentence",
        "prompt": "Defodehu enough elephant wondering finance calories equality",
        "good": " sentence",
        "bad": " multipl"
      },
      {
        "id": "rif-retry-held-q1-1",
        "role": "competitor",
        "name": "Nuyaleji",
        "rel": "zqheld1",
        "text": "Nuyaleji circulatory estimating speculation bioenergy volunteering enthusi snacking.",
        "answer": "snacking",
        "prompt": "Nuyaleji circulatory estimating speculation bioenergy volunteering enthusi",
        "good": " snacking",
        "bad": " analog"
      },
      {
        "id": "rif-retry-held-q1-2",
        "role": "competitor",
        "name": "Zugucoja",
        "rel": "zqheld1",
        "text": "Zugucoja oppressive specialised uttered skyrock ferment sauces dosing.",
        "answer": "dosing",
        "prompt": "Zugucoja oppressive specialised uttered skyrock ferment sauces",
        "good": " dosing",
        "bad": " renewal"
      },
      {
        "id": "rif-retry-held-q1-3",
        "role": "competitor",
        "name": "Busisoda",
        "rel": "zqheld1",
        "text": "Busisoda pondered lubricant hailed fishes imitate multim carriage.",
        "answer": "carriage",
        "prompt": "Busisoda pondered lubricant hailed fishes imitate multim",
        "good": " carriage",
        "bad": " positives"
      },
      {
        "id": "rif-retry-held-q1-4",
        "role": "outsider",
        "name": "Dusegejo",
        "rel": "zqheld1x",
        "text": "Dusegejo affect novice delect administ scholarship defaults critiques.",
        "answer": "critiques",
        "prompt": "Dusegejo affect novice delect administ scholarship defaults",
        "good": " critiques",
        "bad": " diabetics"
      }
    ]
  ]
}
""")


def bank_sha256(kind: str) -> str:
    blob = json.dumps(_PAYLOAD[kind], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


class RifRetryBank:
    def __init__(self, kind: str):
        raw = _PAYLOAD[kind]
        self.kind = kind
        self.PROTOCOL = PROTOCOL_FIT if kind == "fit" else PROTOCOL_HELD
        self.QUARTETS = tuple(tuple(group) for group in raw)

    def sha256(self):
        return bank_sha256(self.kind)


def fit_bank():
    return RifRetryBank("fit")


def held_bank():
    return RifRetryBank("held")
