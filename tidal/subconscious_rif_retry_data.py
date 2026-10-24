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
    VALENCE_FIT_SHA,
    VALENCE_HELD_SHA,
    CIRCADIAN_FIT_SHA,
    CIRCADIAN_HELD_SHA,
)
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
