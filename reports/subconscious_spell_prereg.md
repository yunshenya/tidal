# Exact phone and birthday spelling preregistration

The held-out bank has not been scored.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Replay stays every 32 accepted writes, m = 1, decay 0.5. Readout stays hard top-1. A continuation still encodes the query once.

Phone and birthday spelling copies the winning slot's stored answer, the full number or the full date. It does not spell a trained prefix such as `+1-415-555-` and it does not spell a digit readout. It runs only when the prompt is completing that relation.

## Banks

- fit `9b677681fe0b626a8a7e411bf030e53b7f167e72dcee8b4de2d0393ab4822807`
- held `fd01592f5bdb8e919c113fb0f82c01a26cc42c24c14104fb77ba5bc3bf65f635`
- frozen `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`
- #39 fit `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce`
- #39 held `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a`
- #40 held `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464`
- #41 train `626ec7cb9ac7824666be6a94ab8e6dc428b7e092ccfc7d8db689fde9c92be4d9`
- #41 fit `ee75f1857bfb6b7e812dfb0ef4012af1dc6efb33ca17799f1211b570ad1c0c1d`
- #41 held `e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b`
- #42 fit `4a90e639af898812b9edd13f68cf6c1edebbadaf36c761d0b2d03fdddc30bec0`
- #42 held `41ca5f243131f13deefe85800a775e8378bdeb233954a0b51119d66d93d180f1`
- #43 fit `2ad8038eefbb5bd77f7d8d3d9e52b30f2c1abf4467713c35b8eacad5bb72fe44`
- #43 held `ef0d06d320c3e8e15f1ac78bbe3e486bdd3017ca37f61b193089303fe5e3d6e9`
- #44 fit `35ae15cde3079cee5b034e2e019ef4a76324574507a8dbb58831bdb141b9f552`
- #44 held `e684c422049c720b269a50d762ca8ec1be87b397c93068ea2d61fa20ad4127ce`
- valence fit `72a1836feafaf6df89347b87f527cdc80cbe64d3709746c5ca0e57c89ad814c8`
- valence held `a4b5ce1da2ce61070720487e7e3e5e9d5bc8d78a9b5a1281ea69c9e2223fcbd5`
- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held

## Rule

- on-relation phone: generated text equals the stored number
- on-relation birthday: generated text equals the stored date
- prefix takeover is not the on-relation gate
- off-relation phone takeover: 0
- off-relation birthday takeover: 0
- off-relation text is not the stored string
- new parameters: 0
- forwards inside one remember: 2
- forwards inside one armed phone question: 1
- gap_tau: 0.80
- polarity span: 2
- replay every: 32
- replay m: 1
- replay decay: 0.5

## Fit

- phone exact 1: +1-449-555-3249
- stored phone +1-449-555-3249
- phone prefix detector 0
- birthday exact 1: February 21, 2018
- stored birthday February 21, 2018
- birthday prefix detector 1
- phone off-relation takeover 0.000
- birthday off-relation takeover 0.000
- phone off-relation equal to stored 0
- birthday off-relation equal to stored 0
- neutral NLL mean -0.017, items above 0.5: 0
- choice flips 0/8
- leak 0/8
- incidental slots 32

## Held-out gates

The same spelling gates as the fit bank. In addition, rescore the #44 held-out prefer/avoid items and report them against that run:

- prefer mean at least 8.0, #44 held 20.750, #41 held 10.905
- avoid mean at least 0.6, #44 held 1.453, #41 held 1.277
- forbidden down at least 6 of 8, #44 held lift -1.578, #41 held lift -1.300

One held-out pass. No retune.
