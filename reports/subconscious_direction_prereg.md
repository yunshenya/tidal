# Direction preregistration

Polarity for prefer versus avoid. The frozen bank and the held-out bank have not been scored.

No feature flag. `gap_tau` stays 0.80. Phone and birthday spelling still run only when the winning relation is the one the prompt is completing. A continuation still encodes the query once.

## Banks

- train `626ec7cb9ac7824666be6a94ab8e6dc428b7e092ccfc7d8db689fde9c92be4d9`
- fit `ee75f1857bfb6b7e812dfb0ef4012af1dc6efb33ca17799f1211b570ad1c0c1d`
- held `e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b`
- frozen `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`
- #39 fit `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce`
- #39 held `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a`
- #40 held `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464`
- literal overlap 0 and token Jaccard under 0.50 against those banks and across train, fit, and held

## Architecture

The add is `c * gain * (scale(v) + MLP(scale(v)))`.
`gain = 1 + 2 tanh(w · normalize(k))`.
`w = 0` leaves `gain = 1`, which is the #40 residual.
`gap_tau` is 0.80 and was not retuned.
Polarity parameters: 576. MLP parameters: 664704, frozen at the #38 checkpoint.
The no-key ablation trains the MLP and does not see the fact key. Polarity training updates only `w`.

## Fit selection

Early stop uses the fit bank only. The frozen bank and the held-out bank have not been scored.

No-key ablation, best avoid mean at step 250: prefer 2.893 (5/8), avoid 0.278 (4/8), forbidden down 6/8, NLL -0.015 (0 > 0.5), flips 0, fit_pass False.

Polarity, selected step 350: prefer 2.185 (6/8), avoid 0.667 (6/8), forbidden down 7/8 (mean move -1.303), NLL -0.017 (0 > 0.5), flips 0, gain prefer 0.111, avoid -0.259, incidental 0.935.

The key term is justified because the no-key ablation does not pass these fit gates and polarity does, with the MLP weights unchanged.

## Gates

Held-out is merge-critical. Frozen `e3cdd7fd…` is re-scored as a measurement under the same numbers.

- prefer margin delta > 0 on at least 6/8
- avoid margin delta > 0 on at least 6/8
- forbidden logprob down on at least 6/8 avoid items
- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5
- choice flips 0
- leak 0
- phone and birthday takeover 0 off-relation
- phone question takeover 1, and birthday question takeover 1 when the bank has one

Load top-1, FIFO, and prefer choice lift are reported, not gated.
