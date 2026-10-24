# Sleep pressure pre-registration

Protocol `sleep-pressure-v1`. Fit ids are `sleep-fit-000` through `sleep-fit-007`. Held ids are `sleep-held-000` through `sleep-held-049`. The banks are disjoint. Items are numeric schedules, not sentences. Bank sha256 `543e679c07fade14e2b2b78272aea0f6915e76b5cad04687dd5691f0a20c6483`.

## Constants (HYPOTHESIS)

No grid was run. Any positive wake and sleep time constants keep the monotone inequalities. The numbers below are the hypothesis, in abstract duration units. They are not the papers' hour-scale EEG constants, and they were not retuned on held.

| name | value |
|---|---|
| floor | 0.1 |
| tau_wake | 16 |
| tau_sleep | 4 |
| prediction-error threshold | 0.25 |
| blend | 0.1 |
| strength step | 0.05 |
| spontaneous period | 16 |
| spontaneous S minimum | 0.2 |
| spontaneous ceiling | 0.10, strict |

Wake: `S = 1 - (1 - S) * exp(-t / tau_wake)`. Sleep: `S = floor + (S - floor) * exp(-D / tau_sleep)`. Sleep never assigns 0. Prediction error is `1 - cosine(key, value)`. A slot is rewritten only when that error is above the threshold. Strength becomes `strength * exp(-step * min(error, 1))`. The embedding becomes `(1 - blend) * v + blend * unit(key) * ||v||`.

## Citations

Verified on 2026-10-24 through the Crossref works API.

- Daan, Beersma, and Borbély (1984), American Journal of Physiology 246:R161–R183. Process S increases during wakefulness and decreases during sleep. https://doi.org/10.1152/ajpregu.1984.246.2.R161
- Borbély, Daan, Wirz-Justice, and Deboer (2016), Journal of Sleep Research 25:131–143. Homeostatic Process S interacts with circadian Process C. https://doi.org/10.1111/jsr.12371
- Borbély (2022), Journal of Sleep Research 31. Review of the model's origins. https://doi.org/10.1111/jsr.13598
- Sevenster, Beckers, and Kindt (2013), Science 339:830–833. Prediction error is required before a reactivated human fear memory is updated. https://doi.org/10.1126/science.1231357
- Berntsen (2020), Philosophical Transactions of the Royal Society B 376:20190693. Memories that come to mind with no conscious initiation of retrieval. https://doi.org/10.1098/rstb.2019.0693

Borbély (1982), Human Neurobiology, has no Crossref DOI. It is not cited. A DOI sometimes listed for that paper does not resolve and is not cited.

Berntsen (2020) says involuntary autobiographical memory cannot be equated with mind-wandering. This preregistration does not cite mind-wandering as the mechanism.

growbrain's planned consumer is `θ = θ0 + a·S − b·C`. tidal does not compute that expression. Process C stays the exported circadian phase. There is no growbrain switch.

## Gates

Locked from the fit bank, before held:

1. Wake durations 1, 2, and 4, each from the floor, are strictly increasing, and each pressure is in `(floor, 1]`.
2. From the same post-wake pressure (wake duration 4) on an empty store, sleep duration 3 drops more than sleep duration 1. Both drops are positive. Both pressures stay above the floor and above 0.
3. Replay of one parallel slot does not change strength or the embedding. Replay of one orthogonal slot lowers strength and changes the embedding. `text` and `answer` stay. The audit keys are only `index`, `error`, `updated`, `strength_before`, and `strength_after`. The raw strings are absent from the audit. `cue_fields` does not contain the audit.
4. After wake duration 4, 16 no-cue ticks fire spontaneous on exactly 1 tick. The rate is 0.0625, above 0 and strictly below 0.10. Process S is unchanged by the ticks.
5. Cue export `source` is only `cue` or `none`. `sleep_pressure` matches the scalar within float32 tolerance. `interface_version` is 4.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. The cue guard writes new files and must match those counts and means. The cue guard was not scored.

The held bank was not scored.

## Fit result

Report `reports/subconscious_sleep_fit.json`. Every gate is 8/8, Wilson 95% [0.6755924351161198, 1.0]. Wake pressures are 0.15452824346787175, 0.20575278767386407, and 0.2990792952357356. Sleep from 0.2990792952357356 drops by 0.04403618421284128 and 0.10504089482312046, to 0.25504311102289434 and 0.19403840041261516. Spontaneous fires are 1/16, rate 0.0625. Orthogonal replay strength is 0.951229424500714. `canonical_same` is true. These counts did not move a held floor. Theta, half-life, lambda, fan S, and the RIF span were not changed.
