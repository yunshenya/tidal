# Subconscious exact phone and birthday spelling

Phone and birthday spelling copies the winning slot's stored answer, the full number or the full date. It does not spell a trained prefix and it does not spell a digit readout. It runs only when the prompt is completing that relation. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. The add is `c * s * gain * (scale(v) + MLP(scale(v)))`, `gain = 1 + 2 tanh(w · normalize(k))`. Replay refreshes the top 1 slot every 32 accepted writes. Decay is 0.5. One query per continuation. No feature flag.

Fit bank `9b677681fe0b626a8a7e411bf030e53b7f167e72dcee8b4de2d0393ab4822807`. Held-out bank `fd01592f5bdb8e919c113fb0f82c01a26cc42c24c14104fb77ba5bc3bf65f635`. The rule was locked on the fit bank before this held-out bank was scored. Held-out was scored once.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| fit | phone off-relation takeover | yes | 0.000 |
| fit | birthday off-relation takeover | yes | 0.000 |
| fit | phone exact | yes | '+1-449-555-3249' vs '+1-449-555-3249' |
| fit | birthday exact | yes | 'February 21, 2018' vs 'February 21, 2018' |
| fit | phone off-relation not stored | yes | 0 |
| fit | birthday off-relation not stored | yes | 0 |
| fit | neutral NLL | yes | mean -0.017, above 0.5: 0 |
| fit | choice flips | yes | 0/8 |
| fit | leak | yes | 0/8 |
| fit | incidental slots | yes | 32 slots, max cosine 0.839 |
| fit | new parameters | yes | 0 |
| fit | remember forwards | yes | 2 |
| fit | query forwards | yes | 1 |
| held | phone off-relation takeover | yes | 0.000 |
| held | birthday off-relation takeover | yes | 0.000 |
| held | phone exact | yes | '+1-802-555-3145' vs '+1-802-555-3145' |
| held | birthday exact | yes | 'July 26, 1990' vs 'July 26, 1990' |
| held | phone off-relation not stored | yes | 0 |
| held | birthday off-relation not stored | yes | 0 |
| held | neutral NLL | yes | mean -0.013, above 0.5: 0 |
| held | choice flips | no | 1/8 |
| held | leak | yes | 0/8 |
| held | incidental slots | yes | 32 slots, max cosine 0.880 |
| held | new parameters | yes | 0 |
| held | remember forwards | yes | 2 |
| held | query forwards | yes | 1 |
| held | prefer mean | yes | 20.750 >= 8.0 |
| held | avoid mean | yes | 1.453 >= 0.6 |
| held | forbidden logprob down | yes | 8/8 down, mean stored lift -1.578 |

## Stored strings

- phone question `+1-802-555-3145`
- stored phone `+1-802-555-3145`
- exact 1, prefix detector 0
- birthday question `July 26, 1990`
- stored birthday `July 26, 1990`
- exact 1, prefix detector 1

## Prefer and avoid against #44 held

- prefer mean 20.750 (8/8 > 0), delta -0.000 against 20.750
- avoid mean 1.453 (8/8 > 0), delta -0.000 against 1.453
- forbidden mean stored lift -1.578 (8/8 down), delta -0.000 against -1.578

## Prefer and avoid against #41 held

- prefer delta 9.845 against 10.905
- avoid delta 0.176 against 1.277
- forbidden delta -0.278 against -1.300

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one remember: 2
- forwards inside one armed phone question: 1
- The phone question returns the stored string. There is no second query and no feature flag.

Polarity span is 2. Held-out pass: no.
