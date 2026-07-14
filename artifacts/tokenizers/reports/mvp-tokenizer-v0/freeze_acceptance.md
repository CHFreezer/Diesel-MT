# mvp-tokenizer-v0 freeze acceptance

- Status: **frozen**
- Vocabulary: 49,152
- Artifact manifest SHA-256: `eb79ae22f523f1d9c9fcf75b80f2b322e3c2882a8fddb7545b5933dd4053fa7f`
- Tokenizer SHA-256: `22bceccab939afe1003d1fbdd95d9d4e78eb954b2e9003d70131155666b1201c`
- Training corpus manifest SHA-256: `b3d8d6f4f559813929c75086e6060b74a922a87cdb06646973d1168b5618c977`
- Holdout manifest SHA-256: `c5bec116578ea88d37f325c3e18c66a889ef34aa263bb876e821456c500f9ffe`
- Fixed evaluation manifest SHA-256: `4175b4063bc8e4a8969eca48a0d60a5da0e0d09ff55b1bce985e9898717244e6`

## Coverage

| Language | Tokens/char | P95 | P99 | Source loss | Frequency coverage | Unique coverage | Roundtrip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eng_Latn | 0.3152 | 420 | 539 | 0.000557% | 99.999443% | 99.107143% | 99.800000% |
| zho_Hans | 0.6599 | 329 | 727 | 0.007265% | 99.992735% | 99.859748% | 99.600000% |
| zho_Hant | 0.7044 | 579 | 951 | 0.043771% | 99.956229% | 98.912467% | 95.600000% |
| jpn_Jpan | 0.5325 | 350 | 689 | 0.000000% | 100.000000% | 100.000000% | 100.000000% |
| kor_Hang | 0.6351 | 411 | 778 | 0.002387% | 99.997613% | 99.787686% | 99.400000% |

## Simplified/Traditional parity

- Traditional/Simplified tokens-per-character ratio: 1.0674
- Traditional/Simplified P95 ratio: 1.7599

## Integrity and runtime checks

- Artifact files verified: 6
- Save/reload vocabulary and backend equality: True
- Five language tokens verified: True
- Micro M2M100 forwards with finite loss: 5/5
- M2M100 embedding/lm_head rows: 49,152

Synthetic rare-Unicode probes remain diagnostic and are not training data. The frozen decision uses the independently generated holdout metrics above.
