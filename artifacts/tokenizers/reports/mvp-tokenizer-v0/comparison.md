# Tokenizer candidate comparison

> Sampled-corpus comparison. At least one candidate used sample_fraction < 1. Compare candidates only when their training provenance matches; this report does not claim full-corpus coverage.

| Candidate | Vocab | Train fraction | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage | Vocab used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mvp-48k | 49152 | 1.0 | 0.5355 | 0.017307% | 0.009443% | 99.990557% | 99.519843% | 56.597900% |
| fallback-32k | 32768 | 0.5 | 0.6376 | 0.110800% | 0.072394% | 99.927606% | 98.187163% | 45.114136% |

## Per-language comparison

| Language | Candidate | Mean tokens | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | mvp-48k | 113.0780 | 0.3152 | 0.001769% | 0.000557% | 99.999443% | 99.107143% |
| eng_Latn | fallback-32k | 132.4380 | 0.3691 | 0.000000% | 0.000000% | 100.000000% | 100.000000% |
| zho_Hans | mvp-48k | 108.9960 | 0.6599 | 0.011010% | 0.007265% | 99.992735% | 99.859748% |
| zho_Hans | fallback-32k | 124.1860 | 0.7519 | 0.004831% | 0.003633% | 99.996367% | 99.929874% |
| zho_Hant | mvp-48k | 141.6100 | 0.7044 | 0.060730% | 0.043771% | 99.956229% | 98.912467% |
| zho_Hant | fallback-32k | 187.5240 | 0.9327 | 0.427679% | 0.408862% | 99.591138% | 95.145889% |
| jpn_Jpan | mvp-48k | 89.1380 | 0.5325 | 0.000000% | 0.000000% | 100.000000% | 100.000000% |
| jpn_Jpan | fallback-32k | 102.6940 | 0.6135 | 0.000000% | 0.000000% | 100.000000% | 100.000000% |
| kor_Hang | mvp-48k | 159.6340 | 0.6351 | 0.003759% | 0.002387% | 99.997613% | 99.787686% |
| kor_Hang | fallback-32k | 182.3980 | 0.7257 | 0.000000% | 0.000000% | 100.000000% | 100.000000% |
