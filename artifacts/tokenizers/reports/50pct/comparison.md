# Tokenizer candidate comparison

> Sampled-corpus comparison. At least one candidate used sample_fraction < 1. Compare candidates only when their training provenance matches; this report does not claim full-corpus coverage.

| Candidate | Vocab | Train fraction | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage | Vocab used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32k | 32768 | 0.5 | 0.5820 | 0.001948% | 0.001275% | 99.998725% | 99.879032% | 43.826294% |
| 48k | 49152 | 0.5 | 0.4953 | 0.002289% | 0.001275% | 99.998725% | 99.879032% | 57.584635% |

## Per-language comparison

| Language | Candidate | Mean tokens | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 32k | 197.8340 | 0.3757 | 0.003033% | 0.001139% | 99.998861% | 97.959184% |
| eng_Latn | 48k | 165.7820 | 0.3148 | 0.003619% | 0.001139% | 99.998861% | 97.959184% |
| zho_Hans | 32k | 230.1460 | 0.7531 | 0.003476% | 0.002618% | 99.997382% | 99.889807% |
| zho_Hans | 48k | 201.7620 | 0.6602 | 0.003965% | 0.002618% | 99.997382% | 99.889807% |
| jpn_Jpan | 32k | 140.7600 | 0.6070 | 0.001421% | 0.001725% | 99.998275% | 99.910314% |
| jpn_Jpan | 48k | 118.7840 | 0.5122 | 0.001684% | 0.001725% | 99.998275% | 99.910314% |
| kor_Hang | 32k | 252.7240 | 0.7276 | 0.000000% | 0.000000% | 100.000000% | 100.000000% |
| kor_Hang | 48k | 212.7980 | 0.6126 | 0.000000% | 0.000000% | 100.000000% | 100.000000% |
