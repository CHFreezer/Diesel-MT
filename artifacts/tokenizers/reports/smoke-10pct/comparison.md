# Tokenizer candidate comparison

> Preliminary smoke comparison. At least one candidate was trained on a corpus sample; final TD-07 selection requires full-corpus artifacts.

| Candidate | Vocab | Train fraction | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage | Vocab used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32k | 32768 | 0.1 | 0.5821 | 0.002921% | 0.001700% | 99.998300% | 99.852151% | 43.838501% |
| 48k | 49152 | 0.1 | 0.4955 | 0.003718% | 0.001842% | 99.998158% | 99.838710% | 57.552083% |

## Per-language comparison

| Language | Candidate | Mean tokens | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 32k | 197.8400 | 0.3757 | 0.003033% | 0.001139% | 99.998861% | 97.959184% |
| eng_Latn | 48k | 165.9000 | 0.3150 | 0.003617% | 0.001139% | 99.998861% | 97.959184% |
| zho_Hans | 32k | 230.1980 | 0.7532 | 0.005213% | 0.003927% | 99.996073% | 99.834711% |
| zho_Hans | 48k | 201.7940 | 0.6603 | 0.005947% | 0.003927% | 99.996073% | 99.834711% |
| jpn_Jpan | 32k | 140.6580 | 0.6066 | 0.001422% | 0.000862% | 99.999138% | 99.955157% |
| jpn_Jpan | 48k | 118.9240 | 0.5128 | 0.003363% | 0.001725% | 99.998275% | 99.910314% |
| kor_Hang | 32k | 252.8660 | 0.7280 | 0.001582% | 0.001152% | 99.998848% | 99.930216% |
| kor_Hang | 48k | 212.7420 | 0.6125 | 0.001880% | 0.001152% | 99.998848% | 99.930216% |
