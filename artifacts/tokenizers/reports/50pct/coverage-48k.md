# Tokenizer coverage report: 48k

> Sampled-corpus result: this tokenizer was trained with sample_fraction=0.5. Compare it only with candidates that use the same training provenance; this report does not claim full-corpus coverage.

## Provenance

- Artifact: `artifacts/tokenizers/50pct/mvp-48k`
- Vocabulary size: 49,152
- Training sample fraction: 0.5
- Evaluation seed: 20260713
- Corpus samples per language: 500
- Evaluation corpus manifest SHA-256: `11daef6a8b38c7dc66dc17d51ab4ab2ab9053a0816340261ad76e53740be0477`

## Corpus metrics

| Language | Samples | Mean tokens | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage | Exact roundtrip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 500 | 165.7820 | 0.3148 | 0.003619% | 0.001139% | 99.998861% | 97.959184% | 99.600000% |
| zho_Hans | 500 | 201.7620 | 0.6602 | 0.003965% | 0.002618% | 99.997382% | 99.889807% | 99.200000% |
| jpn_Jpan | 500 | 118.7840 | 0.5122 | 0.001684% | 0.001725% | 99.998275% | 99.910314% | 99.800000% |
| kor_Hang | 500 | 212.7980 | 0.6126 | 0.000000% | 0.000000% | 100.000000% | 100.000000% | 100.000000% |

## Token length distribution

| Language | P50 | P95 | P99 | Max | >500-char samples | Long-sample UNK |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 79 | 579 | 1401 | 3679 | 152 | 0.000000% |
| zho_Hans | 150 | 521 | 1111 | 3713 | 72 | 0.002299% |
| jpn_Jpan | 43 | 414 | 1299 | 3385 | 60 | 0.000000% |
| kor_Hang | 169 | 532 | 1316 | 1742 | 215 | 0.000000% |

## Synthetic stress probes

| Language | Category | Samples | Tokens | UNK tokens | Source loss | Exact roundtrip |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | daily | 1 | 17 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | mixed_language | 1 | 22 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | numeric_punctuation | 2 | 54 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | rare_unicode | 1 | 18 | 27.777778% | 37.500000% | 0.000000% |
| eng_Latn | technical_news | 1 | 23 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | very_short | 1 | 2 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | daily | 1 | 17 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | mixed_language | 1 | 23 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | numeric_punctuation | 2 | 48 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | rare_unicode | 1 | 14 | 14.285714% | 33.333333% | 0.000000% |
| zho_Hans | technical_news | 1 | 22 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | very_short | 1 | 1 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | daily | 1 | 17 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | mixed_language | 1 | 22 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | numeric_punctuation | 2 | 49 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | rare_unicode | 1 | 13 | 15.384615% | 27.272727% | 0.000000% |
| jpn_Jpan | technical_news | 1 | 19 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | very_short | 1 | 2 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | daily | 1 | 14 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | mixed_language | 1 | 23 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | numeric_punctuation | 2 | 52 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | rare_unicode | 1 | 18 | 27.777778% | 40.000000% | 0.000000% |
| kor_Hang | technical_news | 1 | 20 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | very_short | 1 | 1 | 0.000000% | 0.000000% | 100.000000% |

## Script-specific analysis

- Shared Chinese/Japanese Han: 1,100 unique, 100.000000% covered, 0 standalone source-language split mismatches.
- Korean Hangul: 1,218 unique syllables/Jamo, 100.000000% frequency-weighted coverage and 100.000000% unique coverage.
- English subwords: 1.4602 pieces/word, 72.246538% one-piece words, 1.228163% over four pieces.
- Evaluation vocabulary utilization: 28,304/49,152 (57.584635%).

## Highest-frequency uncovered characters

### eng_Latn

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| ใ | U+0E43 | Other letter | 1 | e innocent party. He is like the หมาเห่าใบตองแห้ง or the dog who could dry banana |
| ﴾ | U+FD3E | Punctuation | 1 | ﴾"O my father! I did see eleven stars and |
| ﴿ | U+FD3F | Punctuation | 1 |  I saw them prostrate themselves to me!”﴿ |

### zho_Hans

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| 慤 | U+6164 | Han | 1 | 届人大常委会第十四次会议通过，鍏ラ棬鍗囩骇鏈€瓒呭€硷紒600鍏冪骇鏈€寮洪珮棰慤灏辨槸瀹。宜都市机械行业工会联合会成立暨第一次会员代表大会召开，新版新闻记者证特 |
| 沋 | U+6C8B | Han | 1 | 治区十三届人大三次会议主席团举行第四次会议，中国战略定力在悄然积累积极因素。渭南沋河水兵舞队：舞出健康快乐 舞出精彩人生，海南房地产新闻海南别墅新闻海南楼盘新闻海 |
| 溍 | U+6E8D | Han | 1 | 人身边新锐丨雷米，写到鬼上身就对了人物丨许士中，用弘一书法，悟弘一佛法品书丨朱家溍的猫与王世襄的冬瓜典藏丨传奇之外的齐白石艺界.Artist大咖丨杨立新，舞台40 |
| 錽 | U+933D | Han | 1 | 拼搏pinbo体育\|官网，研讨贴，錽金银器具如何辨别新老，拿什么拯救你，心脏骤停，猛降到29万，特斯拉的屠杀开始了  |

### jpn_Jpan

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| ᗨ | U+15E8 | Other letter | 1 | 未経験の子が一番稼いでくれてます.(❁′ᗨ‵❁)求人で応募してくれた方は※正直お給料は上げやすいです!!（スカウトを通すと |
| ‵ | U+2035 | Punctuation | 1 | 未経験の子が一番稼いでくれてます.(❁′ᗨ‵❁)求人で応募してくれた方は※正直お給料は上げやすいです!!（スカウトを通すと、 |

### kor_Hang

No uncovered characters in the fixed corpus sample.

## Metric definitions

- Main metrics use only the fixed corpus sample; synthetic probes are reported separately.
- Source-character loss counts non-whitespace Unicode code points covered by `<unk>` offsets. A fused unknown span may count as several lost characters.
- Frequency-weighted and unique-character coverage encode each observed character independently; they are distinct from contextual source-character loss.
- Tokens/char divides non-special tokens by non-whitespace source characters.
- Exact roundtrip requires `decode(encode(text)) == text` with no cleanup or normalization.
