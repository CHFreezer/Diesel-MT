# Tokenizer coverage report: 32k

> Preliminary smoke result: this tokenizer was trained with sample_fraction=0.1; do not use it for final TD-07 selection.

## Provenance

- Artifact: `artifacts/tokenizers/smoke-10pct/mvp-32k`
- Vocabulary size: 32,768
- Training sample fraction: 0.1
- Evaluation seed: 20260713
- Corpus samples per language: 500
- Evaluation corpus manifest SHA-256: `11daef6a8b38c7dc66dc17d51ab4ab2ab9053a0816340261ad76e53740be0477`

## Corpus metrics

| Language | Samples | Mean tokens | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage | Exact roundtrip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 500 | 197.8400 | 0.3757 | 0.003033% | 0.001139% | 99.998861% | 97.959184% | 99.600000% |
| zho_Hans | 500 | 230.1980 | 0.7532 | 0.005213% | 0.003927% | 99.996073% | 99.834711% | 98.800000% |
| jpn_Jpan | 500 | 140.6580 | 0.6066 | 0.001422% | 0.000862% | 99.999138% | 99.955157% | 99.800000% |
| kor_Hang | 500 | 252.8660 | 0.7280 | 0.001582% | 0.001152% | 99.998848% | 99.930216% | 99.800000% |

## Token length distribution

| Language | P50 | P95 | P99 | Max | >500-char samples | Long-sample UNK |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 92 | 721 | 1625 | 4266 | 152 | 0.000000% |
| zho_Hans | 175 | 598 | 1292 | 4063 | 72 | 0.004053% |
| jpn_Jpan | 52 | 485 | 1548 | 4002 | 60 | 0.000000% |
| kor_Hang | 206 | 625 | 1648 | 2090 | 215 | 0.001985% |

## Synthetic stress probes

| Language | Category | Samples | Tokens | UNK tokens | Source loss | Exact roundtrip |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | daily | 1 | 20 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | mixed_language | 1 | 29 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | numeric_punctuation | 2 | 63 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | rare_unicode | 1 | 18 | 27.777778% | 37.500000% | 0.000000% |
| eng_Latn | technical_news | 1 | 29 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | very_short | 1 | 2 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | daily | 1 | 21 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | mixed_language | 1 | 28 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | numeric_punctuation | 2 | 50 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | rare_unicode | 1 | 14 | 14.285714% | 33.333333% | 0.000000% |
| zho_Hans | technical_news | 1 | 24 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | very_short | 1 | 2 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | daily | 1 | 19 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | mixed_language | 1 | 28 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | numeric_punctuation | 2 | 53 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | rare_unicode | 1 | 13 | 15.384615% | 27.272727% | 0.000000% |
| jpn_Jpan | technical_news | 1 | 24 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | very_short | 1 | 2 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | daily | 1 | 16 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | mixed_language | 1 | 29 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | numeric_punctuation | 2 | 58 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | rare_unicode | 1 | 18 | 27.777778% | 40.000000% | 0.000000% |
| kor_Hang | technical_news | 1 | 23 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | very_short | 1 | 1 | 0.000000% | 0.000000% | 100.000000% |

## Script-specific analysis

- Shared Chinese/Japanese Han: 1,100 unique, 100.000000% covered, 0 standalone source-language split mismatches.
- Korean Hangul: 1,218 unique syllables/Jamo, 100.000000% frequency-weighted coverage and 100.000000% unique coverage.
- English subwords: 1.7538 pieces/word, 59.840395% one-piece words, 2.820435% over four pieces.
- Evaluation vocabulary utilization: 14,365/32,768 (43.838501%).

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
| 矆 | U+77C6 | Han | 1 | 留言板），銆愭寜娌堥槼涓滆蒋鏁板瓧鍖荤枟绯荤粺鑲′唤鏈夐檺鍏徃鏌ヨ鍖荤枟鍣ㄦ銆戞矆闃充笢杞暟瀛楀尰鐤楃郴缁熻偂浠芥湁闄愬叕鍙镐骇鍝佸ぇ鍏▅鍖荤枟鍣ㄦ鏌ヨ缃戠珯。市 |
| 筄 | U+7B44 | Han | 1 | 戈筄窾货苯ず杜ㄩ，GDC大奖提名出炉 《死亡搁浅》独占七项领衔名单，新年贺词奏响20 |
| 錽 | U+933D | Han | 1 | 拼搏pinbo体育\|官网，研讨贴，錽金银器具如何辨别新老，拿什么拯救你，心脏骤停，猛降到29万，特斯拉的屠杀开始了  |

### jpn_Jpan

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| ᗨ | U+15E8 | Other letter | 1 | 未経験の子が一番稼いでくれてます.(❁′ᗨ‵❁)求人で応募してくれた方は※正直お給料は上げやすいです!!（スカウトを通すと |

### kor_Hang

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| 😚 | U+1F61A | Symbol/emoji | 2 | 👍 꽃속에서 폭 올라온 늬낌으로 꽃하니가 되었어요 너무 사랑스러워 보여요😚🥰💕 진짜 소품 배치 하나하나 자연스럽고 인물과 넘 잘 어울러지는… 진정 |

## Metric definitions

- Main metrics use only the fixed corpus sample; synthetic probes are reported separately.
- Source-character loss counts non-whitespace Unicode code points covered by `<unk>` offsets. A fused unknown span may count as several lost characters.
- Frequency-weighted and unique-character coverage encode each observed character independently; they are distinct from contextual source-character loss.
- Tokens/char divides non-special tokens by non-whitespace source characters.
- Exact roundtrip requires `decode(encode(text)) == text` with no cleanup or normalization.
