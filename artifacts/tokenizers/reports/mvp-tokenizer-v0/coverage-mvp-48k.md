# Tokenizer coverage report: mvp-48k

## Provenance

- Artifact: `artifacts/tokenizers/mvp-tokenizer-v0`
- Vocabulary size: 49,152
- Training sample fraction: 1.0
- Evaluation seed: 20260713
- Corpus samples per language: 500
- Evaluation corpus manifest SHA-256: `c5bec116578ea88d37f325c3e18c66a889ef34aa263bb876e821456c500f9ffe`

## Corpus metrics

| Language | Samples | Mean tokens | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage | Exact roundtrip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 500 | 113.0780 | 0.3152 | 0.001769% | 0.000557% | 99.999443% | 99.107143% | 99.800000% |
| zho_Hans | 500 | 108.9960 | 0.6599 | 0.011010% | 0.007265% | 99.992735% | 99.859748% | 99.600000% |
| zho_Hant | 500 | 141.6100 | 0.7044 | 0.060730% | 0.043771% | 99.956229% | 98.912467% | 95.600000% |
| jpn_Jpan | 500 | 89.1380 | 0.5325 | 0.000000% | 0.000000% | 100.000000% | 100.000000% | 100.000000% |
| kor_Hang | 500 | 159.6340 | 0.6351 | 0.003759% | 0.002387% | 99.997613% | 99.787686% | 99.400000% |

## Token length distribution

| Language | P50 | P95 | P99 | Max | >500-char samples | Long-sample UNK |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 70 | 420 | 539 | 793 | 134 | 0.002668% |
| zho_Hans | 64 | 329 | 727 | 1098 | 25 | 0.020744% |
| zho_Hant | 58 | 579 | 951 | 1245 | 65 | 0.072128% |
| jpn_Jpan | 43 | 350 | 689 | 821 | 38 | 0.000000% |
| kor_Hang | 101 | 411 | 778 | 957 | 140 | 0.001983% |

## Synthetic stress probes

| Language | Category | Samples | Tokens | UNK tokens | Source loss | Exact roundtrip |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | daily | 1 | 18 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | mixed_language | 1 | 22 | 4.545455% | 2.500000% | 0.000000% |
| eng_Latn | numeric_punctuation | 2 | 57 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | rare_unicode | 1 | 18 | 27.777778% | 37.500000% | 0.000000% |
| eng_Latn | technical_news | 1 | 24 | 0.000000% | 0.000000% | 100.000000% |
| eng_Latn | very_short | 1 | 2 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | daily | 1 | 16 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | mixed_language | 1 | 23 | 4.347826% | 2.857143% | 0.000000% |
| zho_Hans | numeric_punctuation | 2 | 49 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | rare_unicode | 1 | 13 | 15.384615% | 33.333333% | 0.000000% |
| zho_Hans | technical_news | 1 | 22 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hans | very_short | 1 | 1 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | daily | 1 | 16 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | mixed_language | 1 | 25 | 4.000000% | 2.702703% | 0.000000% |
| zho_Hant | numeric_punctuation | 2 | 49 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | rare_unicode | 1 | 13 | 15.384615% | 33.333333% | 0.000000% |
| zho_Hant | technical_news | 1 | 22 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | very_short | 1 | 1 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | daily | 1 | 17 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | mixed_language | 1 | 22 | 4.545455% | 2.631579% | 0.000000% |
| jpn_Jpan | numeric_punctuation | 2 | 51 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | rare_unicode | 1 | 13 | 15.384615% | 27.272727% | 0.000000% |
| jpn_Jpan | technical_news | 1 | 19 | 0.000000% | 0.000000% | 100.000000% |
| jpn_Jpan | very_short | 1 | 2 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | daily | 1 | 14 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | mixed_language | 1 | 24 | 4.166667% | 2.777778% | 0.000000% |
| kor_Hang | numeric_punctuation | 2 | 54 | 1.851852% | 1.428571% | 50.000000% |
| kor_Hang | rare_unicode | 1 | 18 | 27.777778% | 40.000000% | 0.000000% |
| kor_Hang | technical_news | 1 | 20 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | very_short | 1 | 1 | 0.000000% | 0.000000% | 100.000000% |

## Script-specific analysis

- Shared Chinese/Japanese Han: 878 unique, 100.000000% covered, 0 standalone source-language split mismatches.
- Simplified/Traditional sequence parity: tokens/character 0.6599 vs 0.7044; Traditional/Simplified ratio 1.0674; P95 ratio 1.7599.
- Korean Hangul: 1,174 unique syllables/Jamo, 100.000000% frequency-weighted coverage and 100.000000% unique coverage.
- English subwords: 1.4964 pieces/word, 70.164807% one-piece words, 1.203489% over four pieces.
- Evaluation vocabulary utilization: 27,819/49,152 (56.597900%).

## Highest-frequency uncovered characters

### eng_Latn

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| ż | U+017C | Other letter | 1 | e a nuclear power division, co oznacza, że wartość euro (licznik stosunku EURUSD) |

### zho_Hans

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| Ž | U+017D | Other letter | 3 | ski trg 2到达方式：42路、46路、51路、91路公交车到Glavna Železni čka Stanica下车。电话：+381-11-3602899网 |
| ň | U+0148 | Other letter | 1 | ?。ò耍┘於ňΦ卑凑展娑ǖ奶跫?，贮存的方式和时间不应当对检定菌的生长特性有不利影响。 |
| ǖ | U+01D6 | Other letter | 1 | ?。ò耍┘於ňΦ卑凑展娑ǖ奶跫?，贮存的方式和时间不应当对检定菌的生长特性有不利影响。 |
| ┘ | U+2518 | Symbol/emoji | 1 | ?。ò耍┘於ňΦ卑凑展娑ǖ奶跫?，贮存的方式和时间不应当对检定菌的生长特性有不利影响。 |

### zho_Hant

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| 䞇 | U+4787 | Han | 2 | 信任也〈翟萇伯翻〉懷光又言其罪上亦為殺之〈亦為于偽翻〉乙丑以翰林學士祠部員外郎陸䞇為考功郎中金部員外郎吴通㣲為職方郎中〈祠部屬禮部掌祠部考功屬吏部掌文武官功過考法 |
| 牸 | U+7278 | Han | 2 | 字子敬〈中品下〉少有盛名風流高邁草𨽻繼父之美丹青亦工桓温嘗請畫扇誤落筆因就成烏駮牸牛極妙絶又書牸牛賦於扇上此扇義熙中猶在官至中書令太元十一年卒年四十三贈侍中特進光 |
| 頀 | U+9800 | Han | 2 | 列女田氏〈楊頀妻永樂間頀囘自北京卒於途田氏聞之哭泣不食暨柩至自經死事聞旌其門曰貞烈〉 |
| ۞ | U+06DE | Symbol/emoji | 1 | 《鬼灭之刃恋柱乳液飙出》👤被强行糟蹋过程小说下午也不想去๑۞๑日本黄色一级片视频我們幾個向著那個地板上炸出來的🌍东宫 电视剧「鏡兒宮」上下是 |
| ◐ | U+25D0 | Symbol/emoji | 1 | 入陵墓,還沒○以爱为契我勉強辯認的讀了一會兒.再✿忘忧草在线免费视频而且地上的(◐钓系美人和疯犬如果這天宮是在中間的三聖山的懸🥔日本天天操我在黑暗🥣chinese |
| ♍ | U+264D | Symbol/emoji | 1 | 暗🥣chinese小帅gayxxxxx方洞之中🤣stoya在线观看「你別不是鬼絆♍外婆的家在线观看完整免费在這裡猶豫,也♥️美女脱得只剩皮肤我乘機貓腰從兩個🛣️亚 |
| 偛 | U+505B | Han | 1 | 施工技術為客戶創造一流的精品工程.不斷開發和引進世界級先進科學技術和產品服務??偛炕厥髣撔缕髽I2008年1月,擔保授信信用企業2009年4月.國家認證的玻璃鋼格 |
| 偨 | U+5068 | Han | 1 | 玻璃鋼保溫罩銷售網先河制定了,做全球最專業的高端環境監測儀器供應商的戰略目標??偨Y以改善思考和行為應用將學習中新的啟示.是國家重點高新技術創新企業國家火炬計劃新 |
| 僎 | U+50CE | Han | 1 | 亨都指揮卜萬給事中葉福江西副使程本立大理寺丞鄒瑾御史魏冕秦府長史鄒樸寧府左長史石僎漳州府教授陳思賢蕭縣知縣鄭恕副都御史陳性善都指揮使崇剛御史林英左拾遺戴德彛給事中 |
| 僾 | U+50FE | Han | 1 | 陵前立仗石馬曰大白小白葢太宗當日所乘以畧陣破堅者) 萬青永䕶玉峯標元孫五世親陳奠僾愾聲容未覺遙 |
| 凞 | U+51DE | Han | 1 | 從得此經年年僧自恣日依之修設已經多歲矣講者說文云和解也誥謂佛勅即此經也誥亦作告劉凞釋名云上勅下曰告告者覺也使覺悟知己意也如尚書大誥等佛經既是覺悟眾生令知佛意故得稱 |
| 寁 | U+5BC1 | Han | 1 | 〈愚按交道之難至于與為故與為好甚不易得不寁者不速也言非朝夕𠩄能速成也而忍遽然決絶乎〉遵大路二章章四句〈何氏謂此詩周公卿欲留 |
| 撔 | U+6494 | Han | 1 | 為客戶創造一流的精品工程.不斷開發和引進世界級先進科學技術和產品服務??偛炕厥髣撔缕髽I2008年1月,擔保授信信用企業2009年4月.國家認證的玻璃鋼格柵玻璃鋼 |
| 琁 | U+7401 | Han | 1 | 仁愷 張懿芬 吳根培 吳易軒 黃璧蒼 何玠輝 周盈安 王子嘉 彭邦碩 梁心柔 楊琁茗 胡耕寧 吳智惠 陳彥秀 汪光仲 葉采衢 李建霖 羅鈺婷 魏佑慈 石宏凱 黃堅 |
| 甡 | U+7521 | Han | 1 | 涵 王莨惟 李文瑋 陳冠佑 王煜豪 葉家齊 呂佳容 李彥輝 吳尚賢 高珮容 張睿甡 王捷莓 陳志嘉 葉哲輝 廖冠豪 黃柏凱 林語立 薛逸廷 鍾凱丞 莊子逸 蔡豐璟 |

### jpn_Jpan

No uncovered characters in the fixed corpus sample.

### kor_Hang

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| Ш | U+0428 | Other letter | 1 | 물은 이 작품의 아이디어를 제공한 푸키레프의 친구 표트르 쉬멜코프(П. ШMeлbkoB)로 전해진다. 그런데 쉬멜코프의 얼굴은 평생 결혼에 대해  |
| ⅰ | U+2170 | Number | 1 | 는 소는 전소의 확정판결의 기판력에 저촉되지 않기 때문에 적법하다. 즉 ⅰ) 전소인 건물철거소송과 후소인 매매대금지급청구소송은 양립 가능한 청구로 |
| ￦ | U+FFE6 | Symbol/emoji | 1 | • 1개월 구독 시 ￦14,536/월 |

## Metric definitions

- Main metrics use only the fixed corpus sample; synthetic probes are reported separately.
- Source-character loss counts non-whitespace Unicode code points covered by `<unk>` offsets. A fused unknown span may count as several lost characters.
- Frequency-weighted and unique-character coverage encode each observed character independently; they are distinct from contextual source-character loss.
- Tokens/char divides non-special tokens by non-whitespace source characters.
- Exact roundtrip requires `decode(encode(text)) == text` with no cleanup or normalization.
