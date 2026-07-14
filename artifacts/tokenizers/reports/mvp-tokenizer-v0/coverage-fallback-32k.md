# Tokenizer coverage report: fallback-32k

> Sampled-corpus result: this tokenizer was trained with sample_fraction=0.5. Compare it only with candidates that use the same training provenance; this report does not claim full-corpus coverage.

## Provenance

- Artifact: `artifacts/tokenizers/50pct/mvp-32k`
- Vocabulary size: 32,768
- Training sample fraction: 0.5
- Evaluation seed: 20260713
- Corpus samples per language: 500
- Evaluation corpus manifest SHA-256: `c5bec116578ea88d37f325c3e18c66a889ef34aa263bb876e821456c500f9ffe`

## Corpus metrics

| Language | Samples | Mean tokens | Tokens/char | UNK tokens | Source loss | Freq coverage | Unique coverage | Exact roundtrip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 500 | 132.4380 | 0.3691 | 0.000000% | 0.000000% | 100.000000% | 100.000000% | 100.000000% |
| zho_Hans | 500 | 124.1860 | 0.7519 | 0.004831% | 0.003633% | 99.996367% | 99.929874% | 99.800000% |
| zho_Hant | 500 | 187.5240 | 0.9327 | 0.427679% | 0.408862% | 99.591138% | 95.145889% | 72.800000% |
| jpn_Jpan | 500 | 102.6940 | 0.6135 | 0.000000% | 0.000000% | 100.000000% | 100.000000% | 100.000000% |
| kor_Hang | 500 | 182.3980 | 0.7257 | 0.000000% | 0.000000% | 100.000000% | 100.000000% | 100.000000% |

## Token length distribution

| Language | P50 | P95 | P99 | Max | >500-char samples | Long-sample UNK |
| --- | --- | --- | --- | --- | --- | --- |
| eng_Latn | 83 | 482 | 619 | 879 | 134 | 0.000000% |
| zho_Hans | 73 | 367 | 830 | 1232 | 25 | 0.018339% |
| zho_Hant | 73 | 864 | 1201 | 1436 | 65 | 0.413688% |
| jpn_Jpan | 50 | 388 | 762 | 923 | 38 | 0.000000% |
| kor_Hang | 116 | 465 | 915 | 1097 | 140 | 0.000000% |

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
| zho_Hant | daily | 1 | 22 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | empty | 1 | 0 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | mixed_language | 1 | 29 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | numeric_punctuation | 2 | 52 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | rare_unicode | 1 | 14 | 14.285714% | 33.333333% | 0.000000% |
| zho_Hant | technical_news | 1 | 29 | 0.000000% | 0.000000% | 100.000000% |
| zho_Hant | very_short | 1 | 2 | 0.000000% | 0.000000% | 100.000000% |
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
| kor_Hang | technical_news | 1 | 22 | 0.000000% | 0.000000% | 100.000000% |
| kor_Hang | very_short | 1 | 1 | 0.000000% | 0.000000% | 100.000000% |

## Script-specific analysis

- Shared Chinese/Japanese Han: 878 unique, 100.000000% covered, 0 standalone source-language split mismatches.
- Simplified/Traditional sequence parity: tokens/character 0.7519 vs 0.9327; Traditional/Simplified ratio 1.2405; P95 ratio 2.3542.
- Korean Hangul: 1,174 unique syllables/Jamo, 100.000000% frequency-weighted coverage and 100.000000% unique coverage.
- English subwords: 1.7651 pieces/word, 59.313587% one-piece words, 2.726964% over four pieces.
- Evaluation vocabulary utilization: 14,783/32,768 (45.114136%).

## Highest-frequency uncovered characters

### eng_Latn

No uncovered characters in the fixed corpus sample.

### zho_Hans

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| 餵 | U+9935 | Han | 2 | 王雯身后看到这情景，我的阳具慢慢的也硬了起来。王雯转过头对我说：「你还不快点去『餵』妻。」我没有答话，我把一只手放在她的肩上一直在看，她也没有把我推开。看了大概两 |
| 踫 | U+8E2B | Han | 1 | 背慢慢的靠在我的胸前，转过头来闭上眼睛。我把嘴凑过去把舌头伸入她口中，两根舌头一踫到就拼个你死我活，她的淫水也流到我手掌也湿了，她口裏也发出…「唔…唔…呀…」的声 |

### zho_Hant

| Character | Code point | Script | Count | Example |
| --- | --- | --- | --- | --- |
| 幫 | U+5E6B | Han | 29 | 用到現在滿意極了好了先不說 打算拿來當我第500篇心得第一步就是先用左上角的米色幫整個眼窩打底不得不說EXCEL的眼影真的是完全不飛粉耶雖然打底後眼周有些許白感但 |
| 逺 | U+903A | Han | 16 | 〈如祭禰之儀但祝辭雲嵗序流易諱日復臨追逺感時不勝永慕考妣改不勝永慕為昊天罔極旁親雲諱日復臨不勝感愴若考妣則祝興主人以下哭 |
| 傢 | U+50A2 | Han | 14 | 聽了李玄的話翼龍居然飛在前面帶路讓李玄和凡心相視苦笑這傢伙還真是聰明只是不會說話不好交流要不然會更好玩. |
| 彥 | U+5F65 | Han | 14 | 俊翔 黃 詠 蔡明璋 徐子恆 黃欣媚 鄒義峰 莊覲瑄 廖晉賢 郭曰誠 黃紹誠 黃彥文 陳柏廷 謝宜廷 廖宏基 何庚諺 蔡孟峰 林千雯 曾元康 李姿瑄 陳柏勳 李俊 |
| 𨽻 | U+28F7B | Han | 11 | 羲之子獻之字子敬〈中品下〉少有盛名風流高邁草𨽻繼父之美丹青亦工桓温嘗請畫扇誤落筆因就成烏駮牸牛極妙絶又書牸牛賦於扇上此扇義熙中 |
| 瑋 | U+744B | Han | 10 | 泰余 朱紹綺 陳鈞隆 陳映竹 潘哲毅 王立中 柯穎志 陳柏瑞 林亮華 陳芷安 林瑋信 劉尚霖 區穗中 黃 立 楊 震 洪若霓 李彥融 吳佳潔 鄭存翔 林煒楗 何其 |
| 紓 | U+7D13 | Han | 10 | 2.前置協商諮詢宜蘭汽車借貸免留車>勞工保險局紓困貸款 |
| 繳 | U+7E73 | Han | 10 | 缺錢急用錢怎麼辦~買房子要繳什麼稅1.個別協商諮詢 |
| 嵗 | U+5D57 | Han | 9 | 〈如祭禰之儀但祝辭雲嵗序流易諱日復臨追逺感時不勝永慕考妣改不勝永慕為昊天罔極旁親雲諱日復臨不勝感愴若考 |
| 樁 | U+6A01 | Han | 8 | 凡山前山後各有禁限如紅樁以內盜砍樹株取土取石開30燒造放火燒山者比照盜大祀神御物律斬奏請定奪為從者發近邊 |
| 𤣥 | U+248E5 | Han | 8 | 蕙田案天子諸侯白舄以配韋弁皮弁冠弁黒舄以配𤣥端但司服所掌九服别無𤣥端故康成於冠弁之下注云王卒食以居則𤣥端以𤣥端與冠弁大同小異 |
| 䟽 | U+47FD | Han | 7 | 官即雲職業有守〉未由奔 慰其於憂戀無任下誠〈平交已下但云某未由奉慰悲係増深〉謹奉䟽〈平交雲狀〉伏惟 鑒察〈平交以下去此四字〉不備謹䟽〈平交雲不宣謹狀〉月日具位〈降 |
| 祔 | U+7954 | Han | 7 | 自憲宗穆宗敬宗文宗四世祔廟睿𤣥肅代以次遷至武宗崩徳宗以次當遷而于世次為髙祖禮官始覺其非以謂兄弟不相為後不 |
| 䘮 | U+462E | Han | 6 | 有爵之人必有德有德則能為父母致病深故許以其杖扶病雖無爵然以適子故假取有爵之杖為之䘮主衆子雖非為主子為父母致病是同亦輔病也童子不杖此庶童子也案問䘮雲童子當室則免而杖 |
| 囉 | U+56C9 | Han | 6 | 崑崙山俘斬萬計留兵戍諸要害而還)質實(崑崙山元史志朶甘思東北有大雪山番名伊拉瑪博囉即崑崙也按伊拉瑪博囉今名阿木尼瑪勒占木遜山明師窮追至崑崙即此乃大積石元明誤以為崑 |

### jpn_Jpan

No uncovered characters in the fixed corpus sample.

### kor_Hang

No uncovered characters in the fixed corpus sample.

## Metric definitions

- Main metrics use only the fixed corpus sample; synthetic probes are reported separately.
- Source-character loss counts non-whitespace Unicode code points covered by `<unk>` offsets. A fused unknown span may count as several lost characters.
- Frequency-weighted and unique-character coverage encode each observed character independently; they are distinct from contextual source-character loss.
- Tokens/char divides non-special tokens by non-whitespace source characters.
- Exact roundtrip requires `decode(encode(text)) == text` with no cleanup or normalization.
