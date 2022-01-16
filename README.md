# Comparing Writing Systems with Multilingual Grapheme-to-Phoneme and Phoneme-to-Grapheme Conversion

## Data

SIGMORPHON 2021 Task 1 medium-resource data is used.
This data was extracted from the English-language portion of
[Wiktionary](https://en.wiktionary.org/) using
[WikiPron](https://github.com/kylebgorman/wikipron) (Lee et al. 2020), then
filtered and downsampled using proprietary techniques.

## Format

Training and development data are UTF-8-encoded tab-separated values files. Each
example occupies a single line and consists of a grapheme sequence&mdash;a sequence
of [NFC](https://en.wikipedia.org/wiki/Unicode_equivalence#Normal_forms) Unicode
codepoints&mdash;a tab character, and the corresponding phone sequence, a
roughly-phonemic IPA, tokenized using the
[`segments`](https://github.com/cldf/segments) library. The following shows
three lines of Romanian data:

    antonim a n t o n i m
    ploaie  p lʷ a j e
    pornește    p o r n e ʃ t e

## Languages

SIGMORPHON 2021 Task 1 medium-resource data consists of 10,000 words from the following ten languages. The data is randomly split into training (80%), development (10%), and testing (10%) data.

1.  `arm_e`: Armenian (Eastern dialect)
2.  `bul`: Bulgarian
3.  `dut`: Dutch
4.  `fre`: French
5.  `geo`: Georgian
6.  `hbs_latn`: Serbo-Croatian (Latin script)
7.  `hun`: Hungarian
8.  `jpn_hira`: Japanese (Hiragana script)
9.  `kor`: Korean
10. `vie_hanoi`: Vietnamese (Hanoi dialect)

## Evaluation

The metric used to rank systems is *word error rate* (WER), the percentage of
words for which the hypothesized transcription sequence does not match the gold
transcription. This value, in accordance with common practice, is a decimal
value multiplied by 100 (e.g.: 13.53). In the medium- and low-frequency tasks,
WER is macro-averaged across all ten languages. We provide two Python scripts
for evaluation:

-   [`evaluate.py`](evaluation/evaluate.py) computes the WER for one language.
-   [`evaluate_all.py`](evaluation/evaluate_all.py) computes per-language and average WER across multiple languages.