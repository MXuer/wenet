# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This file was copied and modified from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/encoders/byte_utils.py

import re
import unicodedata


WHITESPACE_NORMALIZER = re.compile(r"\s+")
SPACE = chr(32)
SPACE_ESCAPE = chr(9601)

PRINTABLE_BASE_CHARS = [
    256,
    257,
    258,
    259,
    260,
    261,
    262,
    263,
    264,
    265,
    266,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    277,
    278,
    279,
    280,
    281,
    282,
    283,
    284,
    285,
    286,
    287,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    288,
    289,
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    298,
    299,
    300,
    301,
    302,
    303,
    304,
    305,
    308,
    309,
    310,
    311,
    312,
    313,
    314,
    315,
    316,
    317,
    318,
    321,
    322,
    323,
    324,
    325,
    326,
    327,
    328,
    330,
    331,
    332,
    333,
    334,
    335,
    336,
    337,
    338,
    339,
    340,
    341,
    342,
    343,
    344,
    345,
    346,
    347,
    348,
    349,
    350,
    351,
    352,
    353,
    354,
    355,
    356,
    357,
    358,
    359,
    360,
    361,
    362,
    363,
    364,
    365,
    366,
    367,
    368,
    369,
    370,
    371,
    372,
    373,
    374,
    375,
    376,
    377,
    378,
    379,
    380,
    381,
    382,
    384,
    385,
    386,
    387,
    388,
    389,
    390,
    391,
    392,
    393,
    394,
    395,
    396,
    397,
    398,
    399,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    419,
    420,
    421,
    422,
]

for c in PRINTABLE_BASE_CHARS:
    assert unicodedata.normalize("NFKC", chr(c)) == chr(c), c

BYTE_TO_BCHAR = {b: chr(PRINTABLE_BASE_CHARS[b]) for b in range(256)}
BCHAR_TO_BYTE = {bc: b for b, bc in BYTE_TO_BCHAR.items()}


def byte_encode(x: str) -> str:
    normalized = WHITESPACE_NORMALIZER.sub(SPACE, x)
    return "".join([BYTE_TO_BCHAR[b] for b in normalized.encode("utf-8")])


def byte_decode(x: str) -> str:
    try:
        return bytes([BCHAR_TO_BYTE[bc] for bc in x]).decode("utf-8")
    except ValueError:
        return ""


def smart_byte_decode(x: str) -> str:
    x = x.replace("▁", "")
    output = byte_decode(x)
    if output == "":
        # DP the best recovery (max valid chars) if it's broken
        n_bytes = len(x)
        f = [0 for _ in range(n_bytes + 1)]
        pt = [0 for _ in range(n_bytes + 1)]
        for i in range(1, n_bytes + 1):
            f[i], pt[i] = f[i - 1], i - 1
            for j in range(1, min(4, i) + 1):
                if f[i - j] + 1 > f[i] and len(byte_decode(x[i - j : i])) > 0:
                    f[i], pt[i] = f[i - j] + 1, i - j
        cur_pt = n_bytes
        while cur_pt > 0:
            if f[cur_pt] == f[pt[cur_pt]] + 1:
                output = byte_decode(x[pt[cur_pt] : cur_pt]) + output
            cur_pt = pt[cur_pt]
    return output


def tokenize_by_CJK_char(line: str) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return charaters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = pattern.split(line.strip().upper())
    return " ".join([w.strip() for w in chars if w.strip()])



