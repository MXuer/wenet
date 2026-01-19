#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2021 JD AI Lab. All Rights Reserved. (authors: Lu Fan)
# Copyright 2021 Mobvoi Inc. All Rights Reserved. (Di Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import re
import sys

is_python2 = sys.version_info[0] == 2


def exist_or_not(i, match_pos):
    start_pos = None
    end_pos = None
    for pos in match_pos:
        if pos[0] <= i < pos[1]:
            start_pos = pos[0]
            end_pos = pos[1]
            break

    return start_pos, end_pos


def seg_char(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip()) > 0]
    return chars


def get_parser():
    parser = argparse.ArgumentParser(
        description='convert raw text to tokenized text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nchar',
                        '-n',
                        default=1,
                        type=int,
                        help='number of characters to split, i.e., \
                        aabb -> a a b b with -n 1 and aa bb with -n 2')
    parser.add_argument('--skip-ncols',
                        '-s',
                        default=0,
                        type=int,
                        help='skip first n columns')
    parser.add_argument('--space',
                        default='<space>',
                        type=str,
                        help='space symbol')
    parser.add_argument('--bpe-model',
                        '-m',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--non-lang-syms',
                        '-l',
                        default=None,
                        type=str,
                        help='list of non-linguistic symobles,'
                        ' e.g., <NOISE> etc.')
    parser.add_argument('text',
                        type=str,
                        default=False,
                        nargs='?',
                        help='input text')
    parser.add_argument('--trans_type',
                        '-t',
                        type=str,
                        default="char",
                        choices=["char", "phn", "cn_char_en_bpe"],
                        help="""Transcript type. char/phn. e.g., for TIMIT
                             FADG0_SI1279 -
                             If trans_type is char, read from
                             SI1279.WRD file -> "bricks are an alternative"
                             Else if trans_type is phn,
                             read from SI1279.PHN file ->
                             "sil b r ih sil k s aa r er n aa l
                             sil t er n ih sil t ih v sil" """)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    rs = []
    if args.non_lang_syms is not None:
        with codecs.open(args.non_lang_syms, 'r', encoding="utf-8") as f:
            nls = [x.rstrip() for x in f.readlines()]
            rs = [re.compile(re.escape(x)) for x in nls]

    if args.bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)

    if args.text:
        f = codecs.open(args.text, encoding="utf-8")
    else:
        f = codecs.getreader("utf-8")(
            sys.stdin if is_python2 else sys.stdin.buffer)

    sys.stdout = codecs.getwriter("utf-8")(
        sys.stdout if is_python2 else sys.stdout.buffer)
    line = f.readline()
    n = args.nchar
    from tqdm import tqdm

    vocab = []
    for index, line in tqdm(enumerate(f.readlines())):
        try:
            if '\t' in line:
                text = line.strip().split('\t')[1]
            else:
                text = line.strip().split(' ', 1)[1]
            if re.findall('[0-9#%*+]', text):
                print(text)
                continue
        except Exception:
            print(line.strip())
            continue
        text = re.sub(r'([\u4e00-\u9fff])', r' \1 ', text)
        pieces = sp.encode_as_pieces(text)
        if '▁' in pieces:
            pieces.remove('▁')
        vocab.extend(pieces)
        if index % 50000 == 0:
            vocab = list(set(vocab))
            
    vocab = sorted(list(set(vocab)))
    if '▁' in vocab:
        vocab.remove('▁')
    with open('data/dict_bpe/lang_char.txt', 'w') as f:
        f.write("<blank> 0\n")
        f.write("<unk> 1\n")
        f.write("<sos/eos> 2\n")
        f.write("▁ 3\n")
        for index, v in enumerate(vocab):
            f.write(f'{v} {index+4}\n')
        
        
if __name__ == '__main__':
    main()
