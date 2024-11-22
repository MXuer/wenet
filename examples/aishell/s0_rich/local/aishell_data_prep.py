import os
import argparse
from pathlib import Path
from tqdm import tqdm


def do_each_set(setname: str, aishell_dir: Path, data_dir: Path, name2trans: dict):
    set_dir = aishell_dir / 'wav' / setname
    wav_files = set_dir.rglob("*.wav")
    set_data_dir = data_dir / setname
    set_data_dir.mkdir(exist_ok=True, parents=True)
    name2textp = {}
    for line in open(set_data_dir / 'text_with_p').readlines():
        name, textp = line.strip().split('\t')
        name2textp[name] = textp
    with open(set_data_dir / 'text', 'w') as ft, \
        open(set_data_dir / 'wav.scp', 'w') as fw, \
        open(set_data_dir / 'text_pure', 'w') as ftp:
        for wav_file in wav_files:
            spk = wav_file.parent.name
            # text = name2trans.get(wav_file.stem, None)
            text = name2textp.get(wav_file.stem, None)
            if text:
                fw.write(f'{wav_file.stem} {wav_file}\n')
                ft.write(f'{wav_file.stem} <{spk}> {text}\n')
                ftp.write(f'{wav_file.stem}\t{text}\n')



def main(args):
    setnames = ['train', 'dev', 'test']
    aishell_dir = Path(args.aishell_dir) / 'data_aishell'
    data_dir = Path(args.data_dir)

    trans_file = aishell_dir / 'transcript' / 'aishell_transcript_v0.8.txt'
    name2trans = {}
    for line in open(trans_file):
        name, text = line.strip().split(' ', 1)
        text = text.replace(' ', '')
        name2trans[name] = text
    for setname in setnames:
        do_each_set(setname, aishell_dir, data_dir, name2trans)




if __name__=="__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('--aishell-dir',
                        default=None,
                        type=str)
    parser.add_argument('--data-dir',
                        default='data',
                        type=str)
    args = parser.parse_args()
    main(args)