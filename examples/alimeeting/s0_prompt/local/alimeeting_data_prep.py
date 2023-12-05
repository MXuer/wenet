import os
import re
import sys
import argparse
from tqdm import tqdm
from textgrid import TextGrid
from pathlib import Path
from collections import defaultdict

def clean_text(text):
    text = re.sub("[，。？！]", "", text)
    text = re.sub("\(.*?\)", "", text)
    return text


def process_textgrid(textgrid_files):
    name2info = defaultdict(list)
    for tg_file in textgrid_files:
        tg = TextGrid()
        tg.read(tg_file)
        item = tg.tiers[0]
        name = os.path.basename(tg_file)[:-9]
        for index, interval in enumerate(item):
            segname = name + "--" + "%04d"%(index + 1)
            mint, maxt, text = float(interval.minTime), float(interval.maxTime), interval.mark
            text = clean_text(text)
            if text:
                name2info[name].append(
                    [segname, text, mint, maxt]
                )
    return name2info


def process_each_set(data_dir, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)

    wavscp_file = os.path.join(exp_dir, "wav.scp")
    text_file = os.path.join(exp_dir, "text")
    seg_file = os.path.join(exp_dir, "segments")

    textgrid_files = Path(data_dir).rglob("*.TextGrid")
    textgrid_files = [str(ele) for ele in textgrid_files]

    wav_files = Path(data_dir).rglob("*.wav")
    wav_files = [str(ele) for ele in wav_files]

    name2info = process_textgrid(textgrid_files)
    
    with open(wavscp_file, 'w', encoding='utf-8') as f:
        for wav_file in wav_files:
            name = os.path.basename(wav_file)[:-4]
            if name not in name2info.keys():
                print(f"Warning : {name} has wav, but no text and segment information | {wav_file}!!!")
                continue
            f.write(f"{name} {wav_file}\n")

    with open(seg_file, 'w', encoding='utf-8') as fseg, \
        open(text_file, 'w', encoding='utf-8') as ft:
        for name, info in name2info.items():
            for (segname, text, mint, maxt) in info:
                fseg.write(f"{segname} {name} {mint} {maxt}\n")
                ft.write(f"{segname} {text}\n")


def main(args):
    # Train 
    raw_eval_dir = os.path.join(args.data_dir, "Train_Ali_near")
    format_eval_dir = "data/train"
    process_each_set(raw_eval_dir, format_eval_dir)
    # Dev 
    raw_eval_dir = os.path.join(args.data_dir, "Eval_Ali/Eval_Ali_near")
    format_eval_dir = "data/dev"
    process_each_set(raw_eval_dir, format_eval_dir)
    # Test 
    raw_eval_dir = os.path.join(args.data_dir, "Test_Ali/Test_Ali_near")
    format_eval_dir = "data/test"
    process_each_set(raw_eval_dir, format_eval_dir)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        "-d",
                        default=None,
                        type=str,
                        help='directory for the raw data')
    args = parser.parse_args()
    main(args)