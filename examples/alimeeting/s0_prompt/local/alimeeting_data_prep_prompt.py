import os
import re
import sys
import json
import argparse
from tqdm import tqdm
from textgrid import TextGrid
from pathlib import Path
from collections import defaultdict

def clean_text(text):
    text = re.sub("▁?[，。？！]", "", text)
    text = re.sub("\(.*?\)", "", text)
    return text

def process_textgrid(textgrid_files, max_prev):
    name2info = defaultdict(dict)
    for tg_file in textgrid_files:
        tg = TextGrid()
        tg.read(tg_file)
        item = tg.tiers[0]
        name = os.path.basename(tg_file)[:-9]
        name2info[name]["segments"] = []
        prev_tn_text = ""
        prev_itn_text = ""
        gender = name.split("_")[2]
        if gender not in ["M", "F"]:
            gender = None
        for index, interval in enumerate(item):
            segname = name + "--" + "%04d"%(index + 1)
            mint, maxt, text = float(interval.minTime), float(interval.maxTime), interval.mark
            text = re.sub("([\u4e00-\u9fa5]) +", lambda x:x.groups()[0], text.strip())
            # some case like 我，。 exists, so we choose the first punctuation.
            text = re.sub("([，。？！]{2,3})", lambda x:x.groups()[0][0], text)
            text = re.sub("([a-zA-Z]) +([a-zA-Z])",  lambda x:x.groups()[0]+"▁"+x.groups()[1], text)
            text = re.sub("([a-zA-Z]) ?([，。？！])",  lambda x:x.groups()[0]+"▁"+x.groups()[1], text)
            text = re.sub("([，。？！]) ?([a-zA-Z])",  lambda x:x.groups()[0]+"▁"+x.groups()[1], text)
            text = re.sub("([，。？！]) +([a-zA-Z])",  lambda x:x.groups()[0]+"▁"+x.groups()[1], text)
            text = text.upper()
            tn_text = clean_text(text)
            if text:
                name2info[name]["segments"].append(
                    {
                        "segname": segname,
                        "tn_text": tn_text,
                        "itn_text": text,
                        "start": mint,
                        "end": maxt,
                        "prev_tn_text": prev_tn_text[-max_prev:],
                        "prev_itn_text": prev_itn_text[-max_prev:],
                        "gender": gender
                    }
                )
                prev_tn_text += tn_text
                prev_itn_text += text
    return name2info


def process_each_set(data_dir, exp_dir, max_prev):
    os.makedirs(exp_dir, exist_ok=True)

    data_json_file = os.path.join(exp_dir, "data.json")

    textgrid_files = Path(data_dir).rglob("*.TextGrid")
    textgrid_files = [str(ele) for ele in textgrid_files]

    wav_files = Path(data_dir).rglob("*.wav")
    wav_files = [str(ele) for ele in wav_files]

    name2info = process_textgrid(textgrid_files, max_prev)

    final_data = []
    wavscp_file = os.path.join(exp_dir, "wav.scp")
    text_file = os.path.join(exp_dir, "text")
    with open(wavscp_file, 'w', encoding='utf-8') as f, \
        open(text_file, 'w', encoding='utf-8') as ft:
        for wav_file in wav_files:
            name = os.path.basename(wav_file)[:-4]
            f.write(f"{name} {wav_file}\n")
            for ele in name2info[name]['segments']:
                ele["wav"] = wav_file
                segname = ele['segname']
                tn_text = ele['tn_text']
                itn_text = ele['itn_text']
                final_data.append(ele)
                ft.write(f"{segname} {tn_text}\n{segname} {itn_text}\n")

    with open(data_json_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_data, ensure_ascii=False, indent=2))
    

def main(args):
    # Train 
    raw_eval_dir = os.path.join(args.data_dir, "Train_Ali_near")
    format_eval_dir = "data/train"
    process_each_set(raw_eval_dir, format_eval_dir, args.max_prev)
    # Dev 
    raw_eval_dir = os.path.join(args.data_dir, "Eval_Ali/Eval_Ali_near")
    format_eval_dir = "data/dev"
    process_each_set(raw_eval_dir, format_eval_dir, args.max_prev)
    # Test 
    raw_eval_dir = os.path.join(args.data_dir, "Test_Ali/Test_Ali_near")
    format_eval_dir = "data/test"
    process_each_set(raw_eval_dir, format_eval_dir, args.max_prev)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",
                        "-d",
                        default=None,
                        type=str,
                        help='directory for the raw data')
    parser.add_argument("--max-prev",
                        "-p",
                        default=128,
                        type=int,
                        help='maximun prev text context')
    args = parser.parse_args()
    main(args)