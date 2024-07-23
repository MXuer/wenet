import os
import re
import json
import random
from tqdm import tqdm
from pathlib import Path
from pydub import AudioSegment
from collections import defaultdict

def main():
    json_file = "local/lang2projs.json"
    fake_data = "data/fake"
    lang2projs = json.load(open(json_file))
    proj_infos = defaultdict(list)
    for proj in tqdm(os.listdir("data")):
        if not proj.startswith("King"):
            continue
        test_dir = Path(os.path.join('data', proj, 'test'))
        wav_scp_file = test_dir / "wav.scp"
        segments_file = test_dir / "segments"
        text_file = test_dir / "text"
        name2wav_file = {}
        name2text = {}
        for line in open(wav_scp_file).readlines():
            name, wavp = line.strip().split()
            name2wav_file[name] = wavp

        for line in open(text_file).readlines():
            name, text = line.strip().split(" ", 1)
            text = re.sub("<.*?>", "", text).strip()
            name2text[name] = text
        
        for line in open(segments_file).readlines():
            subname, name, start, end = line.strip().split()
            start, end = float(start), float(end)
            wavp = name2wav_file[name]
            text = name2text[subname]
            proj_infos[proj].append([subname, start, end, text, wavp])
    each_total = 5000
    done_langs = []
    for a_lang, a_projs in lang2projs.items():
        for b_lang, b_projs in lang2projs.items():
            if a_lang == b_lang:
                continue
            if (a_lang, b_lang) in done_langs:
                continue
            lang_data = []
            des_lang = f"{a_lang}___{b_lang}".replace("<", "").replace(">", "")
            if os.path.exists(os.path.join(fake_data, f".{des_lang}.done")):
                continue
            des_dir = os.path.join(fake_data, des_lang)
            wav_dir = os.path.join(des_dir, 'wav')
            os.makedirs(wav_dir, exist_ok=True)
            print(f"{a_lang}___{b_lang}".replace("<", "").replace(">", ""))
            for count in tqdm(range(each_total)):
                try:
                    a_proj = random.sample(a_projs, 1)[0]
                    b_proj = random.sample(b_projs, 1)[0]
                    a_info = random.sample(proj_infos[a_proj], 1)[0]
                    b_info = random.sample(proj_infos[b_proj], 1)[0]
                except Exception:
                    continue
                empty = AudioSegment.empty()
                text_merge = ""
                a_piece = AudioSegment.from_wav(a_info[4])[int(a_info[1]*1000):int(a_info[2]*1000)]
                b_piece = AudioSegment.from_wav(b_info[4])[int(b_info[1]*1000):int(b_info[2]*1000)]
                if random.random() <= 0.5:
                    empty += a_piece
                    empty += b_piece
                    text_merge = a_info[3] + b_info[3]
                else:
                    empty += b_piece
                    empty += a_piece
                    text_merge = b_info[3] + a_info[3]
                fake_wavfile = os.path.join(wav_dir, f"{a_info[0]}____{b_info[0]}.wav")
                empty.export(fake_wavfile, format='wav')
                lang_data.append([os.path.basename(fake_wavfile)[:-4], text_merge, fake_wavfile])
            with open(f"{des_dir}/wav.scp", 'w') as fw, \
                open(f"{des_dir}/text", 'w') as ft:
                for ele in lang_data:
                    fw.write(f"{ele[0]} {ele[2]}\n")
                    ft.write(f"{ele[0]} {ele[1]}\n")

            done_langs.append((a_lang, b_lang))
            
            f = open(os.path.join(fake_data, f".{des_lang}.done"), 'w')
            f.close()
    


if __name__=="__main__":
    main()

    
