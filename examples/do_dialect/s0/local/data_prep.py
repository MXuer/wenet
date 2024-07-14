
import os
import sys
import yaml
import argparse

from local.soCorpusCollection import *

name2type = {
    "read": InputReadCorpusFormat,
    "diag": InputDialogCorpusFormat,
    "tr2019006": InputStrangeTGTR2019006,
    "tr2020064": InputVIPKIDTR2020064,
    "tr2020063": InputTR2020063,
    "tr2021106": InputTR2021106,
    "trtextgrid": InputTRTextGrid,
    "result_type0": InputResultTxtType0, # 可能得全部重新跑，因为/的正则不对
    "ali": InputAliTransFormat,
    "ali_advanced": InputAliTransFormatAdvanced,
    "trans_label11": InputTransFormatLabel11thLong,
    "short_withtime": InputShortWithTimeInfoTransFormat,
    "k118": InputKingASR118,
    "k214": InputKingASR214,
    "subfolder_024": InputTRTextGridSubFolder,
    "rs9th": InputRS9thTransFormat,
    "qq3th": InputQQ3thTransFormat,
    "result_type1": InputResultTxtType1, # 可能得全部重新跑，因为/的正则不对,
    "tr2017009": Inputtr2017009,
    "stm": InputSTMTransFormat,
    'dat': InputTransFormatDATLong,
    'lbl': InputTransFormatLBLShort,
    'csv': InputTransFormatCSVLong,
    'xlsx': InputTransFormatXLSXShort,
    'rsdp': InputRSDPTransFormat,
    'kobytedance': InputBytedanceKOKR,
    'jabytedance': InputBytedanceJAJP,
    'ptbytedance': InputBytedancePTBR,
    'tts': InputTTSFormat
}

def Touch(path):
    f = open(path, 'w')
    f.close()

def main(args):
        config = yaml.load(open(args.conf_file), Loader=yaml.FullLoader)
        if args.task:
            task = args.task
        else:
            task = config['task']

        if task != "all":
            print(f"=> Processing {task}...")
            task_type = config['each'][task]['type']
            task_conf = config['each'][task]['conf']
            done_file = os.path.join(task_conf["kformat_dir"], task+".done")
            if os.path.exists(done_file):
                return
            print(task_conf)
            task_conf["data_dir"] = os.path.join(task_conf["data_dir"], task)
            task_conf["kformat_dir"] = os.path.join(task_conf["kformat_dir"], task)
            inputGenerator = name2type[task_type](**task_conf)
            inputGenerator.process()
            Touch(done_file)
        else:
            for each in config['each'].keys():
                task_type = config['each'][each]['type']
                task_conf = config['each'][each]['conf']
                done_file = os.path.join(task_conf["kformat_dir"], each+".done")
                if os.path.exists(done_file):
                    continue
                print(f"=> Processing {each}")
                print(task_conf)
                task_conf["data_dir"] = os.path.join(task_conf["data_dir"], each)
                task_conf["kformat_dir"] = os.path.join(task_conf["kformat_dir"], each)
                inputGenerator = name2type[task_type](**task_conf)
                try:
                    inputGenerator.process()
                    Touch(done_file)
                except Exception as e:
                    print(e)
                    continue

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-file", default="conf/data.yaml", type=str)
    parser.add_argument("--task", default="", type=str)
    args = parser.parse_args()
    main(args)
