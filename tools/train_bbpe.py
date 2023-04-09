import argparse
import tempfile
import shutil
import sentencepiece as spm

from pathlib import Path
from wenet.utils.bbpe import byte_encode, tokenize_by_CJK_char
from pathlib import Path
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        The generated bpe.model is saved to this directory.
        """,
    )

    parser.add_argument(
        "--transcript",
        type=str,
        help="Training transcript.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size for BPE training",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="unigram",
        help="Vocabulary size for BPE training",
    )

    parser.add_argument(
        "--extremely_large",
        action='store_true',
        default=False,
        help='whether the corpus is extremely large or not'
    )

    return parser.parse_args()


def _convert_to_bchar(in_path: str, out_path: str):
    with open(out_path, "w") as f:
        for line in tqdm(open(in_path, "r").readlines(), desc="reading text..."):
            try:
                line = line.strip()
                if "\t" in line:
                    line = line.split("\t", 1)[1]
                else:
                    line = line.split(" ", 1)[1]
                f.write(byte_encode(tokenize_by_CJK_char(line)) + "\n")
            except Exception as e:
                print(f"{line} with error {e}.")


def main():
    args = get_args()
    vocab_size = args.vocab_size
    lang_dir = Path(args.lang_dir)

    model_prefix = f"{lang_dir}/{args.model_type}_{vocab_size}"
    character_coverage = 1.0
    input_sentence_size = 100000000



    temp = tempfile.NamedTemporaryFile()
    train_text = temp.name

    _convert_to_bchar(args.transcript, train_text)
    model_file = Path(model_prefix + ".model")

    # no user_defined_symbols and unk_id because they will
    # be added afterwards.
    if not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=train_text,
            vocab_size=vocab_size,
            model_type=args.model_type,
            model_prefix=model_prefix,
            input_sentence_size=input_sentence_size,
            character_coverage=character_coverage,
            bos_id=-1,
            eos_id=-1,
            minloglevel=2,
            train_extremely_large_corpus=args.extremely_large
        )
    else:
        print(f"{model_file} exists - skipping")
        return

    shutil.copyfile(model_file, f"{lang_dir}/bbpe.model")


if __name__ == "__main__":
    main()
