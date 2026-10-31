from pathlib import Path
from io import BytesIO
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent

import sentencepiece as spm

vocab_sizes = [1000, 5000, 10000]
model_types = ["bpe", "unigram"]

for model_type in model_types:
    for vocab_size in vocab_sizes:
        model_name = f"{model_type}_{vocab_size}"

        print("Обучение модели:", model_name)

        model_buffer = BytesIO()
        spm.SentencePieceTrainer.train(
            #input=str(ROOT / "data" / "Dubrovsky.txt"),
            #model_prefix=str(ROOT / "models" / model_name),
            sentence_iterator=iter((ROOT / "data" / "Dubrovsky.txt").read_text(encoding="utf-8").splitlines()),
            model_writer=model_buffer,
            model_type=model_type,
            vocab_size=vocab_size,

            split_digits=True,
            byte_fallback=False,

            max_sentence_length=10000,
            hard_vocab_limit=False,
            minloglevel=2,)

        (ROOT / "models" / (model_name + ".model")).write_bytes(
            model_buffer.getvalue())

model_names = ["bpe_1000", "bpe_5000", "bpe_10000", "unigram_1000", "unigram_5000", "unigram_10000"]

print("\nФактические размеры словарей:")

for model_name in model_names:
    sp = spm.SentencePieceProcessor()
    #sp.load(str(ROOT / "models" / (model_name + ".model")))
    sp.LoadFromSerializedProto((ROOT / "models" / (model_name + ".model")).read_bytes())
    print(model_name, "-", sp.get_piece_size(), "токенов")

