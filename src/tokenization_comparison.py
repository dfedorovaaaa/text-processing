from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import sentencepiece as spm

model_names = ["bpe_1000", "bpe_5000", "bpe_10000", "unigram_1000", "unigram_5000", "unigram_10000"]

with open(str(ROOT / "data" / "Dubrovsky.txt"), "r", encoding="utf-8") as file:
    text = file.read()

paragraphs = text.split("\n\n")

selected_paragraphs = []

for paragraph in paragraphs:
    paragraph = paragraph.strip()

    if len(paragraph) > 200:
        selected_paragraphs.append(paragraph)

    if len(selected_paragraphs) == 3:
        break

#for i in range(len(selected_paragraphs)):
#    print("\nАБЗАЦ", i + 1)
#    print(selected_paragraphs[i])

def word_tokenization(text):
    punctuation = '*.,!?;:()[]{}"«»—–…'

    for symbol in punctuation:
        text = text.replace(symbol, " ")

    return text.split()

models = {}

for model_name in model_names:
    sp = spm.SentencePieceProcessor()
    model_path = ROOT / "models" / (model_name + ".model")

    with open(model_path, "rb") as f:
        sp.LoadFromSerializedProto(f.read())
    models[model_name] = sp

for i in range(len(selected_paragraphs)):
    paragraph = selected_paragraphs[i]
    print("\nАБЗАЦ", i + 1)
    print("\nИсходный текст:")
    print(paragraph)

    word_tokens = word_tokenization(paragraph)

    print("\nПословная токенизация:")
    print(word_tokens)
    print("Количество токенов:", len(word_tokens))

    for model_name in model_names:
        tokens = models[model_name].encode(paragraph, out_type=str)

        print("\n", model_name, sep="")
        print(tokens)
        print("Количество токенов:", len(tokens))

print("\n\nИТОГОВОЕ СРАВНЕНИЕ КОЛИЧЕСТВА ТОКЕНОВ\n")

print(
    f"{'Абзац':<8}"
    f"{'Words':<10}"
    f"{'BPE1000':<10}"
    f"{'BPE5000':<10}"
    f"{'BPE10000':<12}"
    f"{'UNI1000':<10}"
    f"{'UNI5000':<10}"
    f"{'UNI10000':<10}"
)

for i in range(len(selected_paragraphs)):
    paragraph = selected_paragraphs[i]
    word_count = len(word_tokenization(paragraph))
    counts = []

    for model_name in model_names:
        tokens = models[model_name].encode(paragraph, out_type=str)
        counts.append(len(tokens))

    print(
        f"{i + 1:<8}"
        f"{word_count:<10}"
        f"{counts[0]:<10}"
        f"{counts[1]:<10}"
        f"{counts[2]:<12}"
        f"{counts[3]:<10}"
        f"{counts[4]:<10}"
        f"{counts[5]:<10}"
    )