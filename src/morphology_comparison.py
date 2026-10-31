from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pymorphy2
import stanza
import time

with open(str(ROOT / "data" / "Dubrovsky.txt"), "r", encoding="utf-8") as file:
    text = file.read()

nlp = stanza.Pipeline("ru", processors="tokenize,pos,lemma", download_method=None, verbose=False)

start_time = time.time()
doc = nlp(text)
stanza_time = time.time() - start_time

words = []
stanza_lemmas = []

for sentence in doc.sentences:
    for word in sentence.words:
        if word.upos != "PUNCT":
            words.append(word.text)
            stanza_lemmas.append(word.lemma.lower())


morph = pymorphy2.MorphAnalyzer()

pymorphy_lemmas = []
start_time = time.time()

for word in words:
    result = morph.parse(word.lower())[0]
    pymorphy_lemmas.append(result.normal_form)

pymorphy_time = time.time() - start_time

pymorphy_counts = {}
for lemma in pymorphy_lemmas:
    if lemma in pymorphy_counts:
        pymorphy_counts[lemma] += 1
    else:
        pymorphy_counts[lemma] = 1

pymorphy_sorted = list(pymorphy_counts.items())
pymorphy_sorted.sort(key=lambda item: item[1], reverse=True)

stanza_counts = {}
for lemma in stanza_lemmas:
    if lemma in stanza_counts:
        stanza_counts[lemma] += 1
    else:
        stanza_counts[lemma] = 1

stanza_sorted = list(stanza_counts.items())
stanza_sorted.sort(key=lambda item: item[1], reverse=True)

print("\nВремя работы\n")
print("PyMorphy2:", round(pymorphy_time, 3), "сек.")
print("Stanza:", round(stanza_time, 3), "сек.")

print("\nЧастотный список PyMorphy2\n")
print(f"{'Ранг':<8}" f"{'Лемма':<25}" f"{'Частота':<10}")

for i in range(min(30, len(pymorphy_sorted))):
    lemma = pymorphy_sorted[i][0]
    frequency = pymorphy_sorted[i][1]
    print(f"{i + 1:<8}" f"{lemma:<25}" f"{frequency:<10}")

print("\nЧастотный список Stanza\n")
print(f"{'Ранг':<8}" f"{'Лемма':<25}" f"{'Частота':<10}")

for i in range(min(30, len(stanza_sorted))):
    lemma = stanza_sorted[i][0]
    frequency = stanza_sorted[i][1]
    print(f"{i + 1:<8}" f"{lemma:<25}" f"{frequency:<10}")

different_lemmas = []
for i in range(len(words)):
    if pymorphy_lemmas[i] != stanza_lemmas[i]:
        different_lemmas.append((words[i], pymorphy_lemmas[i], stanza_lemmas[i]))

print("\nВсего слов:", len(words))
print("Количество различий:", len(different_lemmas))
difference_percent = (len(different_lemmas) / len(words) * 100)
print("Доля различий:", round(difference_percent, 2), "%")

print("\nПримеры различий\n")
print(f"{'Словоформа':<25}" f"{'PyMorphy2':<25}" f"{'Stanza':<25}")

for i in range(min(50, len(different_lemmas))):
    word = different_lemmas[i][0]
    pymorphy_lemma = different_lemmas[i][1]
    stanza_lemma = different_lemmas[i][2]
    print(f"{word:<25}" f"{pymorphy_lemma:<25}" f"{stanza_lemma:<25}")