import Levenshtein

name_q = []
q = input("Type: ")
with open("name_positive.from", "r") as file:
    lines = file.readlines()
    for line in lines:
        name_q.append(line)


ratios = []
for s in enumerate(name_q):
    ratio = Levenshtein.ratio(q, s)
    ratios.append({
        "question": s,
        "ratio": float(ratio)
    })
best_ratio = {
    "question": '',
    "ratio": 0
}
for prediction in ratios:
    if  prediction["ratio"] > best_ratio["ratio"]:
        best_ratio["question"] = prediction["question"]
        best_ratio["ratio"] = prediction["ratio"]

print('Question is similar to: ' + best_ratio["question"] + ', with ratio: ' + str(best_ratio["ratio"]))


