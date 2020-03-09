import json

qs = []
ans = []

with open("result.json") as file:
    new_intents = json.load(file)
    for tag in new_intents[0]:
        for patterns in tag[0]:
            qs.extend(patterns)
        for response in tag[0]:
            ans.extend(response)
    #question = new_intents['patterns']
    #answer = new_intents['responses']
    #qs.extend(question)
    #ans.extend(answer)
print(qs[0])
# with open('outcome.json', 'w') as fp:
#    json.dump(new_intents, fp)
