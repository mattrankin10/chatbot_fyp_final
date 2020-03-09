import json
from glob import glob

for file_name in glob("video_games/*.json"):
    print(file_name)
    for line in open(file_name):
        example = json.loads(line)
        print(example['response'])
