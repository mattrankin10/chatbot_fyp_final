import os

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import sys

sys.path.insert(0, os.getcwd())
from setup.settings import hparams, preprocessing
import math


with open(hparams['out_dir'] + 'checkpoint', 'r') as checkpoint_file:
    start = '-'
    end = '"'
    ckpt = checkpoint_file.readline()
    current_step = int((ckpt.split(start))[1].split(end)[0])

with open('{}/corpus_size'.format(preprocessing['train_folder']), 'r') as f:
    corpus_size = int(f.read())

try:
    with open('{}epochs_passed'.format(hparams['out_dir']), 'r') as f:
        epoch = int(f.read())
        steps = math.ceil(epoch * corpus_size / (hparams['batch_size'] if 'batch_size' in hparams else 128))

except:
    steps = math.floor(current_step / 5000) * 5000

output_file = 'output_dev_{}'.format(str(steps))

# Quick file to pair epoch outputs w/ original test filsle
if __name__ == '__main__':
    with open(os.path.join(hparams['out_dir'], output_file), 'r', encoding='utf-8') as f:
        with open(os.path.join('utils/model-response.txt'), 'w') as answers:
            print('Output from file: ' + os.path.join(hparams['out_dir'], output_file))
            content = f.read()
            to_data = content.split('\n')
            for a in to_data:
                answers.write(str(a) + '\n')

    with open(hparams['dev_prefix'] + '.' + hparams['src'], 'r', encoding='utf-8') as f:
        content = f.read()
        from_data = content.split('\n')
        if preprocessing['use_bpe']:
            from_data = [answer.replace(' ', '').replace('▁', ' ') for answer in from_data]

    for n, _ in enumerate(to_data[:-1]):
        print(30 * '_')
        print('>', from_data[n])
        print('Reply:', to_data[n])
