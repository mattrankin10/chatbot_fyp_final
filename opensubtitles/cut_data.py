def clean_file(fileName):
    with open(fileName, 'a', encoding='utf8') as f:
        f.truncate(0)


def cut_training_data(file, new_file):
    with open(file, 'r') as f:
        lines = f.readlines()[0:117859]
        with open(new_file, 'w') as new_file:
            for line in lines:
                new_file.write(line)
    f.close()


clean_file("train117k.from")
clean_file("train117k.to")
cut_training_data('train.from', 'train117k.from')
cut_training_data('train.to', 'train117k.to')
