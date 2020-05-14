def clean_file(fileName):
    with open(fileName, 'a', encoding='utf8') as f:
        f.truncate(0)


def cut_training_data(file, new_file):
    with open(file, 'r') as f:
        lines = f.readlines()[0:28500]
        with open(new_file, 'w') as new_file:
            for line in lines:
                new_file.write(line)
    f.close()


clean_file("tst117kqs.from")
clean_file("tst117kqs.to")
cut_training_data('testquestions.from', 'tst117kqs.from')
cut_training_data('testquestions.to', 'tst117kqs.to')
