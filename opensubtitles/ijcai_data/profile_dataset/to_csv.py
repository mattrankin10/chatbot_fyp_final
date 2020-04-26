import csv

labels = ['weight']


def create_vector_csv(label, to_vector_csv):
    with open("train_eng.post", "r") as f:
        with open("train.keys", "r") as keys:
            with open(to_vector_csv, "w") as csv_file:
                csv_file.write('text,label' + '\n ')
                lines1 = f.readlines()
                lines2 = keys.readlines()
                i = 0
                while i < len(lines1):
                    if label in lines2[i] and 'positive' in lines2[i]:
                        csv_file.write(lines1[i].replace('\n', '').replace(',', '') + ',' + lines2[i])
                    else:
                        csv_file.write(lines1[i].replace('\n', '').replace(',', '') + ',' + 'negative ' + label + '\n')
                    i += 1


for label in labels:
    create_vector_csv(label, label + '.csv')