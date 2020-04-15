import csv

labels = ['name', 'gender', 'age', 'location', 'constellation']


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
                    if label in lines2[i] and 'negative' in lines2[i]:
                        csv_file.write(lines1[i].replace('\n', '').replace(',', '') + ',' + lines2[i])
                    i += 1


create_vector_csv('name', 'name_only.csv')