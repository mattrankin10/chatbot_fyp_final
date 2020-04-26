import csv

new_labels = ['hobby', 'idol', 'speciality', 'employer']


def create_vector_csv(new_label, new_file):
    with open(new_label + ".csv", "r") as file:
        with open(new_file, "w") as f:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                f.write(lines[i].replace('positive name', 'negative ' + new_label)
                        .replace('negative name', 'negative ' + new_label))
                i += 1


for label in new_labels:
    create_vector_csv(label, label + '_new.csv')