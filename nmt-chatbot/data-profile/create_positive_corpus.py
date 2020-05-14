filenames = ['name', 'age', 'location', 'gender', 'constellation']

def create(filenames, file_type):
    with open('positive' + file_type, 'w') as outfile:
        for fname in filenames:
            with open(fname + file_type) as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == "__main__":
    create(filenames, '_tst.from')
    create(filenames, '_tst.to')

