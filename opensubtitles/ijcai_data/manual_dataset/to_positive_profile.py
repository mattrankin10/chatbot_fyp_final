labels = ['name', 'gender', 'age', 'location', 'constellation', 'weight']


def create_profile_data(label, from_file):
    with open("eng.post", "r") as f:
            with open("test.keys", "r") as keys:
                with open(from_file, "w") as from_f:
                    from_lines = f.readlines()
                    key_line = keys.readlines()
                    i = 0
                    while i < len(from_lines):
                        if label in key_line[i] and 'positive' in key_line[i]:
                            from_f.write(from_lines[i])
                        i += 1
    f.close()
    keys.close()
    from_f.close()


def create_profile_test_data(label, from_file, to_file):
    with open("test_eng.post", "r") as f:
        with open('test_eng.resp', 'r')  as t:
            with open("test.keys", "r") as keys:
                with open(from_file, "w") as from_f:
                    with open(to_file, 'w') as  to_f:
                        from_lines = f.readlines()
                        to_lines = t.readlines()
                        key_line = keys.readlines()
                        i = 0
                        while i < len(from_lines):
                            if label in key_line[i] and 'positive' in key_line[i]:
                                from_f.write(from_lines[i])
                                to_f.write(to_lines[i])
                            i += 1
    f.close()
    t.close()
    keys.close()
    from_f.close()
    to_f.close()



for label in labels:
    #create_profile_data(label, label + '.from', label + '.to')
    create_profile_data(label, label + '_positive.from', )
