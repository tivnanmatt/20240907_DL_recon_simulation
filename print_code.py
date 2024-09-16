

# print all filenames ending in .py and print their contents

import os

def print_code():
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                print('----------------------------')
                print(root + '/' + file)
                with open(root + '/' + file, 'r') as f:
                    print(f.read())
                print('----------------------------')

if __name__ == '__main__':
    print_code()
