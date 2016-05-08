# You will be amazed how easy this is!
if __name__ == '__main__':
    string = 'Hello World from a text file'
    with open('HelloWorld.txt', 'w') as text_file:
        text_file.write(string)
        print('Done!')
