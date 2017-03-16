import re
import os


def uncomment(fileName, content):
    if fileName.endswith('.c'):
        return re.sub(r'/\*.*?\*/', '', content,
                      flags=re.MULTILINE | re.DOTALL)
    elif fileName.endswith(('.cpp', '.cc', '.h', '.hh', '.hxx', 'hpp')):
        pass1 = re.sub(r'/\*.*?\*/', '', content,
                       flags=re.MULTILINE | re.DOTALL)
        pass2 = re.sub(r'//.*', '', pass1)
        return pass2
    elif fileName.endswith('.py'):
        pass1 = re.sub(r'#.*', '', content)
        pass2 = re.sub(r'""".*?"""', '', pass1,
                       flags=re.MULTILINE | re.DOTALL)
        return pass2
    return content


def search_files(data_dirs, suffixes):
    matches = []  # list of files to read
    for data_dir in data_dirs:
        for root, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(tuple(suffixes)):
                    matches.append(os.path.join(root, filename))
    return matches


def tokenize(fileName, retcontent=False):
    allTokens = []
    with open(fileName) as data:
        content = uncomment(fileName, data.read())
        for line in content.split('\n'):
            allTokens += [token.strip()
                          for token in re.split(r'(\W+)', line)
                          if len(token.strip()) > 0]
    if not retcontent:
        return allTokens
    else:
        return allTokens, content
