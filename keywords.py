import os
import argparse
from collections import Counter
from normalize import search_files, tokenize


def extract_keywords(data_dirs, suffixes, max_keywords):
    """
    Returns a dictionary of min(max_keywords, percentile_keywords)
    Giving keyword with its count.
    """
    matches = search_files(data_dirs, suffixes)
    print("found %s file matches. " % len(matches))

    token_count = Counter()
    files_done = 0
    for file_name in matches:
        tokens = tokenize(file_name)
        for token in tokens:
            if len(token) == 0:
                continue
            try:
                token_count[token] += 1
            except:
                token_count[token] = 1
        files_done += 1
        if (files_done % 5000 == 0):
            print("Completed parsing %d files ..." % files_done)
    return token_count.most_common(max_keywords)


def parse_arguments():
    def listparse(arg):
        return arg.split(',')
    p = argparse.ArgumentParser()
    p.add_argument('-p', '--project', default='django',
                   help='project name')
    p.add_argument('-s', '--suffixes', type=listparse, default=['py'],
                   help='languages to use as a comma separated list')
    p.add_argument('-n', '--num', type=int, default=2000,
                   help='Number of keywords to extract')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    root_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(root_dir, 'data', args.project)
    keyword_dir = os.path.join(root_dir, 'keywords')

    if not os.path.exists(data_dir):
        raise ValueError('Data directory %s doesn\'t exist' % data_dir)

    if not os.path.exists(keyword_dir):
        os.makedirs(keyword_dir)

    keyword_file = os.path.join(keyword_dir, args.project)

    # Extract keywords and save them
    print("Extracting keywords for project %s ..." % args.project)
    words = extract_keywords(data_dirs=[data_dir], suffixes=args.suffixes,
                             max_keywords=args.num)
    print("%s words extracted from %s." % (len(words), args.project))
    with open(keyword_file, 'w') as out:
        for keyword, count in words:
            out.write("%s %d\n" % (keyword, count))
    print("Completed extracting keywords for project %s" % args.project)
