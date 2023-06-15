from bs4 import BeautifulSoup
import ast
import json
import os
import re
regex = re.compile(r'<[^>]+>')


def remove_html(string):
    return regex.sub('', string)


if __name__ == "__main__":
    files = [i for i in os.listdir() if '.txt' in i]
    mapping = {}

    for file in files:
        print("Processing:", file)
        with open(file, 'r') as f:
            whole_list = f.read().replace("'https://mediabiasfactcheck.com/skeptiko/': None,",
                                          '').replace('</p>',
                                    "</p>'").replace('<p>',
                                                     "'<p>").split("</p>', ")
            for source in whole_list:
                source = source.split("': '")
                if len(source) == 2:
                    site, labels = source
                    site = site.replace('{', '')[1:]
                    print('\tParsing:', site)
                    soup = BeautifulSoup(labels, 'html.parser')
                    dct = {}
                    txt = soup.text.split('\n')
                    for t in txt:
                        t = t.split(':')
                        if len(t) != 2:
                            continue
                        dct[t[0].strip()] = t[1].strip()
                    mapping[site] = dct


    print(mapping)
    with open('labels_collected.json', 'w') as f:
        json.dump(mapping, f)