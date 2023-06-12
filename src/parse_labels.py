from bs4 import BeautifulSoup
import json
import requests
from tqdm import tqdm

if __name__ == "__main__":
    with open('mbfc_links.txt', 'r') as f:
        content = f.read()
        content = content.replace("'", '"')
    articles = json.loads(content)
    print(articles)
    for branch in tqdm(list(articles.keys())[4:]):
        evaluation = {}
        print(branch)
        for url in tqdm(articles[branch]):
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, 'html.parser')
                div = soup.find('h3')
                #print(div)
                evaluation[url] = div.find_next('p')
            except:
                print(url)
        with open(f'{branch.split("https://mediabiasfactcheck.com/")[1].replace("/", "")}.txt', 'w') as f:
            f.write(str(evaluation))
