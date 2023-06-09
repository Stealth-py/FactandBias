import requests
from bs4 import BeautifulSoup
import json

with open('mbfc_links.txt', 'r') as f:
    data = f.read().replace("'", '"')
    data = json.loads(data)


mapping = {}
# perform get request to the url
for URL in data['https://mediabiasfactcheck.com/left/']:
    reqs = requests.get(URL)
    mapping[URL] = []

    # extract all the text that you received from
    # the GET request
    content = reqs.text

    # convert the text to a beautiful soup object
    soup = BeautifulSoup(content, 'html.parser')

    # Empty list to store the output
    urls = []

    # For loop that iterates over all the <li> tags
    for h in soup.find_all(lambda t: t.name == "p" and t.text.startswith("Source:")):

        # looking for anchor tag inside the <li>tag
        a = h.find('a')
        try:
            
            # looking for href inside anchor tag
            if 'href' in a.attrs:
                
                # storing the value of href in a separate variable
                url = a.get('href')
                
                # appending the url to the output list
                urls.append(url)
                
        except:
            pass

    for url in urls:
        mapping[URL].append(url)
with open("lst.txt", 'w') as f:
    f.write(str(mapping))
