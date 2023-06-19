import requests
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

with open('mbfc_links.txt', 'r') as f:
    data = f.read().replace("'", '"')
    data = json.loads(data)

mapping = {}

# perform get request to the url
for URLs in data.values():
    for URL in tqdm(URLs):
        reqs = requests.get(URL)
        mapping[URL] = []

    # extract all the text that you received from
    # the GET request
        content = reqs.text

        # convert the text to a beautiful soup object
        soup = BeautifulSoup(content, 'html.parser')

        # Empty list to store the output
        text = []

        # For loop that iterates over all the <h2> tags
        for h2 in soup.findAll('h2'):
            
            # looking for anchor tag inside the <h2>tag
            img = h2.find('img')
            try:
                
                # looking for href inside img tag
                if 'alt' in img.attrs:
                    
                    # storing the value of alt in a separate variable
                    alt = img.get('alt')
                    
                    # appending the text to the output list
                    text.append(alt)
                    
            except:
                pass
        
        for alt in text:
            mapping[URL].append(alt)

with open('mbfc_alt_text.json', 'w') as json_file:
    json.dump(mapping, json_file)