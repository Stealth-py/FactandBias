import requests
from bs4 import BeautifulSoup

left = "https://mediabiasfactcheck.com/left/"
left_center= "https://mediabiasfactcheck.com/leftcenter/"
right_center = "https://mediabiasfactcheck.com/right-center/"
right = 'https://mediabiasfactcheck.com/right/'
conspiracy = 'https://mediabiasfactcheck.com/conspiracy/'
fake = 'https://mediabiasfactcheck.com/fake-news/'
science = 'https://mediabiasfactcheck.com/pro-science/'
satire = 'https://mediabiasfactcheck.com/satire/'

mapping = {}
# perform get request to the url
for URL in [left, left_center, right, right_center, conspiracy, fake, science, satire]:
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
    for h in soup.findAll('td'):
        
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
