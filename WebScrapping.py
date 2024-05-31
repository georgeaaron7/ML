import requests
from bs4 import BeautifulSoup

# input URL:
wikipedia_url = str(input("Enter the URL of the wikipedia page: "))
page = requests.get(wikipedia_url)
soup = BeautifulSoup(page.text, 'html.parser')

# find the title of the page using the soup find function
title = soup.find('h1', {'id': 'firstHeading'}).text

# finding and storing all the external links in a list
external_links = []
for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('http'):
            external_links.append(href)

#  storing the urls and alternative captions to all the images in a list
images = []
for img in soup.find_all('img'):
        img_url = img['src']
        if img_url.startswith('//'):
            img_url = 'https:' + img_url
        alt_text = img.get('alt', 'N/A')
        images.append({'url': img_url, 'alt': alt_text})

# printing the results
print("\nTitle:",title)
print("\nExternal links :" , *external_links, sep='\n')
print("\nImages:")
for img in images:
     print(f"URL: {img['url']}, Alt: {img['alt']}")
