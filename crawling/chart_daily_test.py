import requests
import re
from bs4 import BeautifulSoup
import json
import boto3
import time
import pandas as pd

def getList(URL):

    html = requests.get(URL, headers={
                        'User-Agent': "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "lxml")
    
    titles = []
    artist_names = []
    song_ids = []
    likes = []
    response = []
    
    song = soup.select('.ellipsis.rank01>span>a')
    artists = soup.select('.ellipsis.rank02>span>a')
    ##a_tag = soup.find_all('a', href=lambda href: href and 'goSongDetail' in href)
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if "melon.link.goSongDetail" in href:
            start = href.find("'") + 1
            end = href.find("'", start)
            song_id = href[start:end]
            song_ids.append(song_id)
            
    for tag in song:
        titles.append(tag.text)
    print(titles)

    for i in artists:
        artist_names.append(i.text)
    
    response = [{'songId': song_id, 'title': title, 'artist': artist} for song_id, title, artist in zip(song_ids, titles, artist_names)]
        
    return response


def getDetails(songId):
    url = 'https://www.melon.com/song/detail.htm?songId='+str(songId)
    print(songId)
    req = requests.get(url, headers={'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0"})
    raw = req.text.lower()
    html = raw.replace("<br>", "\n")
    soup = BeautifulSoup(html, "lxml")
    lyric = soup.find("div", {"class": "lyric"})
    
    time.sleep(1)
    req = requests.get(url, headers = {'User-Agent': "Mozilla/5.0"})
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    albumImages = soup.find("a", {"class": "image_typeAll"})
    genre_dt = soup.find('dt', text='장르')
    genre = genre_dt.find_next_sibling('dd').text
    albumUrl = albumImages.find('img')["src"]
    payload = {"lyric": lyric.text.strip(), "albumUrl":albumUrl, "genre":genre}    
    return payload
    
url_test = "https://www.melon.com/chart/day/index.htm"
## Payload is response of song data
try:
    payload = getList(url_test)
except Exception as e:
    print(f"에러 발생 : {e}")
    sys.exit(1)    

## song_ids to get lyrics, genre, album datas
song_ids = [song['songId'] for song in payload]
print(song_ids)

## get details (lyrics, genre, album arts)
details = []
try:
    for i in song_ids:
        details.append(getDetails(i))
except Exception as e:
    print(f"에러 발생 : {e}")
    sys.exit(1)
## Before putitems in DDB, need to join all datas in one json line

for song, detail in zip(payload, details):
    song.update(detail)
    #print(song)

df = pd.DataFrame(song)
print(df)
current_time = time.strftime("%Y%m%d_%H%M%S")
file_name = f"data_daily_{current_time}.parquet"
df.to_parquet(file_name)
