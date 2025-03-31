import requests
import re
from bs4 import BeautifulSoup
import json
import boto3
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd

def getList(URL):

    headers = {'Referer' : 'https://www.melon.com/index.htm','User-Agent': "Mozilla/5.0"}
    ##html = requests.get(age_url, params= params, headers=headers)
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--disable-dev-shm-usage")
    path='/home/Downloads'
    driver = webdriver.Chrome(chrome_options)
    driver.get(URL)
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    
    titles = []
    artist_names = []
    song_ids = []
    likes = []
    response = []
        
    song = soup.select('.ellipsis.rank01>span>a')
    ##artists = soup.select('.ellipsis.rank02>span>a')
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
    """
    for i in artists:
        artist_names.append(i.text)
    """

    response = [{'songId': song_id, 'title': title} for song_id, title in zip(song_ids, titles)]
        
    return response
    # html1 = requests.get('https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=POP&chartDate=2000#cntt_chart_year')
    # soup = BeautifulSoup(html1.text, 'html.parser')
    # print(html)
    #button_by_class = driver.find_element(By.CLASS_NAME, "btn_view")
    #button_by_class.click()
    #time.sleep(1)

for indexNum in range (1,101,50):
    url_pre = f'https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=456216805#params%5BplylstSeq%5D=456216805&po=pageObj&startIndex={indexNum}'
    print(url_pre)
    payload=getList(url_pre)
    ## song_ids to get lyrics, genre, album datas
    song_ids = [song['songId'] for song in payload]
    print()
## get details (lyrics, genre, album arts)
    details = []
    for i in song_ids:
        details.append(getDetails(i))
        

## Before putitems in DDB, need to join all datas in one json line

    for song, detail in zip(payload, details):
        song.update(detail)

## For playlists, this part will need to modify
    for song in payload:
        song['feelings'] = '즐거운'
        print(payload)

# 데이터 프레임 읽기
df = pd.read.csv('muda_editor_pick')

songid_list = df['songId'].tolist()
title_list = df['title'].tolist()
feelings = df['feelings'].tolist()
editor_list = df['editor_name'].tolist()
editor_pick = df['editor_pick'].tolist()

payload = [{'songId':songId, 'title':title, 'feelings':feelings, 'editor_name':editor_name, 'editor_pick':editor_pick} for songId, title, feelings, editor_name, editor_pick in zip(songid_list, title_list, feelings, editor_list, editor_pick)]


def getDetails(songId):
    url = 'https://www.melon.com/song/detail.htm?songId='+str(songId)
    print(songId)
    
    time.sleep(1)
    req = requests.get(url, headers={'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0"})
    raw = req.text.lower()
    html = raw.replace("<br>", "\n")

    soup = BeautifulSoup(html, "lxml")
    lyric = soup.find("div", {"class": "lyric"})
    
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    albumImages = soup.find("a", {"class": "image_typeAll"})
    genre_dt = soup.find('dt', text='장르')
    genre = genre_dt.find_next_sibling('dd').text
    try:
        artists = soup.select_one("a.artist_name:nth-child(1) > span:nth-child(1)").text
    except AttributeError:
        artists = "Various Artists"
    albumUrl = albumImages.find('img')["src"]
    
    try:
        lyric = lyric.text.strip()
    except AttributeError:
        lyric = "N/A"
    
    payload = {"lyric": lyric, "albumUrl":albumUrl, "genre":genre, "artist":artists}    
    return payload
    
"""    
url_test = "https://www.melon.com/mymusic/dj/mymusicdjplaylistview_inform.htm?plylstSeq=511430503#params%5BplylstSeq%5D=511430503&po=pageObj&startIndex=51"
## Payload is response of song data
payload = getList(url_test)

## song_ids to get lyrics, genre, album datas
song_ids = [song['songId'] for song in payload]
print(song_ids)

## get details (lyrics, genre, album arts)
details = []
for i in song_ids:
    details.append(getDetails(i))


## Before putitems in DDB, need to join all datas in one json line

for song, detail in zip(payload, details):
    song.update(detail)

## For playlists, this part will need to modify
for song in payload:
    song['feelings'] = '즐거운'
print(payload)

"""

details = []
for i in song_ids:
    details.append(getDetails(i))
song = []        
## Before putitems in DDB, need to join all datas in one json line
for song, detail in zip(data, details):
    song.update(detail)


dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')
table = dynamodb.Table('table_name')


for item in payload:
    response = table.put_item(Item=item)
    print(response)
    
