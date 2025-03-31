import requests
import re
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

import boto3

dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')
table = dynamodb.Table('table_name')


def getDetails(songId):
    print(songId)
    url = 'https://www.melon.com/song/detail.htm?songId='+str(songId)
    req = requests.get(url, headers = {'User-Agent': "Mozilla/5.0"})
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    song_name = soup.find('div', class_='song_name')
    try:
        for strong_tag in song_name.find_all('strong'):
            strong_tag.decompose()
            extracted_name = song_name.text.strip()
    except AttributeError: 
        extracted_name=songId
    
    date_dt = soup.find('dt', text='발매일')
    try:
        date = date_dt.find_next_sibling('dd').text
    except AttributeError:
        date = None
    artist = soup.find("a", {"class": "artist_name"})
    
    try:
        artists = artist.text.strip()
    except AttributeError:
        artists = None
    
    albumImages = soup.find("a", {"class": "image_typeAll"})
    genre_dt = soup.find('dt', text='장르')
    
    try:
        genre = genre_dt.find_next_sibling('dd').text
    except AttributeError:
        genre = None
        
    try:
        albumUrl = albumImages.find('img')["src"]
    except AttributeError:
        albumUrl = None
    raw = req.text.lower()
    html = raw.replace("<br>", "\n")
    soup = BeautifulSoup(html, "lxml")
    
    lyric = soup.find("div", {"class": "lyric"})
    try:
        lyrics=lyric.text.strip()
    except AttributeError:
        lyrics= None
    payload = {"name":extracted_name, "ranking":"000","date":date, "artists":artists,"albumUrl":albumUrl, "genre":genre,"lyric":lyrics }
    return json.dumps(payload,ensure_ascii=False)
   
def age_songid(age):
    age_url = 'https://www.melon.com/chart/age/index.htm'
    ## https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=POP&chartDate=2000
    ## https://www.melon.com/chart/age/index.htm?chartType=AG&chartGenre=KPOP&chartDate=2010
    
    params = {
        'idx' : 2,
        'chartType' : 'YE',
        'chartGenre' : 'KPOP',
        'chartDate':age,
        'moved':'Y'
    }
    
    headers = {'Referer' : 'https://www.melon.com/index.htm','User-Agent': "Mozilla/5.0"}
    ##html = requests.get(age_url, params= params, headers=headers)
    
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--disable-dev-shm-usage")
    path='/home/Downloads'
    driver = webdriver.Chrome(chrome_options)
    url_age = f"https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenreKPOP&chartDate={age}"
    driver.get(url_age)
    html = driver.page_source
    # html1 = requests.get('https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=POP&chartDate=2000#cntt_chart_year')
    # soup = BeautifulSoup(html1.text, 'html.parser')
    # print(html)
    button_by_class = driver.find_element(By.CLASS_NAME, "btn_view")
    button_by_class.click()
    time.sleep(1)
    
    song_num =[]

    lst50 = driver.find_elements(By.CSS_SELECTOR, "#lst50 > td:nth-child(5) > div > button")

    for i in lst50:
        song_num.append(i.get_attribute('data-song-no'))

    ###51-100 xpath : /html/body/div/div[3]/div/div/div[4]/div/div[2]/div[2]/span/a
    ### span.page_num:nth-child(1) > a:nth-child(2)
    ### span.page_num:nth-child(1) > a:nth-child(1)
    
    ### span.page_num:nth-child(1) > a:nth-child(2)

    """
    driver.find_element(By.XPATH, "/html/body/div/div[3]/div/div/div[4]/div/div[2]/div[2]/span").click()
    time.sleep(1)
    
    lst100 = driver.find_elements(By.CSS_SELECTOR, "#lst100 > td:nth-child(5) > div > button")


    for i in lst100:
        song_num.append(i.get_attribute('data-song-no'))
    """
    
    print(song_num)
    test=[]
    
    for i in song_num:
        data = getDetails(i)
        data_list = json.loads(data)
        response=table.put_item(Item=data_list)

    """
    for item in temp:
        response = table.put_item(Item=item)
        print(response)
    """
    
for i in range(2000,2024):
    age_songid(i)



