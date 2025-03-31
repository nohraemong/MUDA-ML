from pytube import YouTube
from moviepy.editor import *
import os
import boto3
import requests
import re
from bs4 import BeautifulSoup
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import numpy as np
import librosa
import pandas as pd


dynamodb = boto3.client('dynamodb', region_name='ap-northeast-2')
NextToken = None
query_cnt = 0
max_query_count = 1


search_lists_artist = []
search_lists_song = []
feeling_list = []

while True:
    params = {'Statement' : "SELECT title, artist, feelings FROM {table_name}}", }

    if NextToken:
        params['NextToken'] = NextToken
        
    resp = dynamodb.execute_statement(**params)

    text = resp['Items']
    
    print(len(text))
    
    if query_cnt == 0:
        for i in range(0,len(text)):
            artist = text[i]['artist']['S']
            song_name = text[i]['title']['S']
            feelings = text[i]['feelings']['S']
            
            search_lists_artist.append(artist)
            search_lists_song.append(song_name)
            feeling_list.append(feelings)
    """
    query_cnt +=1
    
    NextToken = resp.get('NextToken')
    
    if query_cnt > max_query_count or not NextToken:
        break
    """
    break

print(search_lists_song)
print(len(feeling_list))
print(len(search_lists_song))


musicList = []
videoList = []
artistList= []
feelingList=[]


for i in range(0,len(search_lists_song)):  
    try:  # 전체 크롤링 프로세스를 try 블록 안에 넣습니다.
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument("--disable-dev-shm-usage")
        path = '/home/Downloads'
        driver = webdriver.Chrome(options=chrome_options)

        driver.get("https://www.youtube.com/")

        # 검색창 찾기
        search_box = driver.find_element(By.NAME, "search_query")

        time.sleep(3)
        # 검색어 입력
        search_query = search_lists_artist[i] + " " + search_lists_song[i]
        print(search_query)
        search_box.send_keys(search_query)
        time.sleep(3)
        # Enter 키 누르기
        search_box.send_keys(Keys.RETURN)
        time.sleep(3)
        first_video = driver.find_element(By.ID, "video-title")

        # 비디오 링크에서 video ID 추출
        video_url = first_video.get_attribute("href")
        time.sleep(3)
        try:
            video_id = video_url.split("v=")[1]
            result = video_id.split('&')[0]
            videoList.append(result)
            artistList.append(search_lists_artist[i])
            feelingList.append(feeling_list[i])
            musicList.append(search_lists_song[i])

            print("Video ID:", result)
            driver.quit()
            
        except Exception as e:
            video_id='N/A'
            videoList.append(video_id)
            print("no video ID.", e)
            audio_filename =search_lists_song[i]
            musicList.append(audio_filename)
            artistList.append(search_lists_artist[i])
            feelingList.append(feeling_list[i])
            driver.quit()
            continue

        
        # 유튜브 비디오의 URL
        url = f'https://www.youtube.com/watch?v={result}'
        
        # YouTube 객체 생성
        yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        download_path = '/home/ec2-user/environment/files_omitted'
        # 비디오에서 오디오 스트림 가져오기
    
        time.sleep(3)
        audio_stream = yt.streams.filter(only_audio=True).first()

        time.sleep(3)
        # 오디오 다운로드 경로 설정
        os.makedirs(download_path, exist_ok=True)

        # 원본 오디오 파일명
        original_audio_filename = audio_stream.default_filename

        # MP3 파일명
        audio_filename = f'{yt.title}.mp3'
        if '/' in audio_filename:
            audio_filename = audio_filename.replace('/', '')

        musicList[i]=audio_filename
        """
        audio_path = download_path + "/" + audio_filename
        # 오디오 다운로드
        time.sleep(3)
        
        audio_stream.download(output_path=download_path, filename=original_audio_filename)
        
        # 원본 오디오 파일 경로
        original_audio_filepath = os.path.join(download_path, original_audio_filename)

        # MP3 파일 경로
        audio_filepath = os.path.join(download_path, audio_filename)

        # 원본 오디오를 MP3로 변환
        audio_clip = AudioFileClip(original_audio_filepath)

        audio_clip.write_audiofile(audio_filepath)

        os.remove(original_audio_filepath)
        """
    except Exception as e:  # 예외가 발생하면 여기서 잡습니다.
        print(f"Error processing video {i}: {e}")
        
        # 실패한 경우에도 웹 드라이버를 닫아 자원을 해제합니다.
        try:
            driver.quit()
        except:
            pass

data = {'title': musicList, 'artists': search_lists_artist, 'feelings': feeling_list, 'video_id': videoList}
df = pd.DataFrame(data)
print(df)

df.to_csv('local_path', index=False)
