import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
import pandas as pd
import json
from muda_filter_test import filtering, insert_recommended_songId

## User Id dummy로 넣어줌
userId = 'honeybee'
print(filtering(userId))

test_df  = pd.read_csv('/Users/cayley/downloads/scalediary_db.csv')
test_df['songId'] = test_df['songId'].astype(str)
## filtering 시작합니당
print('user 확인 및 추천받은 곡 조회')
pre_filter = filtering(userId)
# pre_filter = list(map(int, pre_filter))

print(type(pre_filter))

## 리스트가 비어있는지 확인한 뒤 lyrics_df에 적용하기
if pre_filter:
    print(len(test_df))
    test_df = test_df[~test_df['songId'].isin(pre_filter)]

else:
    print('pass')
    
## dataframe 
print(len(test_df))
print(test_df)

## 테스트하기
save_songId = test_df['songId'].head(3)
print(save_songId)

# insert_recommended_songId(userId, save_songId)
