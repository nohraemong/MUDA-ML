import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
import pandas as pd
import json

def filtering(userId):
    # recommended table
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('table_name')

    # get userId from payload
    # userId = json_data.get('userId')

    ## for test - dummy user name

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    print(end_date, start_date)

    # parsing date time as sort key includes
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S')
    print(start_date_str, end_date_str)

    try:
        response = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key('userId').eq(userId) & 
        boto3.dynamodb.conditions.Key('date').between(start_date_str, end_date_str)
    )
        
        song_ids = [item['songIds'] for item in response['Items'] if 'songIds' in item]
        filtering_songids = []

        for item in song_ids:
            for songid in item:
                    filtering_songids.append(songid)

        return filtering_songids


    except Exception as e:
        print(f"{e}")

def insert_recommended_songId(userId, songIds):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('musicdiary_recommended_dummy')

    date = datetime.now()

    # parsing date time as sort key includes
    date_parse = date.strftime('%Y-%m-%dT%H:%M:%S')
    print(date_parse)

    try:
        response = table.put_item(
            Item={
                'userId': userId, 
                'songIds': set(songIds),
                'date': date_parse
            }
        )
        print("songIds saved:", response)
    except Exception as e:
        print("Error log:", e)

