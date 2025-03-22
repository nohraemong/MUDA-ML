import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
import ast
import pickle
# from sagemaker import get_execution_role
import json
from newspaper import Article
from konlpy.tag import Kkma, Okt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import jpype
import re
import random
import boto3
from boto3.dynamodb.conditions import Key
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response
from muda_prefilter import filtering, insert_recommended_songId

app = Flask(__name__)


tqdm.pandas()


# 모델 로드

model = DistilBertModel.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1')
tokenizer = DistilBertTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v1')

@app.route('/ping', methods=['GET'])
def ping():
  return '', 200

# 데이터 로드
with open('/opt/program/text_list.pkl', 'rb') as f:
    text_list = pickle.load(f)

text_list['editor_pick'] = ['Y' if i in random.sample(range(len(text_list)), max(1, len(text_list) * 3 // 100)) else 'N' for i in range(len(text_list))]

feelings_df  = pd.read_csv('/opt/program/diary_feelings.csv', index_col=0)
text_list['feelings'] = text_list['feelings'].str.split('(').str[0]
lyrics_df = text_list.merge(feelings_df, left_on='feelings', right_on='feelings', how='left')


# SentenceTokenizer Class (최적화 1 - 수정)
class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        # self.okt = Okt()
        # self.stopwords_ko = ['중인', '만큼', '마찬가지', '꼬집었', "연합뉴스", "데일리", "동아일보", "중앙일보", "조선일보", "기자",
        #                      "아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가"]
        # self.stopwords_en = set(stopwords.words('english'))
        
    def url2sentences(self, url):
        article = Article(url, language='ko')
        article.download()
        article.parse()
        
        # if self._is_korean(article.text):
        sentences = self.kkma.sentences(article.text)

        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = ''

        return [sentence for sentence in sentences if sentence]
        
    
    def text2sentences(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|\\n|\n', text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences


    def get_nouns(self, sentences):
        nouns = []
        if sentences and self._is_korean(sentences[0]):
            for sentence in sentences:
                if sentence != '':
                    nouns.append(' '.join([noun for noun in self.kkma.nouns(str(sentence))]))
        else:
            for sentence in sentences:
                if sentence:
                    words = word_tokenize(sentence)
                    nouns.append(' '.join([word for word in words if word.isalpha() and word.lower() not in nltk.pos_tag([word])[0][1] in ['NN', 'NNS', 'NNP', 'NNPS']]))
        # print("추출된 명사: ", nouns)  # 디버그 출력
        return nouns

    def _is_korean(self, text):
        # 간단한 휴리스틱을 사용하여 텍스트가 한국어인지 감지
        return any('\uac00' <= char <= '\ud7a3' for char in text)
# GraphMatrix Class
class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []

    def build_sent_graph(self, sentence):
        if not sentence:
            raise ValueError("Empty sentence list")
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence

    def build_words_graph(self, sentence):
        if not sentence:
            raise ValueError("Empty sentence list")
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]: word for word in vocab}

# Rank Class
class Rank(object):
    def get_ranks(self, graph, d=0.85): # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 # diagonal 부분을 0으로
            link_sum = np.sum(A[:, id]) # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1

        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}

class TextRank:
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()

        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)

        self.nouns = self.sent_tokenize.get_nouns(self.sentences)

        # Skip processing if no valid nouns extracted
        if not any(self.nouns):
            self.sent_graph = None
            self.words_graph = None
            self.idx2word = {}
            return

        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)

        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)

    def summarize(self, sent_num=3):
        summary = []
        index = []
        seen_sentences = set()

        for idx in self.sorted_sent_rank_idx:
            if self.sentences[idx] not in seen_sentences:
                seen_sentences.add(self.sentences[idx])
                index.append(idx)
            if len(index) >= sent_num:
                break

        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])

        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []
        index = []
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        for idx in index:
            keywords.append(self.idx2word[idx])

        return keywords

# 수정 7.14 오전 3:00
def get_embeddings(text):
    try :
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings
    except :
        return np.zeros((3, 768))  # 임베딩 벡터의 크기에 맞게 0으로 채운 벡터 반환


### 키워드 적재
# 사용자 입력 받기 및 결과 출력
def keyword_extraction(input_text):
    try:
        textrank = TextRank(input_text)
        # return textrank.keywords(word_num=10)
        return textrank.summarize(3)
    except Exception as e:
        print(f"Error processing text: {e}")
        return []

# 데이터프레임에 적용하기
def apply_keyword_extraction(df, text_column):
    df['keywords'] = df[text_column].progress_apply(keyword_extraction)
    return df

# 수정 7.14 오전 3:00
def parse_embedding(embedding):
    # 이미 numpy 배열일 경우 그대로 반환
    if isinstance(embedding, np.ndarray) and embedding.shape == (3, 768):
        return embedding
        ## return하는 부분 numpy array 그대로 안나오게 수정
    else:
        try:
            # 1차 시도: ast.literal_eval로 문자열을 리스트로 변환
            return np.array(ast.literal_eval(embedding))
        except (ValueError, SyntaxError):
            # 2차 시도: 수동으로 문자열을 파싱하여 리스트로 변환
            embedding = embedding.replace('\n', '').replace('[', '').replace(']', '').strip()
            return np.array([float(x) for x in embedding.split()])

# 수정 7.14 오전 3:00
lyrics_embeddings = text_list['embeddings'].progress_apply(parse_embedding)

@app.route('/invocations', methods=['POST'])
def recommend_song():
    tqdm.pandas()

    #수정 - 장르, 감정 받기
    payload = request.get_data().decode('utf-8')
    json_data = json.loads(payload) 
    data = json_data.get('data', '')
    selected_genres = json_data.get('selected_genres', [])
   #selected_feeling = json_data.get('selected_feeling', 0)
    selected_feeling = json_data.get('selected_feeling')
    selected_feeling_2 = json_data.get('selected_feeling_2', [])
    genre_yn = json_data.get('genre_yn')

    #### 추천 받은 곡 추천하지 않는 로직 - filtering 함수에 전달
    user_id = json_data.get('userId')
    global lyrics_df
    # selected_genres, selected_feeling > dummy

    #selected_genres = ['발라드', '인디음악', '댄스'] #최대3개
    #selected_feeling = 1 # 1: 매우좋음, 2: 좋음, 3: 보통, 4:나쁨, 5:매우나쁨


    # 일기의 키워드 추출
    diary_keywords = keyword_extraction(data)
    diary_embeddings = get_embeddings(diary_keywords)
    print(diary_embeddings)
    print(type(diary_embeddings))
    
    ## filtering 시작합니당
    print('user 확인 및 추천받은 곡 조회')
    pre_filter = filtering(user_id)
    print(pre_filter)

    ## 리스트가 비어있는지 확인한 뒤 lyrics_df에 적용하기
    if pre_filter :  
        lyrics_df = lyrics_df[~lyrics_df['songId'].isin(pre_filter)]
    
    else:
        print('pass')

    # 노래 가사 임베딩 계산
    lyrics_df['embeddings'] = lyrics_df['embeddings'].apply(parse_embedding)
    lyrics_embeddings = lyrics_df['embeddings'].tolist()
    

    # 코사인 유사도 계산
    cosine_sim = []
    for lyrics_embedding in lyrics_embeddings:
        sim_scores = cosine_similarity(lyrics_embedding, diary_embeddings)
        mean_sim = np.mean(sim_scores)
        cosine_sim.append(mean_sim)
    
    cosine_sim = np.array(cosine_sim)

######### 20240807 수정#################
    # 장르 필터링 적용된 곡 중 상위 2곡
    
    genre_filter = lyrics_df['genre'].apply(lambda x: any(genre in selected_genres for genre in x.split(', ')))
    genre_filtered_sim = cosine_sim * genre_filter
    
    # 동일 감정일 경우 유사도 조정
    try: 
        feeling_filter = lyrics_df['feelings'].isin(selected_feeling)
    except: 
        feeling_filter = lyrics_df['feelings'] == selected_feeling

    adjusted_genre_filtered_sim = genre_filtered_sim * (1.1 * feeling_filter + 1 * ~feeling_filter)

    # 비선호 장르 추천 받을게요
    if genre_yn == 1 : #
        print('비선호 장르 추천 받을게요')
        adjusted_genre_filtered_idxs = np.argsort(adjusted_genre_filtered_sim)[-2:][::-1]
        # adjusted_genre_filtered_idxs = np.argsort(adjusted_genre_filtered_sim)[-5:][::-1] #테스트용
        
        # 장르 필터링 적용되지 않은 곡 중 상위 1곡 (에디터 픽)
        non_genre_filter = ~genre_filter
        non_genre_filtered_sim = cosine_sim * non_genre_filter
        editor_pick_filter = lyrics_df['editor_pick'] == 'Y'
        adjusted_non_genre_filtered_sim = non_genre_filtered_sim * (1.25 * editor_pick_filter + 1 * ~editor_pick_filter)
        non_genre_filtered_idxs = np.argsort(adjusted_non_genre_filtered_sim)[-1:][::-1]
        # non_genre_filtered_idxs = np.argsort(adjusted_non_genre_filtered_sim)[-5:][::-1] #테스트용
        
        # 최종 추천 곡 인덱스
        final_idxs = np.concatenate((adjusted_genre_filtered_idxs, non_genre_filtered_idxs))
        final_scores = np.concatenate((adjusted_genre_filtered_sim[adjusted_genre_filtered_idxs], adjusted_non_genre_filtered_sim[non_genre_filtered_idxs]))

        print(final_scores)
    
    # 비선호 장르 추천 받지않음
    else :
        print('비선호 장르 추천 받지않음')
        editor_pick_filter = lyrics_df['editor_pick'] == 'Y'
        adjusted_genre_filtered_sim = adjusted_genre_filtered_sim * (1.25 * editor_pick_filter + 1 * ~editor_pick_filter)
        final_idxs = np.argsort(adjusted_genre_filtered_sim)[-3:][::-1]
        final_scores = np.array(adjusted_genre_filtered_sim[final_idxs])
##############수정끝##################

    df = lyrics_df.iloc[final_idxs].copy()
    df['similarity'] = final_scores
    df = df[['similarity','editor_pick','songId']]
    save_songId = df['songId'].tolist()
    print(save_songId)
    df = df.to_dict()

    print('save recommended songs in ddb table')
    insert_recommended_songId(user_id, save_songId)
    #print(diary_keywords)

    return df
    del df
# 수

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



