import pandas as pd

# 예시 DataFrame 생성
data = {
    'songId': [1, 2, 3, 4, 5],
    'title': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E']
}
test_df = pd.DataFrame(data)

# pre_filter 리스트
pre_filter = [2, 4]  # 제거할 songId 목록

# pre_filter에 있는 songId를 제외한 DataFrame 생성
test_df = test_df[~test_df['songId'].isin(pre_filter)]

# 결과 출력
print(test_df)
