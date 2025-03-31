# 기본 이미지 설정
FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# JAVA_HOME 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# 작업 디렉토리 생성
WORKDIR /opt/program

# 의존성 파일 복사
COPY requirements.txt /opt/program/requirements.txt
COPY text_list.pkl /opt/program/text_list.pkl
COPY diary_feelings.csv /opt/program/diary_feelings.csv

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords punkt


COPY serve_v2.py /opt/program

ENTRYPOINT ["gunicorn", "-b", ":8080", "--workers", "6", "--threads", "46", "serve:app"]



