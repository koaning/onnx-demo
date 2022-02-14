FROM python:3.7

# This needs to be set manually, appearantly.
RUN apt-get update && apt-get install -y locales locales-all
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD uvicorn webapp:app --host=0.0.0.0 --port=8080
