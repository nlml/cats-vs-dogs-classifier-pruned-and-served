FROM ufoym/deepo:pytorch-py36-cu100
# For CPU replace the above line with that below:
# FROM ufoym/deepo:pytorch-py36-cpu

RUN apt-get update -y

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app
COPY templates /app/

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
