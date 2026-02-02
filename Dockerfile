FROM python:3.13.1

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --default-timeout=300 -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]