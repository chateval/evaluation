FROM python:3
ENV EMBEDDING_FILE='vectors.magnitude'
ADD . /
WORKDIR /
ADD requirements.txt /
RUN pip install -r requirements.txt
RUN apt install wget
RUN wget http://magnitude.plasticity.ai/glove/glove.6B.50d.magnitude
RUN mv glove.6B.50d.magnitude vectors.magnitude
CMD [ "python", "./service.py" ]
