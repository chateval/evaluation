FROM python:3
ENV EMBEDDING_FILE='vectors.magnitude'
ADD . /
WORKDIR /
ADD requirements.txt /
RUN pip install -r requirements.txt
RUN apt install wget
RUN wget http://magnitude.plasticity.ai/glove/glove.6B.50d.magnitude
RUN mv glove.6B.50d.magnitude vectors.magnitude
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "service:app"]
