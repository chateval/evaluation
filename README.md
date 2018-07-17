# Evaluation
Microservice of [ChatEval](https://github.com/chateval/application) to handle evaluation of neural chatbot models. Uses both word embeddings and Amazon Mechanical Turk to evaluate models.

## Usage
The Evaluation microservice can be initialized by running `source init.sh` to `wget` the pre-trained word embeddings (configurable with an enviroment variable named `EMBEDDING_FILE`) and to run the Flask server at port 8001.

To run the automatic evaluation, a `POST` request must be made to `/auto` containing parameters `model_responses` and `baseline_responses`, as **equal length** string lists. The response is a JSON object containing keys for the evaluation metrics and their corresponding float values.
  
## (Optional) Docker Installation
ChatEval supports the use of Docker as both a development and deployment tool.

0. Install [Docker](https://docker.com/).
1. Configure environment variables in `Dockerfile` by adding `ENV variable value` for each environment variable.
2. Build Docker image by using `docker build -t evaluation .` (this may take some time).
3. Run Evaluation on port 8001 by using `docker run evaluation`
8. Access app at [localhost:8001](localhost:8001).
