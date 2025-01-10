# c6-server
Web service for Count Chocolate II.

## Local
Create a virtual environment.

`pyenv virtualenv 3.12.7 venv-c6-server`

Activate the environment

`pyenv local venv-c6-server`

Install requirements.

`pip install setuptools`
`pip install -r requirements.txt`

Build a Docker image.

`docker build -t chocolate-docker:latest .`

Run the Docker container locally. Remember to expose the port.

`docker run -p 8080:8080 chocolate-docker:latest`

Test the Docker container.

`curl -X POST http://localhost:8080/predict \
  -H "Content-Type: multipart/form-data" \
  -F "file=@workspace/c6-models/prediction.jpg"`

## Tag and push it to AWS ECR
`aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 021891586863.dkr.ecr.us-west-2.amazonaws.com`

`docker tag chocolate-docker:latest 021891586863.dkr.ecr.us-west-2.amazonaws.com/chocolate-repository:latest`

`docker push 021891586863.dkr.ecr.us-west-2.amazonaws.com/chocolate-repository:latest`

## Deploy AWS App Runner
Go to AWS App Runner and deploy chocolate-repository:latest.

## Build frontend
`npm run build`

## Sync frontend to S3
`aws s3 sync ./build s3://chocolate-frontend`
