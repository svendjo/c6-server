# README.md
## Build a Docker image
`docker build -t chocholate-docker:latest .`

## Tag and push it to AWS ECR
`aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 021891586863.dkr.ecr.us-west-2.amazonaws.com`

`docker tag chocolate-docker:latest 021891586863.dkr.ecr.us-west-2.amazonaws.com/chocolate-repository:latest`

`docker push 021891586863.dkr.ecr.us-west-2.amazonaws.com/chocolate-repository:latest`

## Build frontend
`npm run build`

## Sync frontend to S3
`aws s3 sync ./build s3://chocolate-frontend`
