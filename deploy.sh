#!/usr/bin/env bash
#
# Deploy c6-server (backend) to AWS ECR.
#
# Builds the Docker image, pushes it to the chocolate-repository ECR repo, then
# leaves it to you to trigger the App Runner deployment (the service uses a Manual
# trigger; see c6-server/DEPLOY.md part A). Run from anywhere — it cd's to its own dir.
#
# Usage: ./deploy.sh
set -euo pipefail

cd "$(dirname "$0")"

# Constants for this account (see DEPLOY.md).
REGION="us-west-2"
REGISTRY="021891586863.dkr.ecr.us-west-2.amazonaws.com"
REPO="chocolate-repository"
SERVICE="chocolate-backend"
IMAGE="chocolate-docker:latest"
REMOTE="${REGISTRY}/${REPO}:latest"

echo "==> Preflight: checking Docker daemon"
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker isn't running (docker info failed). Start Docker and retry." >&2
  exit 1
fi

echo "==> Preflight: checking AWS credentials"
if ! aws sts get-caller-identity >/dev/null 2>&1; then
  echo "ERROR: not signed in to AWS (aws sts get-caller-identity failed)." >&2
  echo "       Authenticate first, then re-run ./deploy.sh" >&2
  exit 1
fi

echo "==> Building image ${IMAGE}"
docker build -t "${IMAGE}" .

# Optional local smoke test (skipped in the deploy flow — it blocks while serving):
#   docker run -p 8080:8080 "${IMAGE}"
#   curl -X POST http://localhost:8080/predict -F "file=@../c6-models/prediction.jpg"

echo "==> Logging in to ECR (${REGISTRY})"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

echo "==> Tagging and pushing ${REMOTE}"
docker tag "${IMAGE}" "${REMOTE}"
docker push "${REMOTE}"

echo "==> Pushed. Now trigger the App Runner deploy:"
echo "    Console → App Runner → ${SERVICE} → Deploy"
echo "    (or: aws apprunner start-deployment --region ${REGION} \\"
echo "           --service-arn arn:aws:apprunner:${REGION}:021891586863:service/${SERVICE}/16259e694e104af4a63582232513e1c2)"
echo "==> Done"
