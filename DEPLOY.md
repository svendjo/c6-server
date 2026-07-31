# Count Chocolate II — AWS deployment guide

How to deploy **c6-server** (backend) and **c6-www** (frontend) to AWS. Everything
below is already provisioned and running — this guide covers both the routine
redeploy (`./deploy.sh` in each repo) and the one-time setup, so the stack can be
rebuilt from scratch.

| | Service | Where |
|---|---|---|
| Backend (`c6-server`) | **ECR** → **App Runner** | region `us-west-2` |
| Frontend (`c6-www`) | **S3** (static website) → **CloudFront** | bucket in `us-west-2`, CloudFront global |

Constants for this account:
- AWS account: `021891586863`
- ECR registry: `021891586863.dkr.ecr.us-west-2.amazonaws.com`
- Names: ECR repo `chocolate-repository`, App Runner service `chocolate-backend`,
  S3 bucket `chocolate-frontend`, CloudFront distribution `E18CMTJ2YTU1FL`,
  local image `chocolate-docker`.
- Live URLs: backend `https://22y7kxtymk.us-west-2.awsapprunner.com`,
  frontend `https://<E18CMTJ2YTU1FL>.cloudfront.net` (no custom domain).

> **Order matters:** deploy the **backend first**, get its App Runner URL, then put
> that URL into the frontend (`c6-www/src/config.js`) before building and uploading it.

The sibling Balut Eye stack (`abi-*` / `balut-*`) is the same shape on the same
account — see `abi-server/DEPLOY.md` if you want to compare.

---

## Part A — Backend → ECR + App Runner

`./deploy.sh` in `c6-server/` does A1 + A3 in one go and prints the A5 command.
The steps below are what it automates, plus the one-time setup it assumes.

### A1. Build the image (and test locally)
```sh
docker build -t chocolate-docker:latest .
docker run -p 8080:8080 chocolate-docker:latest    # in another shell:
curl -X POST http://localhost:8080/predict -F "file=@../c6-models/prediction.jpg"
curl http://localhost:8080/health
```
> The Dockerfile installs `requirements.txt` (the light **tflite-runtime** path, not
> full TensorFlow) and copies every module the server imports — `server.py`,
> `config.py`, `results_store.py` — plus `config/` and the two models named in
> `config/aws-prod.yaml`: `counting-model0926.tflite` and
> `classification-model0927.tflite`.
>
> **Deploying a retrained model** means editing two places: `models.counting` /
> `models.classification` in `config/local-dev.yaml` **and** `config/aws-prod.yaml`,
> and the Dockerfile's `COPY` line. The models are listed by name rather than globbed
> on purpose — c6-models drops every model it trains into this directory, and a glob
> would bake all of them (including the 101 MB `model0802.h5`) into the image.
>
> You're on an Intel Mac, so the image is already `linux/amd64` (what App Runner
> needs). On Apple Silicon you'd have to add `--platform linux/amd64` to the build.

### A2. Create the ECR repository (one time — already done)
```sh
aws ecr create-repository --repository-name chocolate-repository --region us-west-2
```

### A3. Tag & push
```sh
aws ecr get-login-password --region us-west-2 \
  | docker login --username AWS --password-stdin 021891586863.dkr.ecr.us-west-2.amazonaws.com
docker tag chocolate-docker:latest 021891586863.dkr.ecr.us-west-2.amazonaws.com/chocolate-repository:latest
docker push 021891586863.dkr.ecr.us-west-2.amazonaws.com/chocolate-repository:latest
```

### A4. Create the App Runner service (console, one time — already done)
Console → App Runner → **Create service**:
- Source: **Container registry** → Amazon ECR → `chocolate-repository:latest`
- Deployment trigger: **Manual** (so a push doesn't redeploy on its own)
- ECR access role: let the console create `AppRunnerECRAccessRole`
- Service name `chocolate-backend`, port **8080**
- Size: **0.25 vCPU / 0.5 GB** — what the service runs on today. The two TFLite
  models are ~43 MB combined and load once, so this fits; raise it if you add a
  bigger model.
- Health check: **TCP** (the default). `/health` exists if you'd rather switch it to
  HTTP — set the path to `/health`.

### A5. Redeploy later
After pushing a new `:latest` (A3 / `./deploy.sh`), trigger the deployment — the
service is on a **Manual** trigger, so pushing alone changes nothing:

```sh
aws apprunner start-deployment --region us-west-2 \
  --service-arn arn:aws:apprunner:us-west-2:021891586863:service/chocolate-backend/16259e694e104af4a63582232513e1c2
```
or Console → App Runner → `chocolate-backend` → **Deploy**.

### A6. Environments & results storage (S3) — ⚠️ not provisioned yet
c6-server selects an environment from `config/<APP_ENV>.yaml` (`config.py`); the YAML
carries the run flag (`reload`), the model filenames, the classifier's
`confidence_threshold`, and the results backend, so the run command stays just
`python server.py`.

| `APP_ENV`              | results backend                     | set by |
|------------------------|-------------------------------------|--------|
| `local-dev` (default)  | `results/<id>/` on local disk        | nothing — it's the default |
| `aws-prod`             | `s3://chocolate-results/<id>/`       | `ENV APP_ENV=aws-prod` in the Dockerfile |

Every prediction saves its `input.jpg` and `prediction.json` (or `error.txt`) under
its own id, which is what makes a complained-about count reproducible afterwards.
S3 is needed in prod because App Runner's local disk is **wiped on every
restart/redeploy** and **not shared across instances**.

**⚠️ The two AWS resources this needs do NOT exist yet.** Saving is deliberately
non-fatal — until they are created, prod predictions still return normally and just
log `WARNING: couldn't save this prediction: …` per request. To turn it on:

```sh
# 1. The bucket (all public access blocked by default).
aws s3api create-bucket --bucket chocolate-results --region us-west-2 \
  --create-bucket-configuration LocationConstraint=us-west-2

# 2. An instance role App Runner can assume, with write access to that bucket.
#    (Trust policy principal: tasks.apprunner.amazonaws.com)
aws iam create-role --role-name ChocolateAppRunnerInstanceRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"tasks.apprunner.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam put-role-policy --role-name ChocolateAppRunnerInstanceRole \
  --policy-name ChocolateResultsS3 --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["s3:PutObject","s3:GetObject"],"Resource":"arn:aws:s3:::chocolate-results/*"},{"Effect":"Allow","Action":"s3:ListBucket","Resource":"arn:aws:s3:::chocolate-results"}]}'
```

**3. Attach the instance role to the service** — Console → App Runner →
`chocolate-backend` → **Configuration → Edit → Security → Instance role** →
`ChocolateAppRunnerInstanceRole` → Save (redeploys). This is a *different* role from
the ECR access role in A4. The console is recommended over `aws apprunner
update-service` so you don't have to re-specify CPU/memory. **Until it is attached,
the container has no AWS credentials and every save will fail** (loudly, in the log).

If you'd rather not run S3 at all, set `results.backend: local` in
`config/aws-prod.yaml` — predictions are then saved to the container's disk and lost
on restart, which is at least honest about what you get.

---

## Part B — Frontend → S3 + CloudFront

`./deploy.sh` in `c6-www/` does B2 + B4 + B6 in one go.

### B1. Point the frontend at the backend (do this first!)
Nothing to edit per deploy: `src/config.js` holds one entry per environment, and
`npm run build` picks `aws-prod` automatically. You only touch it when the backend
URL itself changes:

```js
const CONFIG = {
  'local-dev': { apiBase: 'http://localhost:8080' },
  'aws-prod': { apiBase: 'https://22y7kxtymk.us-west-2.awsapprunner.com' },
};
```

### B2. Build
```sh
npm run build          # NODE_ENV=production -> the aws-prod entry above
```

### B3. Create + configure the S3 bucket (one time — already done)
```sh
aws s3api create-bucket --bucket chocolate-frontend --region us-west-2 \
  --create-bucket-configuration LocationConstraint=us-west-2
# allow a public-read bucket policy
aws s3api put-public-access-block --bucket chocolate-frontend \
  --public-access-block-configuration "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"
# static website hosting, index.html as both index and error document (SPA routing)
aws s3 website s3://chocolate-frontend --index-document index.html --error-document index.html
```
plus a public-read bucket policy on `arn:aws:s3:::chocolate-frontend/*`.

### B4. Upload the build
```sh
aws s3 sync ./build s3://chocolate-frontend --delete
```

### B5. Create the CloudFront distribution (console, one time — already done)
Origin: the bucket's **website endpoint**
(`chocolate-frontend.s3-website-us-west-2.amazonaws.com`, *not* the REST endpoint),
viewer protocol policy **Redirect HTTP to HTTPS**, default root object `index.html`.
This is distribution **`E18CMTJ2YTU1FL`**. No custom domain / ACM certificate — the
site lives at its `*.cloudfront.net` URL.

### B6. Update later
```sh
aws cloudfront create-invalidation --distribution-id E18CMTJ2YTU1FL --paths "/*"
```
CloudFront caches aggressively; without the invalidation you'll keep serving the old
bundle even after the S3 sync.

---

## Quick recap

```sh
cd c6-server && ./deploy.sh     # build + push, then trigger the App Runner deploy
cd ../c6-www  && ./deploy.sh    # build + sync + invalidate
```
