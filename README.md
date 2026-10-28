[![ci](https://github.com/Miflal501/Credit-Risk-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/Miflal501/Credit-Risk-MLOps/actions/workflows/ci.yml)
# Credit Risk MLOps Pipeline

End-to-end ML engineering project: tabular model training with experiment tracking and a
model registry, a containerised prediction API, CI, drift monitoring, and a SageMaker pipeline.

| Stage | Tools | Where |
|---|---|---|
| Features + model | scikit-learn Pipeline, XGBoost, 5-fold CV, randomized search, SHAP | `src/train.py` |
| Tracking + registry | MLflow (SQLite backend), quality gate before registration | `src/train.py` |
| Serving | FastAPI (`/health`, `/metadata`, `/predict`), Docker | `src/app.py`, `Dockerfile` |
| CI | GitHub Actions: lint, tests, train, build image, container smoke test | `.github/workflows/ci.yml` |
| Monitoring | PSI-based data drift check | `src/monitor.py` |
| Cloud | SageMaker Pipeline (train, register), endpoint deploy and teardown | `sagemaker/` |

Dataset: German Credit (OpenML `credit-g`), target = bad credit risk.

## Quick start (local)

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest -q                          # unit tests (synthetic data, no download)
    python src/train.py                # downloads credit-g, trains, logs to MLflow, registers if AUC >= 0.70
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse runs and the model registry

## Serve with Docker

    docker build -t credit-risk-api .   # needs models/ from the training step
    docker run -p 8000:8000 credit-risk-api
    curl localhost:8000/health
    curl localhost:8000/metadata        # shows the expected feature columns

`POST /predict` takes `{"records": [{...feature columns...}]}` and returns default probabilities.

## Drift check

    python src/monitor.py --reference models/reference.csv --current new_data.csv
    # exit code 1 if any numeric feature has PSI > 0.2

## SageMaker (needs an AWS account and an execution role)

    pip install -r requirements-aws.txt
    python sagemaker/pipeline.py --role <role-arn>          # train, then register a model package
    # approve the model package version in the console, copy the model artifact S3 URI, then:
    python sagemaker/deploy_and_test.py --role <role-arn> --model-data s3://.../model.tar.gz

Cost control: use `ml.m5.large`, run `deploy_and_test.py` (it deletes the endpoint when done), and
set an AWS Budget alert first. In the console, check there are no endpoints left running.

## Notes
- Keep `requirements.txt` and `requirements-serve.txt` versions identical: the model is a pickle.
- The SageMaker scripts target SDK v2 (`sagemaker<3`) and the SKLearn 1.2-1 container.
