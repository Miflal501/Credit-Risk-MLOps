"""SageMaker Pipeline: train (script mode) -> register in the Model Registry.

Written for sagemaker SDK v2 (pip install -r requirements-aws.txt). Run from the repo root:
    python sagemaker/pipeline.py --role arn:aws:iam::<account>:role/<SageMakerExecutionRole>
Approve the registered model version in SageMaker Studio / console (or via API) before deploying.
"""

import argparse

import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import TrainingStep

FRAMEWORK_VERSION = "1.2-1"


def ensure_model_package_group(name: str) -> None:
    client = boto3.client("sagemaker")
    try:
        client.create_model_package_group(
            ModelPackageGroupName=name, ModelPackageGroupDescription="Credit risk models"
        )
    except client.exceptions.ClientError as exc:
        if "already exists" not in str(exc):
            raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--group", default="credit-risk-models")
    p.add_argument("--pipeline-name", default="credit-risk-pipeline")
    p.add_argument("--instance-type", default="ml.m5.large")
    args = p.parse_args()

    ensure_model_package_group(args.group)
    session = PipelineSession()

    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src",  # also installs src/requirements.txt in the container
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        hyperparameters={"n-iter": 10, "cv": 5},
        sagemaker_session=session,
    )
    train_step = TrainingStep(name="TrainCreditRisk", step_args=estimator.fit())

    model = SKLearnModel(
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=args.role,
        entry_point="inference.py",
        source_dir="src",
        framework_version=FRAMEWORK_VERSION,
        py_version="py3",
        sagemaker_session=session,
    )
    register_step = ModelStep(
        name="RegisterCreditRisk",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=args.group,
            approval_status="PendingManualApproval",
        ),
    )

    pipeline = Pipeline(
        name=args.pipeline_name, steps=[train_step, register_step], sagemaker_session=session
    )
    pipeline.upsert(role_arn=args.role)
    execution = pipeline.start()
    print("Started:", execution.arn)


if __name__ == "__main__":
    main()
