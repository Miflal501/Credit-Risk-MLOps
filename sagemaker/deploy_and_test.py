"""Deploy a trained model artifact to a real-time endpoint, test it, then DELETE it.

python sagemaker/deploy_and_test.py --role <role-arn> --model-data s3://.../model.tar.gz
(find the S3 path in the training job, or the registered model package, in the console)
Deleting the endpoint at the end is what keeps the cost near zero.
"""

import argparse

from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer
from sagemaker.sklearn.model import SKLearnModel

SAMPLE = {
    "records": [
        {
            "checking_status": "<0", "duration": 24, "credit_history": "existing paid",
            "purpose": "radio/tv", "credit_amount": 3000, "savings_status": "<100",
            "employment": "1<=X<4", "installment_commitment": 3, "personal_status": "male single",
            "other_parties": "none", "residence_since": 2, "property_magnitude": "car",
            "age": 35, "other_payment_plans": "none", "housing": "own",
            "existing_credits": 1, "job": "skilled", "num_dependents": 1,
            "own_telephone": "none", "foreign_worker": "yes",
        }
    ]
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--model-data", required=True)
    p.add_argument("--instance-type", default="ml.m5.large")
    args = p.parse_args()

    model = SKLearnModel(
        model_data=args.model_data,
        role=args.role,
        entry_point="inference.py",
        source_dir="src",
        framework_version="1.2-1",
        py_version="py3",
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )
    try:
        print(predictor.predict(SAMPLE))
    finally:
        predictor.delete_endpoint()
        predictor.delete_model()
        print("Endpoint and model deleted.")


if __name__ == "__main__":
    main()
