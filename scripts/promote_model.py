import os
import shutil
import mlflow

METRIC_KEY = "accuracy"
MODEL_DIR = "./mlruns"
PROMOTED_DIR = os.environ.get("MODEL_PROMOTION_DIR", "./mlruns_promoted")
DEPLOY_TARGET = os.environ.get("DEPLOY_TARGET", "")

client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("streaming-churn-prediction")
last_run_id = client.list_run_infos(exp.experiment_id)[-1].run_id
metrics = client.get_run(last_run_id).data.metrics
print("Metrics from last run:", metrics)

# Fail if metric below threshold
THRESHOLD = 0.8
if metrics.get(METRIC_KEY, 0) < THRESHOLD:
    raise RuntimeError(f"Metric {METRIC_KEY} below threshold {THRESHOLD}, rollback triggered.")

# Check previous promoted model
if os.path.exists(PROMOTED_DIR):
    prev_metrics_file = os.path.join(PROMOTED_DIR, "metrics.txt")
    if os.path.exists(prev_metrics_file):
        prev_val = float(open(prev_metrics_file).read())
        if metrics[METRIC_KEY] <= prev_val:
            raise RuntimeError(f"New model metric {metrics[METRIC_KEY]} <= previous {prev_val}, rollback triggered.")

# Promote new model
if os.path.exists(PROMOTED_DIR):
    shutil.rmtree(PROMOTED_DIR)
shutil.copytree(MODEL_DIR, PROMOTED_DIR)
with open(os.path.join(PROMOTED_DIR, "metrics.txt"), "w") as f:
    f.write(str(metrics[METRIC_KEY]))
print("Model promoted successfully with metric:", metrics[METRIC_KEY])

# Optional deploy to S3/MinIO
if DEPLOY_TARGET:
    import subprocess
    subprocess.run(f"aws s3 cp {PROMOTED_DIR} {DEPLOY_TARGET} --recursive", shell=True, check=True)
    print(f"Model deployed to {DEPLOY_TARGET}")
