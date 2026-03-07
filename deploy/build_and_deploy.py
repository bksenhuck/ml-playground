"""Build the Docker image via Cloud Build, push to GCR and deploy to Cloud Run.

Experiment runs are stored per-session in the browser (dcc.Store) — no DB
artifact is included in the image.

Usage:
    python -m deploy.build_and_deploy
    python -m deploy.build_and_deploy --deploy-only   # skip build+push

Requirements:
    - gcloud CLI authenticated and project set (no local Docker needed)
    - .env with GCP_PROJECT_ID (and optionally GCR_IMAGE, CLOUDRUN_SERVICE, GCP_REGION)
"""
import sys
import argparse
import logging
import subprocess

from deploy.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run(cmd: list[str], description: str) -> bool:
    logger.info("[%s] %s", description, " ".join(cmd))
    # Using shell=True for complex commands with arguments on Windows
    result = subprocess.run(" ".join(cmd), shell=True)
    if result.returncode != 0:
        logger.error("[%s] failed (exit %d)", description, result.returncode)
        return False
    logger.info("[%s] done", description)
    return True


def build(image: str) -> bool:
    """Build and push image using Google Cloud Build (no local Docker needed)."""
    return _run(
        ["gcloud", "builds", "submit", "--tag", image, "."],
        "BUILD (Cloud Build)",
    )


def deploy(image: str) -> bool:
    """Deploy the image to Cloud Run."""
    return _run(
        [
            "gcloud", "run", "deploy", settings.CLOUDRUN_SERVICE,
            "--image", image,
            "--region", settings.GCP_REGION,
            "--platform", "managed",
            "--allow-unauthenticated",
            "--memory", "4Gi",
            "--cpu", "1",
            "--set-env-vars", "LLM_ENGINE=qwen2.5,USE_LLM=true,GCS_MODEL_PATH=gs://{}/models/Qwen2.5-0.5B-Instruct".format(settings.GCP_PROJECT_ID),
        ],
        "DEPLOY",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and deploy ML Playground to Cloud Run"
    )
    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help="Skip build — redeploy using the existing image",
    )
    args = parser.parse_args()

    settings.validate()
    image = settings.get_docker_image()

    logger.info("=== ML PLAYGROUND DEPLOY ===")
    logger.info("Image : %s", image)
    logger.info("Service: %s  |  Region: %s", settings.CLOUDRUN_SERVICE, settings.GCP_REGION)

    if not args.deploy_only:
        # gcloud builds submit already builds and pushes
        if not build(image):
            sys.exit(1)

    if not deploy(image):
        sys.exit(1)

    logger.info("=== DEPLOY COMPLETE ===")


if __name__ == "__main__":
    main()
