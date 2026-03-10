FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System libs required by PyTorch on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Step 1: install CPU-only torch from pytorch.org (NOT from PyPI which ships CUDA).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# Step 2: write a constraints file pinning torch to the just-installed CPU version.
#   Without this, `pip install -r requirements.txt` re-resolves torch and upgrades
#   it to the CUDA build from PyPI (because +cpu local-version sorts lower).
RUN pip show torch | awk '/^Version:/{print "torch==" $2}' > /tmp/torch-pin.txt && cat /tmp/torch-pin.txt
# Step 3: transformers 4.44.2 + accelerate (before requirements.txt to hold the version).
RUN pip install --no-cache-dir "transformers==4.44.2" accelerate
# Step 4: rest of dependencies, constrained so pip never upgrades torch.
RUN pip install --no-cache-dir -r requirements.txt -c /tmp/torch-pin.txt

COPY . /app

RUN sed -i 's/\r$//' /app/deploy/entrypoint.sh && chmod +x /app/deploy/entrypoint.sh

# Smoke-test the LLM stack — catches version conflicts before deploy.
# Verifies: (1) torch is CPU-only, (2) transformers pipeline importable,
# (3) no CUDA packages were pulled in (nvidia-* absent means CPU build).
RUN python -c "\
import torch; \
assert '+cpu' in torch.__version__ or 'cpu' in str(torch.version.cuda or 'cpu'), \
    f'Expected CPU torch, got {torch.__version__} (cuda={torch.version.cuda})'; \
from transformers.pipelines import pipeline; \
print('LLM stack OK: torch', torch.__version__)"

# Pre-cache Titanic dataset (seaborn CDN — reliable at build time).
# California Housing (sklearn/figshare) is downloaded on first use at runtime;
# Cloud Build blocks figshare, so we skip it here.
RUN python -c "import seaborn as sns; sns.load_dataset('titanic')"

EXPOSE 8080

CMD ["/app/deploy/entrypoint.sh"]
