FROM runpod/comfyui:latest-5090

# 1. Install GCSFuse
RUN apt-get update && apt-get install -y gnupg2 curl && \
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && apt-get install -y gcsfuse && \
    rm -rf /var/lib/apt/lists/*

# 2. Copia script di avvio e handler
COPY start.sh /start.sh
COPY model_cacher.py /model_cacher.py
COPY rp_handler.py /rp_handler.py
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml

RUN chmod +x /start.sh

# 3. Entrypoint
CMD ["/start.sh"]
