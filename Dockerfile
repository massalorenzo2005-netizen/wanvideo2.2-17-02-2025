# Usa l'immagine ufficiale RunPod ottimizzata per ComfyUI
FROM runpod/worker-comfy:3.1.0-cuda12.1.1

# Installiamo i nodi presenti nel tuo repository (quelli di Kijai)
WORKDIR /comfyui/custom_nodes/ComfyUI-WanVideoWrapper
COPY . .

# Installiamo le dipendenze Python necessarie
RUN pip install --no-cache-dir -r requirements.txt

# Torniamo alla cartella principale del worker
WORKDIR /

# Il server cercherà di caricare i modelli dal Network Volume (/workspace)
ENV COMFYUI_PATH=/comfyui
ENV MODELS_PATH=/workspace/ComfyUI/models

# Comando di avvio automatico
CMD [ "python", "-u", "/handler.py" ]
