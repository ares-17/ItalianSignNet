# Istruzioni e informazioni sull'utilizzo del kernel per notebook in locale


```bash
docker build -t signnet-notebook-kernel .

#Per utilizzare il kernel Jupyter esclusivamente su CPU
docker run -p 8888:8888 -v "$PWD:/workspace" -it signnet-notebook-kernel

#Per utilizzare il kernel Jupyter con supporto alle GPU in locale:
docker run --gpus all -p 8888:8888 -v "$PWD:/workspace" -it signnet-notebook-kernel
```bash
