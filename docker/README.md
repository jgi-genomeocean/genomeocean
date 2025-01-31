We provided a docker image on DockerHub. You can access it through:

```bash
docker pull fengchenlbl/genomeocean:latest
docker run --rm -it --gpus all fengchenlbl/genomeocean:latest
git pull
python examples/test_model.py
```

Or you can build your own image with the Dockerfile.