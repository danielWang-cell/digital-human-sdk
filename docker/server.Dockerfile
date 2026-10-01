FROM weijhang/digital-human-env:ubuntu22.04-cpu-v1.0
WORKDIR /workspace
RUN python3 -m pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    fastapi==0.115.6 uvicorn==0.34.0 pydantic==2.10.6 \
    starlette==0.41.3 httpx==0.28.1
COPY server/requirements.txt /tmp/server-requirements.txt
EXPOSE 8000
