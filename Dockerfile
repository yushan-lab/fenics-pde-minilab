FROM dolfinx/dolfinx:stable

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace
RUN python3 -m pip install --no-cache-dir -e ".[dev]"

CMD ["make", "reproduce"]
