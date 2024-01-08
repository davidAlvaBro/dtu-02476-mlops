# Base image
FROM python:3.10-slim 

# Install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy over everything we need 
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY workspace/ workspace/
COPY data/ data/

# Set working directory
WORKDIR /
# Install dependencies and the library itself (our .toml file installs the dependencies)
RUN pip install . --no-cache-dir #(1)

# What is run when we start the container 
ENTRYPOINT ["python", "-u", "workspace/train_model.py"]
