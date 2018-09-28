FROM r-base:3.5.1
LABEL maintainer="luk.zim91@gmail.com"

# The environment
ENV CODE_DIR /opt/train/code
ENV DATA_DIR /opt/train/data
ENV MODEL_DIR /opt/train/model
ENV REGISTRY_FILE /opt/train/registry.RData

# Copy the code for the distributed learning
COPY distributed_learning "${CODE_DIR}"

# Copy required R scripts (init.R is only needed on build time)
COPY code/init.R /tmp/init.R

# Install required packages and init empty model
WORKDIR "${CODE_DIR}"
RUN apt-get update -y && \
    apt-get install -y --no-install-suggests --no-install-recommends  \
        git \
    	libcurl4-openssl-dev \
        libgit2-dev \
        libssh2-1-dev \
    	libssl-dev \
        libxml2-dev \
        python3-requests \
        python3-setuptools && \
    Rscript -e 'install.packages(c("devtools", "roxygen2", "RcppArmadillo")); devtools::load_all()' && \
    mkdir -p "${MODEL_DIR}" && \
    mkdir -p "${DATA_DIR}" && \
    Rscript /tmp/init.R && \
    cd /tmp && \
    git clone https://github.com/PersonalHealthTrain/train-api-python.git && \
    cd train-api-python && \
    python3 setup.py install && \
    cd /opt && \
    rm -rf /tmp/* /var/tmp/*

COPY code/print_summary.R "${CODE_DIR}/print_summary.R"
COPY code/training.R "${CODE_DIR}/training.R"
COPY entrypoint.py /entrypoint.py

ENTRYPOINT ["python3", "/entrypoint.py"]

