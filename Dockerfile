FROM r-base:3.5.1
LABEL maintainer="luk.zim91@gmail.com"

COPY distributed_learning /opt
COPY init.R /tmp/init.R
WORKDIR /opt

# Install required packages and init empty model
RUN apt-get update -y && \
    apt-get install -y --no-install-suggests --no-install-recommends  \
    	libcurl4-openssl-dev \
        libgit2-dev \
        libssh2-1-dev \
    	libssl-dev \
        libxml2-dev && \
    Rscript -e 'install.packages(c("devtools", "roxygen2", "RcppArmadillo")); devtools::load_all()' && \
    mkdir -p /model && \
    mkdir -p /data && \
    Rscript /tmp/init.R && \
    rm -rf /tmp/* /var/tmp/*

