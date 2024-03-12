FROM continuumio/miniconda3

RUN mkdir -p m2_cw

COPY . /m2_cw
WORKDIR /m2_cw

RUN conda env update --file environment.yml

RUN echo "conda activate m2_cw" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pre-commit install
