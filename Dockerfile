FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get install -y build-essential python3-dev

COPY src /m2_cw/src
COPY configs /m2_cw/configs
COPY environment.yml /mphil_medical_imaging_cw

WORKDIR /m2_cw

RUN conda env update --file environment.yml

RUN echo "conda activate m2_cw_env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
