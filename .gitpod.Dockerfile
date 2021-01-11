FROM continuumio/miniconda3

RUN conda env update -n base -f environment.yml --quiet
