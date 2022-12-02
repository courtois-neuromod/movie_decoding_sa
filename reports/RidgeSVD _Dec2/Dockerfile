FROM continuumio/miniconda3
WORKDIR /app

RUN conda install -c conda-forge git pip numpy scipy cython joblib threadpoolctl pytest compilers

RUN apt update -y && apt install -y gcc g++
RUN pip install  git+https://github.com/glatard/scikit-learn@test/dask-svd
RUN conda install -c conda-forge dask-jobqueue
RUN conda install -c conda-forge asyncssh
RUN conda install -c conda-forge dask-mpi

RUN conda install -c conda-forge "libblas=*=*mkl"
ENV MKL_NUM_THREADS=32

