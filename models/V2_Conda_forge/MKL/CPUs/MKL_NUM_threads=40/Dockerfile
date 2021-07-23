FROM continuumio/miniconda3
WORKDIR /app


RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge "libblas=*=*mkl"

ENV MKL_NUM_THREADS=40

COPY . .
CMD ["pipeline.py"]
ENTRYPOINT ["python3"]
