FROM continuumio/miniconda3
WORKDIR /app


RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge "libblas=*=*openblas"

ENV OPENBLAS_NUM_THREADS=16

COPY . .
CMD ["pipeline.py"]
ENTRYPOINT ["python3"]
