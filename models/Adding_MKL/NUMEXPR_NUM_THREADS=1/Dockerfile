

FROM continuumio/miniconda3
WORKDIR /app

RUN conda install -c conda-forge numpy
RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge mkl


ENV export NUMEXPR_NUM_THREADS=1

COPY . .
CMD ["pipeline.py"]
ENTRYPOINT ["python3"]

