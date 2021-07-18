

FROM continuumio/miniconda3
WORKDIR /app

RUN conda install -c conda-forge numpy
RUN conda install -c conda-forge scikit-learn

ENV export MKL_NUM_threads=1

COPY . .
CMD ["pipeline.py"]
ENTRYPOINT ["python3"]

