FROM tensorflow/tensorflow:2.7.0

RUN apt update && \
    apt install -y git

RUN pip install black \
    click \
    jupyterlab \
    ipykernel \
    ipywidgets \
    matplotlib \
    pandas \
    tensorflow-datasets
