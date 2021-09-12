FROM rust:1.55.0

# install tensorflow=2.6.0
# https://www.tensorflow.org/install/lang_c
WORKDIR /
RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz &&\
    tar -C /usr -xzf libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz &&\
    rm libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz

COPY ./pkgconfig/tensorflow.pc /usr/lib/pkgconfig/
