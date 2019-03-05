# syntax=docker/dockerfile:experimental
FROM sharadmv/domain-randomization

WORKDIR /root/work/domain-randomization
RUN pipenv run pip uninstall -y tensorflow-gpu
RUN pipenv run pip install tensorflow==1.4.0
