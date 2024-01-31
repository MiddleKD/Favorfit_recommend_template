ARG FUNCTION_DIR="/favorfit_color_template_recommend"
ARG DEPS_DIR="/python_lib"

FROM amazon/aws-lambda-python:3.9 as build-image

ARG FUNCTION_DIR
ARG DEPS_DIR

ENV PYTHONPATH=${DEPS_DIR}:${PYTHONPATH}

RUN mkdir -p ${FUNCTION_DIR}
RUN mkdir -p ${DEPS_DIR}
WORKDIR ${FUNCTION_DIR}

RUN yum -y update && yum -y install mesa-libGL glib2

COPY ./ ${FUNCTION_DIR}

RUN pip install --target ${DEPS_DIR} --no-cache-dir -r requirements.txt

CMD ["lambda_function.lambda_handler"]