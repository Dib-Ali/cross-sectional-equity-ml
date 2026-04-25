FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
	&& pip install -r /app/requirements.txt

COPY . /app
COPY README.md /README.md

RUN sed -i 's/\r$//' /app/scripts/docker_entrypoint.sh \
	&& chmod +x /app/scripts/docker_entrypoint.sh

EXPOSE 8501

CMD ["/app/scripts/docker_entrypoint.sh"]
