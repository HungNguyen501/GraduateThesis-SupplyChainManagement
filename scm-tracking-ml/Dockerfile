FROM python:3.7 AS compile-image

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.7 AS build-image
COPY --from=compile-image /opt/venv /opt/venv

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ .

RUN python Prepare.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]