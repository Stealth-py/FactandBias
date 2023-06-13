FROM  nvcr.io/nvidia/pytorch:23.05-py3
#nvcr.io/nvidia/pytorch:23.05-py3
#python:3.9

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./ /code

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["python", "main.py"]