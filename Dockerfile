FROM python:3.9

#
WORKDIR /code
#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./ /code
RUN ls

RUN ls /code
#
CMD ["streamlit", "run", "/code/app.py", "--server.port=8501", "--server.address=0.0.0.0"]