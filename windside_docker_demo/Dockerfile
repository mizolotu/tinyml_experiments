FROM ubuntu:latest

EXPOSE 8050

RUN mkdir /app
WORKDIR /app
RUN apt update
RUN apt -y install python3 python3-pip lsb-core curl
ADD requirements.txt /app/
RUN pip install -r requirements.txt
ADD install_pyodbc_driver.sh /app/
RUN bash install_pyodbc_driver.sh
ADD . /app/

ENTRYPOINT ["python3"]
CMD ["app.py"]