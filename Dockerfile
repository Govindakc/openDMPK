FROM hassanmohsin/rdkit-openbabel:latest

RUN apt update && apt upgrade -y
RUN apt install -y python3-pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY models ./models
COPY run_opendmpk.py descriptors.py models.txt ./

ENTRYPOINT ["python3", "run_opendmpk.py"]
