FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY rafiki/admin/requirements.txt admin/requirements.txt
RUN pip install -r admin/requirements.txt

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_admin.py start_admin.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

EXPOSE 8000

CMD ["python", "start_admin.py"]