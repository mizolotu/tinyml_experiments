curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt update
ACCEPT_EULA=Y apt-get install -y msodbcsql17
apt install -y unixodbc-dev
