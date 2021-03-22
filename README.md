# SignalsApp
This project is about cosuming the data coming from the stock market and being able to produce statistical analysis on them to determine how the stocks behave


### Development

1. Branch master to dev
2. Setting enviroment
    env file
    SITE_ENVIRONMENT=DEV
3. Work in dev
4. Once the changes are ready, merge dev with master

### Dependencies

### Ubuntu/Debian based Linux:

- **Install Git** *sudo apt get install git*
- **Install Python 3.8** *sudo apt-get install python3*
- **Install the following dependencies**

```bash
sudo apt install -y build-essential libffi-dev libssl-dev \
                    libxml2-dev libxslt1-dev \
                    python3-venv python3-pip python3-dev \
                    libjpeg8-dev zlib1g-dev wkhtmltopdf \
                    x11vnc xvfb \
                    latexmk texlive-formats-extra
```

- **Install the libraries from the requirements.txt**

---
### Custom Commandline Commands:

As of now we're only having *load_stock_by_file*, probably we're going to have more in the future
to run it you need to run the following script where csvfile is the path of the file you're looking for loading:
```
python3.8 manage.py load_stocks_by_file /path/to/your/csv/file/ --delimiter ';'
```
**the *--delimiter* option is optional in case you're using a different delimiter, you can specify it but by default it's a semicolon (;)** 
---
