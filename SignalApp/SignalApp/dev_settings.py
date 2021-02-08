import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEBUG = True
ALGORITHM_USING_THREADS = 1
ALLOWED_HOSTS = ['127.0.0.1', '0.0.0.0']

# Database
DATABASES = {  # development
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}