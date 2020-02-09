import os
from .settings import *
import dj_database_url

DEBUG = False
TEMPLATE_DEBUG = False
ALLOWED_HOSTS = [os.environ.get('ALLOWED_HOSTS'),
                 'localhost',
                 'mission-compostable.herokuapp.com',
                 'missioncompostable.online',
                 'www.missioncompostable.online']

# DATABASES['default'].update(dj_database_url.config(conn_max_age=500))

SECRET_KEY = os.environ.get('SECRET_KEY')

STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
