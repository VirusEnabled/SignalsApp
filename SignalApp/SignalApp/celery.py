from __future__ import absolute_import, unicode_literals
import os
import celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE','SignalApp.settings')
app = celery.Celery('SignalApp')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
