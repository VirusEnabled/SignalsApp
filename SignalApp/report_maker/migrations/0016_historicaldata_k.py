# Generated by Django 3.1.2 on 2021-04-15 22:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('report_maker', '0015_auto_20210413_2145'),
    ]

    operations = [
        migrations.AddField(
            model_name='historicaldata',
            name='k',
            field=models.FloatField(default=0.0),
        ),
    ]
