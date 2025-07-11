# Generated by Django 3.1.2 on 2021-03-13 06:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('report_maker', '0003_auto_20210313_0640'),
    ]

    operations = [
        migrations.AlterField(
            model_name='historicaldata',
            name='adr',
            field=models.DecimalField(decimal_places=30, default=0.0, max_digits=30),
        ),
        migrations.AlterField(
            model_name='historicaldata',
            name='close',
            field=models.DecimalField(decimal_places=30, default=0.0, max_digits=30),
        ),
        migrations.AlterField(
            model_name='historicaldata',
            name='high',
            field=models.DecimalField(decimal_places=30, max_digits=30),
        ),
        migrations.AlterField(
            model_name='historicaldata',
            name='low',
            field=models.DecimalField(decimal_places=30, max_digits=30),
        ),
        migrations.AlterField(
            model_name='historicaldata',
            name='macd',
            field=models.DecimalField(decimal_places=30, default=0.0, max_digits=30),
        ),
        migrations.AlterField(
            model_name='historicaldata',
            name='open',
            field=models.DecimalField(decimal_places=30, max_digits=30),
        ),
        migrations.AlterField(
            model_name='historicaldata',
            name='rsi',
            field=models.DecimalField(decimal_places=30, default=0.0, max_digits=30),
        ),
        migrations.AlterField(
            model_name='historicaldata',
            name='stochastic',
            field=models.DecimalField(decimal_places=30, default=0.0, max_digits=30),
        ),
        migrations.AlterField(
            model_name='historicaldata',
            name='volume',
            field=models.DecimalField(decimal_places=30, default=0.0, max_digits=30),
        ),
    ]
