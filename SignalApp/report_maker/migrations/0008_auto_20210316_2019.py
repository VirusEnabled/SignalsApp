# Generated by Django 3.1.2 on 2021-03-16 20:19

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('report_maker', '0007_auto_20210316_1826'),
    ]

    operations = [
        migrations.AlterField(
            model_name='macdindicator',
            name='historical_data',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='report_maker.historicaldata'),
        ),
        migrations.AlterField(
            model_name='stochasticindicator',
            name='historical_data',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='report_maker.historicaldata'),
        ),
    ]
