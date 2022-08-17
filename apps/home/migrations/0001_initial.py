# Generated by Django 3.2.6 on 2022-07-24 07:21

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='case',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('case_name', models.CharField(max_length=120)),
                ('description', models.TextField()),
                ('case_type', models.CharField(max_length=40)),
            ],
        ),
    ]