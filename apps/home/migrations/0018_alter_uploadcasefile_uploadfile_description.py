# Generated by Django 4.0.5 on 2022-08-14 19:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0017_alter_uploadcasefile_uploadfile_description'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadcasefile',
            name='uploadfile_description',
            field=models.TextField(blank=True, default=None, null=True),
        ),
    ]