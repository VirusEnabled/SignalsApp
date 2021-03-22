import json
import os
import sys
import warnings
from datetime import datetime
from django.conf import settings
from django.core.management import BaseCommand, CommandError
from django.utils.timezone import make_aware
from django.conf import settings
from report_maker.helpers import process_file_data

class Command(BaseCommand):

    def add_arguments(self, parser):
        """
        adds arguments for the console parser, so that different options required
        with the command are capable of being processed
        :param parser: ArgumentParser object
        :return: None
        """
        parser.add_argument(
            "csvfile",
            type=str,
            help="path of the file to be loaded into the DB"
        )

        parser.add_argument(
            "--delimiter",
            type=str,
            help="delimiter character to identify the end of a column",
            default=';'
        )

    def handle(self, **options):
        """
        handles the commands prompted through the terminal or any type of call
        :param options: dict: dictionary containing the keyword
         arguments needed for the operations
        :return: None
        """
        if options.get("csvfile") is not None:
            file = options.get('csvfile')
            delimiter = options.get('delimiter')
            flag, result = process_file_data(csvfile=file, delimiter=delimiter)
            self.stdout.write(
                self.style.SUCCESS(
                    f'{"Successfully loaded" if flag else "There was an error loading the data in the DB:"}'
                                   f'{result}'))
        else:
            self.stderr.write("The parameter csvfile is mandatory to load the data in the DB.")
        exit(0)



