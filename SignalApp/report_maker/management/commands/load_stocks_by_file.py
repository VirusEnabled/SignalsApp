import json
import os
import sys
import warnings
from datetime import datetime
from django.conf import settings
from django.core.management import BaseCommand, CommandError
from django.utils.timezone import make_aware
from django.conf import settings
from report_maker.helpers import process_file_data, stock_excel_loader

class Command(BaseCommand):

    def add_arguments(self, parser):
        """
        adds arguments for the console parser, so that different options required
        with the command are capable of being processed
        :param parser: ArgumentParser object
        :return: None
        """
        parser.add_argument(
            "--csvfile",
            type=str,
            help="path of the file to be loaded into the DB"
        )

        parser.add_argument(
            "--delimiter",
            type=str,
            help="delimiter character to identify the end of a column",
            default=';'
        )

        parser.add_argument(
            '--excel_file',
            type=str,
            help='Loads the given file with the stocks into the database,'
                 'Location of the excel file to load the information from. This file must be in .xls extension to work.'
        )

        parser.add_argument(

            '--sheet_number',
            type=int,
            help='Sheet number to load the information from, this is based on a multi-paged excel file, the default is 2',
            default=2

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

        elif options.get('excel_file'):
            file = options.get('excel_file')
            sheet_number=options.get('sheet_number')
            if file and sheet_number:
                result = stock_excel_loader(file, sheet_number)
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully loaded in the DB!!') if result['status'] else
                self.style.ERROR(f"{result['error']}"))

            else:
                self.stderr.write("The parameters --excel_file and --sheet_number are mandatory to "
                                  "load the data in the DB., check the --help option")

        else:
            self.stderr.write("you must provide either a csvfile or a excel_file path to make this work.")
        exit(0)



