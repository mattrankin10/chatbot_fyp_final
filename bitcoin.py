import requests
import subprocess
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

from postgres import run_sql_query, create_connection
from sentiment import analyze_text

r = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
client = language.LanguageServiceClient()
# create database connection
connection = create_connection()

chatting = input('Would you like to ask a question? Y/N?')
while chatting == 'Y':

    question = input('What would you like to know?: ')
    analysis = analyze_text(question)
    postgres_insert_query = "INSERT INTO questions(question,sent_score,sent_magnitude) VALUES (%s,%s,%s)"
    record_to_insert = (question,analysis[0],analysis[1])
    run_sql_query(connection, postgres_insert_query, record_to_insert)

    this_split = question.split()
    if 'bitcoin' in this_split:
        print('The current price of bitcoin is: £' + r.json()['bpi']['GBP']['rate'] + ', Equivalent to: $' +
              r.json()['bpi']['USD']['rate'])

    chatting = input('Would you like to ask a question? Y/N?')

# close connection before exiting program
connection.close()
