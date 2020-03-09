import psycopg2


def create_connection():
    connection = psycopg2.connect(user="fyp",
        password="chatbotfyp",
        host="fyp.czhkaabceukg.us-east-2.rds.amazonaws.com",
        port="5432",
        database="postgres")
    return connection


def run_sql_query(connection, query, record_to_insert):
    try:
        cursor = connection.cursor()
        cursor.execute(query, record_to_insert)
        connection.commit()

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        cursor.close()
