import json

import google.cloud
from glob import glob
from google.cloud import storage
import os
import errno

original_cwd = os.getcwd()

bucket_name = 'amazonqavideogames'
client = storage.Client()
bucket = client.get_bucket(bucket_name)
blobs = client.list_blobs(bucket_name)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def clean_file(file):
    return file.truncate(0)


def clean_line(line):
    return line.replace("\n", "").replace("\r", "")


def prepare():
    mkdir('blobs')
    # create JSON object from data
    data_test = []
    data_train = []

    # download objects from gcs
    for blob_name in blobs:
        if str(blob_name.name).startswith('reddit/20200307/train'):
            download_blob(bucket_name, blob_name, 'blobs/train/' + blob_name.partition('reddit/20200307/')[2])
        elif str(blob_name.name).startswith('reddit/20200307/test'):
            download_blob(bucket_name, blob_name, 'blobs/test/' + blob_name.partition('reddit/20200307/')[2])

    # append test and train arrays with threads
    for file in os.listdir('blobs/train/'):
        os.chdir('blobs/train/')
        with open(file) as f:
            for line in f:
                thread = json.loads(line)
                if thread["context"] and thread["response"]:
                    data_train.append(thread)

    os.chdir('..')
    for file in os.listdir('test/'):
        os.chdir('test/')
        with open(file) as f:
            for line in f:
                thread = json.loads(line)
                if thread["context"] and thread["response"]:
                    data_test.append(thread)

    os.chdir('../..')

    # create question and answer files
    with open("test.from", 'a', encoding='utf8') as f:
        clean_file(f)
        for thread in data_test:
            f.write(clean_line(thread["context"]) + '\n')

    with open("test.to", 'a', encoding='utf8') as f:
        clean_file(f)
        for thread in data_test:
            f.write(clean_line(thread["response"]) + '\n')

    with open("train.from", 'a', encoding='utf8') as f:
        clean_file(f)
        for thread in data_train:
            f.write(clean_line(thread["context"]) + '\n')

    with open("train.to", 'a', encoding='utf8') as f:
        clean_file(f)
        for thread in data_train:
            f.write(clean_line(thread["response"]) + '\n')

    os.chdir(original_cwd)
