import json

import google.cloud
from glob import glob
from google.cloud import storage
import os
import errno
import pathlib

original_cwd = os.getcwd()

bucket_name = 'amazonqavideogames'
client = storage.Client()
bucket = client.get_bucket(bucket_name)
blobs = client.list_blobs(bucket_name)

# set max length sentence
MAX_LENGTH = 40

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def download_blobs(blobs_list):
    os.chdir('blobs/')
    mkdir('train/')
    mkdir('test/')
    os.chdir("..")
    for blob in blobs_list:
        blob_name = str(blob.name)
        if blob_name.startswith('opensubtitles/20200331/train'):
            download_blob(bucket_name, blob_name, 'blobs/train/' + blob_name.partition('opensubtitles/20200331/')[2])
        elif blob_name.startswith('opensubtitles/20200331/test'):
            download_blob(bucket_name, blob_name, 'blobs/test/' + blob_name.partition('opensubtitles/20200331/')[2])


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def clean_file(fileName):
    with open(fileName, 'a', encoding='utf8') as f:
        f.truncate(0)


def clean_line(line):
    return line.replace("\n", "").replace("\r", "")


def append_to_file(fromFile, toFile, data):
    with open(fromFile, 'a', encoding='utf8') as f:
        with open(toFile, 'a', encoding='utf8') as t:
            count = 0
            for thread in data:
                count += 1
                if "?" in thread["context"] and len(thread["context"]) <= MAX_LENGTH:
                    context = clean_line(thread["context"]) + '\n'
                    response = clean_line(thread["response"]) + '\n'
                    f.write(context)
                    t.write(response)

                if "context/0" in thread and "?" in thread["context/0"] and len(thread["context/0"]) <= MAX_LENGTH:
                    context = clean_line(thread["context/0"]) + '\n'
                    response = clean_line(thread["context"]) + '\n'
                    f.write(context)
                    t.write(response)

                for i in range(1,10):
                    if "context/" + str(i) in thread \
                            and "?" in thread["context/" + str(i)] \
                            and len(thread["context/" + str(i)]) <= MAX_LENGTH:
                        context = clean_line(thread["context/" + str(i)]) + '\n'
                        response = clean_line(thread["context/" + str(i-1)]) + '\n'
                        f.write(context)
                        t.write(response)
    # close after each time we write to the file to make sure we don't run out of memory
    f.close()
    t.close()


def read_json_and_write_prepared_data(directory, fromFile, toFile):
    # append test and train arrays with threads
    count = 0
    for path in pathlib.Path(directory).iterdir():
        print("Doing " + path.stem)
        if path.is_file():
            with open(path, "r") as current_file:
            # create JSON object from data
            data = []
            line_count = 0
            for line in current_file:
                line_count += 1
                try:
                    thread = json.loads(line)
                    if thread["context"] and thread["response"]:
                        data.append(thread)
                except:
                    print("Error reading json on line " + str(line_count))
            append_to_file(fromFile, toFile, data)
            current_file.close()
            count += 1


def prepare():
    mkdir('blobs')
    # download objects from gcs
    dir = os.listdir("blobs")
    if len(dir) == 0:
        download_blobs(blobs)
    # clean files
    clean_file("testquestions.from")
    clean_file("testquestions.to")
    clean_file("trainquestions.from")
    clean_file("trainquestions.to")

    read_json_and_write_prepared_data("blobs/train/", "trainquestions.from", "trainquestions.to")
    read_json_and_write_prepared_data("blobs/test", "testquestions.from", "testquestions.to")


prepare()
