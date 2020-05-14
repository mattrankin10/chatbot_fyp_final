from google.cloud import storage


bucket_name = 'amazonqavideogames'
client = storage.Client()
bucket = client.get_bucket(bucket_name)
blobs = client.list_blobs(bucket_name)

train_count = 0
test_count = 0
for blob in blobs:
    blob_name = str(blob.name)

    if blob_name.startswith('opensubtitles/20200331/train'):
        train_count += 1
    elif blob_name.startswith('opensubtitles/20200331/test'):
        test_count += 1

print('train: ' + str(train_count),'test: ' + str(test_count))