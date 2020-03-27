#!/bin/bash

bucketname=$1

if [[ -n "$bucketname" ]]; then
  gsutil cp train.from gs://$bucketname
  gsutil cp train.to gs://$bucketname
  gsutil cp test.from gs://$bucketname
  gsutil cp test.to gs://$bucketname
  echo "Files uploaded to bucket"
fi
else
    echo "argument error"
