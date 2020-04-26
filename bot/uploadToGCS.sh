#!/bin/bash

file=$1
bucketname=$2
if [[ -n "$file" ]]; then
  if [[ -n "$bucketname" ]]; then
    gsutil cp $file gs://$bucketname
    echo "File uploaded to bucket"
  fi
else
    echo "argument error"
fi