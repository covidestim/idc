#!/bin/bash

docker run \
  -it \
  --rm \
  -v $(pwd):/opt/data \
  -e AZURE_OCP_APIM_SUBSCRIPTION_KEY=e440893ed6154498893cf424c7891081 \
  covidestim/idc:test \
    python ./idc.py \
      -o /opt/data/idc_out.csv \
      --key fips \
      /opt/data/idc_in.csv
