import json
import os

import requests

webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

text = "Hello world! <https://github.com/IMOKURI/petfinder-pawpularity-score|petfinder2 github repo>"

requests.post(webhook_url, data=json.dumps({"text": text}))
