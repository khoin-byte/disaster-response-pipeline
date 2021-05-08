from flask import Flask

app = Flask(__name__)

from disaster_response_app import routes