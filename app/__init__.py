from flask import Flask
from flask_cors import CORS
from flask_compress import Compress

app = Flask(__name__)
CORS(app)
Compress(app)
