import base64
import re
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from blueprints import * 

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT']=0
app.register_blueprint(home_page)
app.register_blueprint(make_sketch)


if __name__ == '__main__':
    app.run(debug=True)