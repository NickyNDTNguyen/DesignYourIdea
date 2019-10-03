from flask import Blueprint, render_template

home_page = Blueprint('home_page', __name__)

@home_page.route("/")
def hello_world():
    return render_template("home.html")