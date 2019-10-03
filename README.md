virtualenv -p python2 py2env

<!-- <img src="{{ url_for('send_file', filename=image_file_name)}}" style ="max-width: 500px; max-height: 450px; margin-top: 30px; margin-bottom: 10px"> -->
<!-- <p style="font-style: italic">{{ image_file_name }}</p> -->

mkdir app app/templates app/static app/static/js app/static/css app/blueprints app/blueprints/home app/middlewares app/models
touch app/main.py app/app.yaml app/templates/base.html app/templates/home.html app/static/js/script.js app/static/css/style.css
touch app/blueprints/__init__.py
export NEW_BLUEPRINT=home # Change 'home' to make a new blueprint
mkdir app/blueprints/$NEW_BLUEPRINT
touch app/blueprints/$NEW_BLUEPRINT/__init__.py app/blueprints/$NEW_BLUEPRINT/blueprint.py
touch README.md
touch app/.gcloudignore


var canvas = $('paint');

export GCS_BUCKET="YOUR_BUCKET_NAME"
export GCP_PROJECT="YOUR_PROJECT_ID"
export FIREBASE_CONFIG="firebase_config.json"