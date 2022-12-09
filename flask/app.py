import os, glob
import pathlib
from flask import Flask, url_for, redirect,  render_template, request, send_from_directory
from werkzeug.utils import secure_filename


SRC_PATH = pathlib.Path(__file__).parent.absolute()
UPLOAD_FOLDER = os.path.join(SRC_PATH,  'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='./templates/')
f=open('./static/result.txt','r')
g= f.read()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.txt')


@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['filename']
    if file.filename != '':
        file.filename = secure_filename('scan.jpg')
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return redirect(url_for('index'))


@app.route('/')
def index():
    return render_template('./templates/index.txt',ans=g)


# @app.route('/<path>')
# def building(path):
#     base_dir = os.path.dirname(__file__)
#     file = open('./static/result.txt', r)
#     result = file.read()
#     file.close()
#     return result
#     # resp = make_response(open(os.path.join(base_dir, path)).read())
#     # resp.headers["Content-type"]="text/plan;charset=UTF-8"
    # return resp

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == "__main__":
    app.run(debug=True)
