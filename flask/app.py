import os, glob
import pathlib
from IPython import get_ipython
from flask import Flask, url_for, redirect,  render_template, request, send_from_directory, Response, request
from werkzeug.utils import secure_filename
import numpy as np 
import pandas as pd
import re

SRC_PATH = pathlib.Path(__file__).parent.absolute()
UPLOAD_FOLDER = os.path.join(SRC_PATH,  'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='./templates/')
# f=open('./static/result.txt','r')
# g= f.read()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/forward/", methods=['POST'])
def move_forward():
    #Moving forward code

    os.chdir('../../darknet')

    os.system('./darknet detector test data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_best.weights ../flask/src/static/uploads/scan.jpg -thresh 0.3')
    os.system('./darknet detector test data/obj.data cfg/yolov4-custom.cfg ../training/yolov4-custom_best.weights -ext_output -dont_show < /home/jcwang/YOLOV4_TRAINING/yolov4_twentyclasses2/darknet/data/scan.txt> result.txt -thresh 0.3')

    result = open('result.txt', 'r')
    index = ['administ:', 'library:', 'audio:', 'science:', 'tower:', 'exercise:', 'mingde:',
            'swim:', 'creative:',  'mainhall:', 'zhishan:', 'duxing:', 'fanglan:', 'wenhui:', 
            'studentcenter:', 'dormitory1:', 'dormitory2:', 'audioback:', 'gallery:', 
            'doorway:']
    index1 = ['administ', 'library', 'audio', 'science', 'tower', 'exercise', 'mingde',
            'swim', 'creative',  'mainhall', 'zhishan', 'duxing', 'fanglan', 'wenhui', 
            'studentcenter', 'dormitory1', 'dormitory2', 'audioback', 'gallery', 
            'doorway']
    counts = []
    percentages = []

    line = result.readlines()[-2]
    str2 = line.split(' ')
        
    for i in range(20):
        count = 0
        percentage = '0'
        if str2[0] == index[i]:
            count = count + 1
            for j in range(len(str2)-1):
                if '%' in str2[j]:
                    percentage = str2[j]
                    numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', percentage)]
                    str_numbers = str(numbers)
                    str_numbers = str_numbers.replace('[', '')                    
                    str_numbers = str_numbers.replace(']', '')
                    percentage = str_numbers
                    
        counts.append(count)
        percentages.append(percentage)

    df = pd.DataFrame((zip(index1, counts, percentages)), columns = ['class', 'counts', 'accuracy'])
    df = df.set_index('class')
    df1 = df[df['counts'] != 0]
    df1 = df1[df['accuracy'] == max(df['accuracy'])]
    df.to_csv("result.txt", sep='\t', index=False)
    result = list(df1.index.astype(str))[0]

    f = open('result.txt', 'w')
    f.writelines(result)
    f.close()

    # testing code start
    forward_message = "辨識完成"
    # path = 'output.txt'
    # f = open(path, 'w')
    # f.write('Hello World')
    # f.close()
    # testing code end
    os.chdir('../flask/src')

    return render_template('index.html', forward_message=forward_message);

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['filename']
    if file.filename != '':
        file.filename = secure_filename('scan.jpg')
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
