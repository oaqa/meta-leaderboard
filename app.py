import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import json
from rouge import Rouge 
import numpy as np
import time
import datetime
from Evaluator import Evaluator

UPLOAD_FOLDER = './uploads'
#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(['json'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#evalInstance = Evaluator("/Users/khyathi/Projects/QA_datasets/common_pipeline/bioasq_train_formatted.json")
evalInstance = Evaluator()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generateHtml(filename):
    if filename != 'initial_reload':
        scoreDict = evalInstance.performEvaluation(filename)
    measuresList = ['Rouge', 'Bleu', 'Precision', 'Recall', 'F-measure', 'F1-match', 'Accuracy']
    table = "<table>\n<tr> <th> Name </th> <th> Origin </th>"
    for el in measuresList:
        table += "<th>" + el + "</th>"
    table += "</tr>"
    #table += "<td>" + username + "</td>" + "<td>" + origin + "</td>"
    f = open("scores.txt", 'r')
    lines = f.readlines()
    for line in lines:
        w = line.strip().split()
        for el in w:
            table += "<td>" + el + "</td>"
        table += "</tr>"
    f.close()
    table += "</table>";
    return table


@app.route('/', methods=['GET', 'POST'])
def leaderboard():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            cur_time = str(datetime.datetime.utcnow())
            cur_time = cur_time.split()[0] + cur_time.split()[1]
            saved_file = os.path.join(app.config['UPLOAD_FOLDER'], filename.split(".")[0] + \
                        "_" + str(cur_time) + ".json") 
            file.save(saved_file)
            generateHtml(saved_file)
            #return redirect(url_for('uploaded_file', filename=filename))
            return redirect(url_for('leaderboard', filename=filename))
    return '''
    <!doctype html>
    <html>
    <head><title> Meta-LeaderBoard </title><link href= \" ./static/results.css \" rel=\"stylesheet\"></head>
    <h1>Meta Leader Board for Question Answering Tasks</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    <body>
    '''+generateHtml('initial_reload')+'''
    </body></html>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #filename += "_" + str(datetime.datetime.utcnow())
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    #generateHtml("x")
    app.run(host= "127.0.0.1" , debug=True, use_reloader=False, threaded=True)
