from flask import Flask, render_template, request
import pickle
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('ind.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        spl = request.form['spl']
        ptl = request.form['ptl']
        ptw = request.form['ptw']
        spw = request.form['spw']
        
        data=[[float(spl),float(spw),float(ptl),float(ptw)]]
        Ir = pickle.load(open('iris.pkl','rb'))
        prediction = Ir.predict(data)[0]
    return render_template('ind.html', prediction=prediction)

if __name__== '__main__':
    app.run()