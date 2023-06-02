
from flask import Flask,request,render_template
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

model=pickle.load(open('D:\\Data Science\\ML\\Classification\\Fruad Transaction Project\\credit_project.pkl', 'rb'))   
standard_scalar = pickle.load(open('D:\\Data Science\\ML\\Classification\\Fruad Transaction Project\\standardscalar.pkl', 'rb'))

app = Flask(__name__)   

@app.route('/') 
def fun():
    return render_template('index.html')


@app.route('/predict', methods =['GET','POST']) 
def fun1():

    a =[i for i in request.form.values()] 

    a = [int(j) if j.isdigit() else float(j) for j in a]

    a = np.array([a])

    res = standard_scalar.transform(a)
    
    sol = model.predict(res)[0]

    if sol == 0:
        return render_template('index.html', value = 'It is a Bad Tranzaction')
    else:
        return render_template('index.html', value = 'It is a Good Tranzaction')




if __name__ == '__main__':
    app.run(debug=True)