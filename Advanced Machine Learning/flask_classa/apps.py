from flask import Flask,render_template, request
import pickle

app=Flask(__name__)
with open('iris_model.pkl','rb') as f:
    model=pickle.load(f)
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['GET',"POST"])
def predict():
    if request.method=='POST':
        sl=float(request.form.get('sepal_length'))
        sw=float(request.form.get('sepal_width'))
        pl=float(request.form.get('petal_length'))
        pw=float(request.form.get('petal_width'))
        arr=[[sl,sw,pl,pw]]
        flowers=['Setosa','Versicolor','Virginica']
        results=model.predict(arr)[0]
        results=flowers[results]
        return render_template('predict.html',pred_value=results)
    return render_template('predict.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
if __name__=="__main__":
    app.run(debug=True)