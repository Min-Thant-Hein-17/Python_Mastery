from flask import Flask,render_template,request
import pickle
app=Flask(__name__)
with open("iris_model.pkl",'rb') as f:
    model=pickle.load(f)
@app.route('/home',methods=["GET","POST"])
def home():
    if request.method=="POST":
        sepal_l=float(request.form.get('sl'))
        sepal_w=float(request.form.get('sw'))
        petal_l=float(request.form.get('pl'))
        petal_w=float(request.form.get('pw'))
        input_arr=[[sepal_l,sepal_w,petal_l,petal_w]]
        print(input_arr)
        result=model.predict(input_arr)[0]
        class_names=['Setosa','Versicolor','Vriginica']
        pred_value=class_names[result]
        return render_template('index.html',prediction=pred_value)
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)