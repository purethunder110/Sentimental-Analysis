from flask import Flask,render_template,request

from classifier_model import model_object

app=Flask(__name__)

@app.route('/',methods=["POST","GET"])
def main():
    classifier=model_object()
    if request.method=="GET":
        return render_template('index.html',result="")
    elif request.method=="POST":
        print("prompt entered="+request.form["prompt"])
        answer=classifier.predict_sentiment(request.form["prompt"])
        return render_template('index.html',result=answer)
    else:
        print("something has gone very wrong!!")
    

app.run(debug=False,port=8000)