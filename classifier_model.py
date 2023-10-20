import joblib

class model_object:
    
    def __init__(self):
        self.tfidf_vectorizer=joblib.load("data/vector.bin")
        self.svm_classifier=joblib.load("data/classifier.bin")
    
    def predict_sentiment(self,prompt):
        prompt_tfidf = self.tfidf_vectorizer.transform([prompt])
        prediction = self.svm_classifier.predict(prompt_tfidf)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        return sentiment


if __name__=="__main__":
    obj=model_object()
    inp=input("enter prompt:")
    print(obj.predict_sentiment(inp))