import joblib
from sklearn.metrics import accuracy_score

def main():
    data = joblib.load("savedmodel.pth")
    model = data["model"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    preds = model.predict(X_test)
    acc = accuracy_score(Y_test, preds)
    print("Loaded Model Test Accuracy:", acc)

if __name__ == "__main__":
    main()

