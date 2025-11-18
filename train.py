from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load dataset
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    # Train-test split (70/30)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, Y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(Y_test, preds)
    print("Model Accuracy:", acc)

    # Save model and test data
    joblib.dump(
        {"model": model, "X_test": X_test, "Y_test": Y_test},
        "savedmodel.pth"
    )
    print("Model saved as savedmodel.pth")

if __name__ == "__main__":
    main()

