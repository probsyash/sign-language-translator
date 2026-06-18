import joblib
from sklearn.metrics import accuracy_score, classification_report

classifier = joblib.load("dataStorage/hand_sign_classifier.pkl")

def load_landmarks(path):
    x, y = [], []
    with open(path) as f:
        for line in f:
            coords = line.strip().split(",")
            x.append([float(v) for v in coords[:-1]])
            y.append(coords[-1])
    return x, y

x_test, y_test = load_landmarks("dataStorage/landmarks_test.txt")
y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred)

print("Held-out accuracy:", accuracy)
print(report)

with open("dataStorage/metrics.txt", "w") as f:
    f.write(f"Held-out session accuracy: {accuracy}\n\n")
    f.write(report)