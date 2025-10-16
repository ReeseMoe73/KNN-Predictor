import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder



def load_data(filepath):
    df = pd.read_csv(filepath)
    x = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
    y = df['Name']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return x, y_encoded, label_encoder

def train_knn(x, y, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, y)
    return knn

def predict_species(knn, label_encoder, features):
    prediction = knn.predict([features])
    species = label_encoder.inverse_transform(prediction)
    return species[0]

def main():
    import sys

    if len(sys.argv) != 5:
        print("Usage: python iris_knn_predictor.py <SepalLength> <SepalWidth> <PetalLength> <PetalWidth>")
        sys.exit(1)

    try:
        input_features = list(map(float, sys.argv[1:]))
    except ValueError:
        print("Error: Inputs must be floating-point numbers.")
        sys.exit(1)

    # Load data model
    x, y, label_encoder = load_data('iris.txt')
    knn_model = train_knn(x, y, k=5)

    # Predict  output
    predicted_species = predict_species(knn_model, label_encoder, input_features)
    print(f"Predicted Iris species: {predicted_species}")

if __name__ == "__main__":
    main()

