"""
Detection of Fake Accounts on Social Media
Basic ML pipeline skeleton - Academic Prototype
"""

from preprocess import create_sample_data, preprocess_data
from model import train_model, evaluate_model


def main():
    print("=" * 50)
    print("Fake Account Detection - Prototype")
    print("=" * 50)

    # Create sample data (replace with real dataset later)
    df = create_sample_data()
    print(f"\nSample data created: {len(df)} records")

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessed and split.")

    # Train
    model = train_model(X_train, y_train)
    print("Model trained.")

    # Evaluate
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
