"""
Detection of Fake Accounts on Social Media
Basic ML pipeline skeleton - Academic Prototype
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import create_sample_data, preprocess_data
from model import train_model, evaluate_model


def main():
    print("=" * 50)
    print("Fake Account Detection - Prototype")
    print("=" * 50)

    df = create_sample_data()
    print(f"\nSample data created: {len(df)} records")

    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessed and split.")

    model = train_model(X_train, y_train)
    print("Model trained.")

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
