import os
from dotenv import load_dotenv
from kusa.client import SecureDatasetClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from inspect import signature
from sklearn.base import BaseEstimator

load_dotenv()

def train_model_factory(model_class: BaseEstimator, fixed_params=None):
    """
    Creates a universal training function that accepts only valid params for the model.
    Also warns about invalid params passed by the user.
    """
    fixed_params = fixed_params or {}

    def train_model(X, y, **params):
        sig = signature(model_class.__init__)
        accepted_params = set(sig.parameters.keys())

        combined = {**fixed_params, **params}
        valid_kwargs = {}
        for k, v in combined.items():
            if k in accepted_params:
                valid_kwargs[k] = v
            else:
                print(f"⚠️ Skipping unsupported param '{k}' for {model_class.__name__}")

        model = model_class(**valid_kwargs)
        model.fit(X, y)
        return model

    return train_model


def main():
    # Load credentials
    PUBLIC_ID = os.getenv("PUBLIC_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")

    # Step 1: Initialize secure client
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    initialization = client.initialize()
    # Step 2: Load encrypted dataset into memory
    client.fetch_and_decrypt_batch(batch_size=500, batch_number=1)

    # Step 3: Configure preprocessing
    client.configure_preprocessing({
         "tokenizer": "nltk",
        "stopwords": True,
        "reduction": "tfidf",
        "target_column": "RainTomorrow"
    })
    client.run_preprocessing()

    train_model = train_model_factory(GradientBoostingClassifier)
    # Step 4: Train model using internal data
    client.train(
        user_train_func=train_model,
        hyperparams={
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 1000,
             "n_estimators": 200,
        "learning_rate": 0.05
        },
        target_column="RainTomorrow"  # Make sure this column is your label (e.g., spam/ham)
    )

    # Step 5: Evaluate the model
    results = client.evaluate()
    print("\n✅ Evaluation Accuracy:", results["accuracy"])
    print("📊 Classification Report:\n", results["report"])

    # Step 6: Visualize Confusion Matrix
    y_true = client._SecureDatasetClient__y_val
    y_pred = client._SecureDatasetClient__trained_model.predict(client._SecureDatasetClient__X_val)
    cm = confusion_matrix(y_true, y_pred)

    print("y_true ",y_true)
    print("y_pred ",y_pred)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Step 7: Save the trained model
    client.save_model("secure_spam_model.joblib")

   

if __name__ == "__main__":
    main()
