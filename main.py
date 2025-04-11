import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

from kusa.client import SecureDatasetClient

load_dotenv()

# ✅ Universal training function builder
def train_model_factory(model_class: BaseEstimator, fixed_params=None):
    """
    Creates a safe training function for any sklearn model,
    filtering out unsupported hyperparameters.
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

# ✅ Main pipeline
def main():
    PUBLIC_ID = os.getenv("PUBLIC_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")

    print("🔐 Initializing secure client...")
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    initialization = client.initialize()
    
    print("📦 Fetching and decrypting dataset...")
    client.fetch_and_decrypt_batch(batch_size=500, batch_number=1)

    print("⚙️ Configuring preprocessing...")
    client.configure_preprocessing({
        "tokenizer": "nltk",
        "stopwords": True,
        "reduction": "tfidf",
        "target_column": "RainTomorrow"
    })
    client.run_preprocessing()

    print("🎯 Creating training function for Gradient Boosting...")
    train_model = train_model_factory(
        GradientBoostingClassifier,
        fixed_params={}  # Add if you want default consistent params
    )

    print("🚀 Training model...")
    client.train(
        user_train_func=train_model,
        hyperparams={
            "n_estimators": 200,
            "learning_rate": 0.05
        },
        target_column="RainTomorrow"
    )

    print("📈 Evaluating model...")
    results = client.evaluate()
    print("\n✅ Evaluation Accuracy:", results["accuracy"])
    print("📊 Classification Report:\n", results["report"])

    print("📉 Visualizing confusion matrix...")
    y_true = client._SecureDatasetClient__y_val
    y_pred = client._SecureDatasetClient__trained_model.predict(client._SecureDatasetClient__X_val)
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("💾 Saving trained model to 'secure_spam_model.joblib'...")
    client.save_model("secure_spam_model.joblib")

    print("\n✅ Done! You can now load and predict securely.")

    # Optional: show metadata preview (debugging/dev purpose only)
    print("\n📌 Dataset Preview:\n", initialization.get("preview").head())

if __name__ == "__main__":
    main()
