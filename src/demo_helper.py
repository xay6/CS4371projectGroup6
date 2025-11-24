# demo_helper.py
#
# Simple helper script for live presentation.
# Shows:
# 1. dataset preview
# 2. encryption working
# 3. encrypted linear math
# 4. encrypted prediction using the trained model
#
# Run with:
#   python -m src.presentation_helper

from src.dataset import load_data
from src.encrypt import make_context, encrypt_vec, decrypt_scalar, encrypted_linear_layer
from src.model import SimpleEncryptedLogReg
import numpy as np

def main():
    print("=== Presentation Helper Demo ===")

    # -------------------------------------------------------
    # 1. Dataset Preview
    # -------------------------------------------------------
    print("\n[1] Small Dataset Preview:")
    X_train, X_test, y_train, y_test = load_data()
    for i in range(3):
        print("Sample", i, ":", X_train[i], "Label:", y_train[i])

    # -------------------------------------------------------
    # 2. Train Model (plaintext)
    # -------------------------------------------------------
    print("\n[2] Training Plaintext Logistic Regression...")
    model = SimpleEncryptedLogReg()
    model.train(X_train, y_train)
    print("Model trained successfully.")

    # -------------------------------------------------------
    # 3. Create Encryption Context
    # -------------------------------------------------------
    print("\n[3] Setting up CKKS Encryption...")
    ctx = make_context()

    # -------------------------------------------------------
    # 4. Encrypt one test sample
    # -------------------------------------------------------
    test_sample = X_test[0]
    print("\n[4] Test Input (plaintext):", test_sample.tolist())

    enc_x = encrypt_vec(ctx, test_sample.tolist())
    print("Encrypted Input Vector:")
    print(enc_x)

    # -------------------------------------------------------
    # 5. Run encrypted linear math like the paper
    # -------------------------------------------------------
    w = model.model.coef_[0]
    b = float(model.model.intercept_[0])

    print("\n[5] Running encrypted dot product and adding bias...")
    enc_output = encrypted_linear_layer(enc_x, w.tolist(), b)
    print("Encrypted Output:")
    print(enc_output)

    # -------------------------------------------------------
    # 6. Decrypt result
    # -------------------------------------------------------
    decrypted_value = decrypt_scalar(enc_output)
    print("\n[6] Decrypted Linear Output:", decrypted_value)

    # -------------------------------------------------------
    # 7. Final Prediction using sigmoid
    # -------------------------------------------------------
    prob = 1 / (1 + np.exp(-decrypted_value))
    pred = 1 if prob >= 0.5 else 0

    print("\n[7] Final Prediction (decrypted):")
    print("Probability:", prob)
    print("Predicted Label:", pred)

    print("\n=== Demo Complete. ===")


if __name__ == "__main__":
    main()