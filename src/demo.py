# demo.py
#
# Runs:
# - data loading
# - training the LR model
# - plaintext predictions
# - encrypted predictions using the "paper-style" layer
# Prints times + accuracies for slides.

import time
import numpy as np

from dataset import load_data
from encrypt import make_context
from model import SimpleEncryptedLogReg


def acc(y, yp):
    return (y == yp).mean()


def main():
    print("=== Encryption + Precision Medicine Demo (Scenario 1) ===")

    X_train, X_test, y_train, y_test = load_data()
    print("Train:", X_train.shape[0], "Test:", X_test.shape[0])

    # Train model
    model = SimpleEncryptedLogReg()
    t0 = time.time()
    model.train(X_train, y_train)
    train_time = time.time() - t0

    # Plain predictions
    t1 = time.time()
    yp_plain = model.predict_plain(X_test)
    plain_time = time.time() - t1
    plain_acc = acc(y_test, yp_plain)

    print("\n--- Plaintext Results ---")
    print("Accuracy:", round(plain_acc, 4))
    print("Prediction time:", round(plain_time, 4), "sec")

    # Encryption setup
    ctx = make_context()

    # Encrypted predictions (small subset)
    X_small = X_test[:30]
    y_small = y_test[:30]

    print("\nRunning encrypted predictions on 30 samples...")
    t2 = time.time()
    yp_enc, _ = model.predict_encrypted_batch(ctx, X_small)
    enc_time = time.time() - t2
    enc_acc = acc(y_small, yp_enc)

    print("\n--- Encrypted Results (Scenario 1) ---")
    print("Accuracy:", round(enc_acc, 4))
    print("Prediction time:", round(enc_time, 4), "sec")

    print("\n=== SUMMARY ===")
    print("Plain Acc:", round(plain_acc, 4))
    print("Enc Acc:", round(enc_acc, 4))
    print("Plain Time:", round(plain_time, 4), "sec")
    print("Enc Time:", round(enc_time, 4), "sec")


if __name__ == "__main__":
    main()