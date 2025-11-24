# model.py
#
# High-school style Logistic Regression model
# and "Scenario 1" encrypted inference.
#
# This version now calls encrypted_linear_layer()
# which matches the paper's MLE math.

import numpy as np
from sklearn.linear_model import LogisticRegression
from .encrypt import encrypt_vec, decrypt_scalar, encrypted_linear_layer


class SimpleEncryptedLogReg:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        self.model.fit(X, y)
        print("Model trained!")

    def predict_plain(self, X):
        return self.model.predict(X)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_one_encrypted(self, ctx, x):
        """
        This reproduces Scenario 1 from the paper:
        
        - encrypt patient data
        - compute encrypted linear layer  (Enc(x·w + b))
        - decrypt result (client side)
        - apply sigmoid + threshold
        """

        x = np.array(x, dtype=float)
        w = self.model.coef_[0]      # plaintext
        b = float(self.model.intercept_[0])

        # Encrypt the input vector (this matches paper's Step 1)
        enc_x = encrypt_vec(ctx, x.tolist())

        # ⭐ Use the "paper-style encrypted math"
        enc_logit = encrypted_linear_layer(enc_x, w.tolist(), b)

        # Decrypt like the client would
        logit = decrypt_scalar(enc_logit)
        prob = self._sigmoid(logit)
        label = 1 if prob >= 0.5 else 0

        return label, prob

    def predict_encrypted_batch(self, ctx, X):
        labs, probs = [], []
        for row in X:
            lab, p = self.predict_one_encrypted(ctx, row)
            labs.append(lab)
            probs.append(p)
        return np.array(labs), np.array(probs)