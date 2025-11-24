# encrypt.py
#
# Makes TenSEAL CKKS context + simple encrypted math.
# This version includes the "paper-style" encrypted linear layer.

import tenseal as ts


def make_context():
    """
    Makes CKKS encryption.
    These params work fine for demo (not production).
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 20, 20, 40]
    )
    ctx.global_scale = 2**20
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    print("CKKS context ready!")
    return ctx


def encrypt_vec(ctx, arr):
    """Encrypts one vector (like the patient data)."""
    return ts.ckks_vector(ctx, arr)


def decrypt_scalar(enc_vec):
    """Decrypts a CKKS vector with 1 value."""
    plain = enc_vec.decrypt()
    return float(plain[0])


# ⭐ NEW: “Paper-Style Encrypted Linear Layer”
def encrypted_linear_layer(enc_x, w_plain, b_plain):
    """
    This is the simplified version of the math in the paper.

    Paper does:
       Enc(x) * w  +  b
    where x is encrypted, w and b are plaintext.
    
    TenSEAL automatically supports:
       ciphertext * plaintext  -> ciphertext
    """
    # encrypted dot product + plaintext bias
    enc_result = enc_x.dot(w_plain) + b_plain
    return enc_result