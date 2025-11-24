
# Privacy Preserving Machine Learning Demo  
### Scenario 1: Encrypting the Patient Data Only

This project shows a simple example of how encrypted data can still be used for machine learning predictions. It is based on the idea from the research paper *Machine Learning in Precision Medicine to Preserve Privacy via Encryption*. The main idea is that the patient data is encrypted, the model is not encrypted, and the server never sees the real data.

This is a toy version for learning. It is not meant to be used in real medical settings.

## Project Layout

CS4371project/
    data/                fake dataset is saved here
    src/
        dataset.py       creates and loads fake data
        encrypt.py       TenSEAL encryption code
        model.py         logistic regression and encrypted math
        demo.py          runs the whole demo
    requirements.txt
    README.md

## How to Install Everything

Run these commands in the main project folder:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Windows users activate with:

.venv\Scripts\activate

## How to Run the Demo

You must run the code as a module so the imports work.

python -m src.demo

## What the Demo Shows

This project does three things:

1. Makes a random fake patient dataset
2. Trains a normal logistic regression model
3. Runs predictions on encrypted patient data

The encrypted version is slower and a little less accurate. This is normal because homomorphic encryption uses approximate math and large encrypted numbers.

Here is an example of what the results look like:

Plaintext Accuracy: 1.0  
Encrypted Accuracy: 0.8667

## How This Matches the Paper

The paper describes “Scenario 1,” where:

- The data is encrypted before being sent to the server
- The model stays plaintext
- The server computes the linear layer on encrypted data
- Only the final answer is decrypted by the user

This demo follows the same idea:

encrypt data -> encrypted dot product -> decrypt result -> prediction

## Why Accuracy Drops

CKKS does approximate calculations.  
When the encrypted dot product is done, the numbers are not exact.  
Small rounding errors can flip borderline predictions.

## Lessons Learned

- Encrypted inference works, but it is slower than normal machine learning.  
- CKKS is strong for privacy but not great for speed.  
- Even with noise, the encrypted model still gets similar accuracy.  
- It is possible to run machine learning predictions on encrypted data without ever seeing the real inputs.  
- A small toy example is the easiest way to understand how the paper’s method works.

## References

Original Paper:  
Machine Learning in Precision Medicine to Preserve Privacy via Encryption.

Foundational Paper:  
Craig Gentry, 2009. Fully Homomorphic Encryption Using Ideal Lattices.

Follow Up Work:  
Cheon et al., 2017. Homomorphic Encryption for Approximate Arithmetic (CKKS).

---

## Authors
Ariana Zapata  
Bella Havel  
Xavier Ortiz  
Rudy Rutiaga

---