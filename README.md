# Hybrid Quantum-Classical Classifier on AWS

This was my individual assignment for the Theory and Practice of Advanced AI Ecosystems module (CS5024) at the University of Limerick, April 2026.

## What this project is about

After a university talk that mentioned AWS Braket, I wanted to find out two things: can a quantum machine learning model run inside a normal cloud setup, and is it any good compared with a standard classifier? I used the Wisconsin Breast Cancer dataset, trained a classical SVM and a small quantum circuit on it, ran everything in Amazon SageMaker, and priced what real quantum hardware would cost.

## Data

- Wisconsin Breast Cancer dataset from scikit-learn: 569 tumours, 30 measurements each, labelled malignant or benign
- Stratified 80/20 split with seed 42: 455 training and 114 test samples (42 malignant, 72 benign)
- All features standardised

## Models

**Classical baseline.** An SVM with an RBF kernel (scikit-learn defaults) using all 30 features.

**Quantum model.** A 4-qubit variational quantum circuit built with PennyLane:

- PCA reduces the 30 features to 4 (keeping 79.6% of the variance), and each one is scaled to the range 0 to pi
- Each feature rotates one qubit with an RY gate (angle encoding)
- 3 layers of general rotation gates on every qubit, each followed by a ring of CNOT gates that entangle the qubits
- The output is the Pauli-Z expectation value on qubit 0 plus a bias, which gives 37 trainable parameters
- Trained with the COBYLA optimiser from SciPy (60 function evaluations) on a mean squared error loss

The circuit ran on PennyLane's `default.qubit` simulator inside SageMaker. I planned to use the Braket simulator, but a library version conflict (antlr4) in the SageMaker environment stopped it from working.

## Results (114 test samples)

| | SVM (30 features) | Quantum circuit (4 features) |
|---|---|---|
| Accuracy | 98.25% | 84.21% |
| Malignant tumours detected | 41 of 42 | 24 of 42 |
| Benign tumours correctly cleared | 71 of 72 | 72 of 72 |

The quantum model used 7.5 times fewer input features but was clearly less accurate. Most importantly, it labelled 18 of the 42 malignant tumours as benign, which is the dangerous kind of mistake in a medical setting.

**Correction to my original report:** the report said the quantum model reached "perfect recall". That was wrong. In scikit-learn's version of this dataset, benign is label 1, so the recall of 1.0 was for benign tumours, not malignant ones. The table above shows the correct picture. The results file also labels the backend as "AWS Braket Local Simulator", but the code actually used PennyLane's `default.qubit`.

## AWS setup

- **SageMaker Studio** (ml.t3.medium, Python 3.11) ran the notebook
- **S3** stored the results file, the charts and the trained circuit weights
- **CloudWatch** stored custom accuracy metrics, with an alarm that emails through **SNS** if accuracy drops below 0.80
- **IAM** controlled which services could access each other. I attached the broad Braket full-access policy to get it working; in a real deployment I would narrow it down
- **AWS Budgets** alerted me at 85% and 100% of a $15 monthly budget

## Cost of real quantum hardware

I used the AWS Pricing Calculator to price the same training workload (1,200 circuit evaluations a month at 1,000 shots each) on real quantum hardware. It came to about $145,104 a month, roughly $1.74 million a year. That is why I recommend developing on simulators and only using real hardware for final checks.

## How to run

The notebook expects to run in Amazon SageMaker Studio with an IAM role that can write to S3 and CloudWatch. Set your own S3 bucket name in the configuration cell.

```
pip install -r requirements.txt
```

Then open `qml-implementation.ipynb` and run the cells in order.

## Project structure

```
.
├── qml-implementation.ipynb
├── experiment_results.json
├── quantum_weights.npy
├── quantum_results.png
├── requirements.txt
└── README.md
```

## Limitations

- The comparison is not equal: 30 features for the SVM and 4 for the quantum model, on a single split with one seed
- The quantum circuit was probably undertrained: the loss only fell from 0.71 to 0.65
- Nothing ran on real quantum hardware
- I wrote some deployment ideas (SageMaker endpoint, Lambda, API Gateway), but nothing was deployed

## Tools used

Python, PennyLane, scikit-learn, NumPy, SciPy, boto3, Amazon SageMaker, S3, CloudWatch, SNS, IAM, AWS Budgets.
