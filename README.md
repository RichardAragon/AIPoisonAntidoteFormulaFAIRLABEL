# AIPoisonAntidoteFormulaFAIRLABEL

This repository is a Python module that implements an algorithm to detect and correct biases in labelled data, based on the research paper "FAIRLABEL: Correcting Bias in Labels" by Srinivasan H Sengamedu and Hien Pham from Amazon.

The module defines a class called FAIRLABEL, which takes the data, labels, and groups as input, and trains a machine learning model to predict the probability that a label is poisoned. The class also provides methods to detect and correct the poisoned labels using various techniques, such as majority voting or averaging.

The module also includes a function called get_bootstrap_confidence, which estimates the confidence interval for the false positive rate (FPR) using the bootstrap method. The function takes the FPR function, the probabilities, the threshold, the alpha level, and the number of bootstrap samples as input, and returns the lower and upper bounds of the confidence interval.

## Installation

To install FAIRLABEL, you can use pip:

```bash
pip install fairlabel
```

## Usage

To use this repository, you need to import the module and instantiate the class with the data, labels, and groups. For example:

```python
import numpy as np
from fairlabel import FAIRLABEL

# Create test set
X = np.arange(100)
Y = np.sin(X) + np.zeros(shape=(100, ))
Z = Y < 0
labels = Z * 2 + (~Z) * (-2)

# Instantiate FAIRLABEL class
fairlabel = FAIRLABEL(data=X, labels=labels, groups=Z)
```

To detect the poisoned labels, you can use the detect_bias method:

```python
# Detect poisoned labels
poisoned_labels = fairlabel.detect_bias(labels)
```

To correct the poisoned labels, you can use the correct_bias method:

```python
# Correct poisoned labels
corrected_labels = fairlabel.correct_bias(poisoned_labels)
```

To estimate the confidence interval for FPR, you can use the get_bootstrap_confidence function:

```python
# Define parameters
threshold = 0.5
alpha = 0.95
b = 1000

# Estimate the confidence interval for FPR
lower_ci, upper_ci = fairlabel.get_bootstrap_confidence(fairlabel.fpr, fairlabel.probabilities, threshold, alpha, b)
```

## License

This repository is released under the MIT Open Source license. See [LICENSE](https://github.com/user/fairlabel/blob/main/LICENSE) for more details.

## References

- Sengamedu, S. H., & Pham, H. (2021). FAIRLABEL: Correcting Bias in Labels. arXiv preprint arXiv:2103.12264. https://arxiv.org/abs/2103.12264

I hope this text is helpful for you. Please let me know if you have any questions or comments about it. 😊
