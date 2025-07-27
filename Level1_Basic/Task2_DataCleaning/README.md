# README: Understanding Negative Values in Standardized Iris Dataset Features

## Overview

When working with the Iris dataset, you might notice that after preprocessing, some feature values become negative. This README explains why this happens and what it means.

## Why are there negative values?

The Iris dataset contains measurements like sepal length, sepal width, petal length, and petal width. These features are originally on different scales and ranges. To prepare the data for many machine learning algorithms, it is common to **standardize** the features.

### What is standardization?

Standardization (also called Z-score normalization) transforms each feature to have:
- A **mean** of 0
- A **standard deviation** of 1

This is done using the formula:

\[
z = \frac{x - \mu}{\sigma}
\]

Where:
- \(x\) = original value
- \(\mu\) = mean of the feature values across the dataset
- \(\sigma\) = standard deviation of the feature values across the dataset

### Effect of standardization on values

- Since the mean \(\mu\) is subtracted from each value, data points below the mean become **negative**.
- Data points above the mean become **positive**.
- Values exactly at the mean become zero.

## Why standardize?

- It makes all features comparable on the same scale.
- It helps algorithms that rely on distances (e.g., k-NN, SVM) perform better.
- It prevents features with large ranges from dominating those with smaller ranges.
- It improves numerical stability and convergence speed in optimization algorithms.

## Summary

- Negative values appear after standardization because the data is centered around zero.
- This is a normal and expected step in data preprocessing.
- It improves the quality and fairness of machine learning models.
