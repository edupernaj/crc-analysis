# `crc-analysis` python package for non-stationary CRC assessment

Cardiorespiratory coupling (CRC) is the reciprocal interaction between the cardiovascular and respiratory systems, reflecting a bidirectional synchronization of heart rate and breathing rhythms.  It is primarily mediated by the autonomic nervous system and brainstem networks, with respiratory sinus arrhythmia (RSA) being the most common manifestation, where heart rate increases during inspiration and decreases during expiration.

## Why was this package developed?

This package was started to analyse a series of experimental data collected within the framework of a HAW Hamburg Master’s thesis on **[leave blank for now]**. The research programme/objective was:

* We start with **raw 12-lead ECG data** recorded while people are simply breathing (https://drks.de/search/de/trial/DRKS00034187/details).
* The package **cleans and prepares the ECG data** (preprocessing), then extracts two useful signals: **RR intervals** (the time between heartbeats) and **ECG-derived respiration** (information about breathing contained in the ECG).
* We use **wavelet analysis** to look at these signals in both **time and frequency**, to see *which rhythms are happening, when they happen, and whether the two signals behave similarly*.
* From this, we calculate **wavelet coherence**, which tells us where heart and breathing rhythms show a consistent relationship in time and frequency.
* We then calculate **Cardiorespiratory Coupling (CRC)**, a measure of how strongly heart activity and breathing are coupled — in simple terms, *how closely are the heart and breathing “working together”?*
* This coupling is thought to reflect **parasympathetic/autonomic regulation** — the brain and body’s ability to coordinate basic functions such as heart rate and breathing. This coordination may become weaker during conditions such as stress or pain.
* The main question we want to answer is therefore: **does CRC decrease during pain, and if so, by how much?**

## What is this package?

This Python package takes **raw ECG data** and provides the complete processing pipeline: preprocessing and filtering, feature extraction, wavelet analysis, calculation of the **CRC metric**, and statistical analysis of the resulting CRC values. The goal is to provide a reproducible way to investigate how strongly **heart and breathing activity are coupled**.

## Use the package

To install the package along with its dependencies:
```sh
poetry install
```

To format and lint the code:
```sh
poetry run ruff format
poetry run ruff check
```

To run the tests:
```sh
poetry run pytest
```

To run the examples:

```sh
# Required data in data/ folder
```