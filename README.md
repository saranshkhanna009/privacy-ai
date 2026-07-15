# Privacy Preserving AI Framework

This framework is a modular, enterprise-ready pipeline designed to generate realistic synthetic datasets (such as sensitive medical or financial records) while mathematically guaranteeing privacy compliance.

By leveraging **Gaussian Copulas** and **PyTorch Autoencoders**, the system learns complex multidimensional distributions to output fictional, privacy-compliant data. An interactive Streamlit dashboard and a cryptographic privacy auditor ensure zero data leakage.

---

## Key Features

*   **Dual Generative Engines:** 
    *   *Statistical Copulas:* Captures linear and non-linear dependencies using multivariate Gaussian copula modeling.
    *   *Deep Learning:* A PyTorch-based autoencoder architecture designed for high-dimensional feature synthesis.
    *<img width="357" height="657" alt="privacy 4" src="https://github.com/user-attachments/assets/24fe931f-2bde-4798-b89a-d3cf18ef5041" />

    *   <img width="251" height="542" alt="privacy 5" src="https://github.com/user-attachments/assets/9d16ec2d-712d-4b53-aab4-c5bc8d7219de" />

*   **Privacy Auditor:** A validation suite running real-time privacy risk assessments, checking for exact duplicates and calculating **Distance to Closest Record (DCR)** to guarantee a safe distance between real and synthetic rows.
*   <img width="1442" height="740" alt="privacy 2" src="https://github.com/user-attachments/assets/2ccd3db9-a503-4f6a-8b77-25a1ee6ec812" />
<img width="1442" height="665" alt="privacy 3" src="https://github.com/user-attachments/assets/8efb9125-1de2-42bc-a77a-d92cd19235f3" />

*   **Streamlit UI:** A simple dashboard to upload sensitive CSVs, configure model parameters, and view compliance certificates dynamically.
<img width="1897" height="892" alt="privacy 1" src="https://github.com/user-attachments/assets/5427b3f7-c377-4591-9c92-05ed7806ec7f" />

---

## Software Architecture (OOP)

The codebase is built on clean, production-grade Object-Oriented Programming (OOP) principles:

*   **Inheritance:** An abstract base class (`BaseGenerativeModel`) enforces a uniform interface (`fit` and `generate`) across all engines.
*   **Polymorphism:** The frontend and orchestration pipeline treat both models identically. Swapping between statistical copulas and neural networks is entirely plug-and-play.
*   **Encapsulation:** Third-party library complexities (like `copulas` and `torch`) are wrapped within concrete classes, isolating dependencies.
*   **Separation of Concerns:** Split cleanly into logical directories: `core/` for base definitions, `models/` for ML/statistical engines, and `validation/` for auditing.

---

## Directory Structure

```text
privacy_ai_framework/
│
├── core/
│   └── base_model.py          # OOP interface (Abstract Base Class)
│
├── models/
│   ├── classical_statistics.py # Gaussian Copula Generative Model
│   └── deep_generative.py      # PyTorch Autoencoder Model
│
├── validation/
│   ├── privacy_auditor.py      # DCR and duplicate validation
│   └── utility_auditor.py      # Statistical similarity validation
│
├── app.py                      # Streamlit UI
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
