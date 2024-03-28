 E-Commerce Recommendation System using Intel OpenVINO™

## Overview

This repository contains an e-commerce recommendation system built using Intel OpenVINO™ toolkit. The system leverages deep learning models for product recommendations, enabling personalized shopping experiences for users.

## Features

- **Recommendation Engine**: Uses neural networks to analyze user behavior and recommend relevant products.
- **Intel OpenVINO™ Integration**: Optimizes deep learning inference across Intel® hardware (including accelerators) for maximum performance.
- **Scalability**: Supports heterogeneous execution across CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA.
- **oneAPI**: Provides a API for CNN-based deep learning inference.

## Prerequisites

- Python 3.x
- Intel OpenVINO™ toolkit (Download from [here](https://www.intel.com/content/dam/develop/public/us/en/include/openvino-download-ih/selector-0290a24.html))

## Installation

1. Clone this repository:

    ```
    git clone https://github.com/TECHBROS00/TechBros.git
    cd TechBros
    ```

2. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Download pre-trained recommendation models from Intel® Model Zoo and convert them to OpenVINO™ format using the Model Optimizer.

## Usage

1. Run the recommendation system:

    ```
    python recommendation_system.py
    ```

2. Access the recommendation results through the provided API endpoints.




