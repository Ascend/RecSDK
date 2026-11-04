# Introduction

## What Is Rec SDK

**Product Background**

As artificial intelligence (AI) technology evolves, industries such as e-commerce, long- and short-form video, and social media increasingly demand effective search, recommendation, and advertising (SRA) systems. The rapid growth of the Internet has led to a massive increase in user data, product information, and video content, underscoring the importance of effective SRA systems. The growing demand for such systems inevitably increases the need for computing power. A primary focus for practitioners is how to deploy more computing power and fully utilize its potential.

**Definition**

Rec SDK offers an **SDK for search, recommendation, and advertising services in the Internet market**. It offers a framework for these services based on the Ascend platform to meet related model training requirements, thus supporting large-scale SRA scenarios and facilitating efficient training of models for such scenarios.

**Product Value**

**Table 1** Product value

|Product Feature|Product Value|
|--|----|
|Ease of use|Build algorithm models quickly using minimalist APIs.|
|Accuracy|Achieve less than 0.01% accuracy error in standard model verification.|
|Performance|Maximize performance with efficient multi-level pipeline acceleration, high-speed collective communication, and extreme optimization.|

## Component Overview

Rec SDK provides sparse recommendation frameworks based on TensorFlow and PyTorch. Based on how computing tasks are distributed between the CPU and NPU, the frameworks support two modes: Non-fully-offloaded and Fully-offloaded.

- **Non-fully-offloaded**: a hybrid mode in which some computing tasks run on the NPU and others run on the CPU
- **Fully-offloaded**: a mode in which all computing tasks are offloaded to the NPU

The components are as follows:

| Component | Adaptation Status | Description |
| --- | --- | --- |
| tf_rec_v1 | Non-fully-offloaded | Non-fully-offloaded sparse recommendation framework based on TensorFlow, adapted for NPU devices |
| tf_rec_v2 | Fully-offloaded | Fully-offloaded sparse recommendation framework based on TensorFlow, adapted for NPU devices |
| torch_rec_v1 | Non-fully-offloaded | Non-fully-offloaded sparse recommendation framework based on the open-source PyTorch and TorchRec software, adapted for NPU devices |
| torch_rec_v2 | Fully-offloaded | Fully-offloaded sparse recommendation framework based on the open-source PyTorch and TorchRec software, adapted for NPU devices |
