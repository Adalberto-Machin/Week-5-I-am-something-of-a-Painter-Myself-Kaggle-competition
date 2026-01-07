# Monet Style Transfer with CycleGAN 🎨

This project implements a **CycleGAN** architecture to translate real-world photos into the artistic style of Claude Monet. Developed for the Kaggle competition ["I’m Something of a Painter Myself"](https://www.kaggle.com/competitions/gan-getting-started), the model learns the unpaired mapping between the photo domain and the Monet domain without needing one-to-one matched examples.
Note that this is also part of the Week 5 assigment of CU Boulder Intro to Deep learning Coursera class.
Author: Adalberto Machin


## Project Overview

* **Goal:** Generate 7,000+ images in the style of Monet from a given set of photos.
* **Architecture:** CycleGAN (Generative Adversarial Network)
    * **Generators:** U-Net based with Instance Normalization.
    * **Discriminators:** PatchGAN classifier.
* **Loss Functions:**
    * Adversarial Loss (Least Squares / MSE)
    * Cycle Consistency Loss (L1)
    * Identity Loss (L1)
* **Framework:** PyTorch (Backbone) with TensorFlow (Data Loading).
* **Score:** Achieved a **MiFID score of 70.40** on the Kaggle Leaderboard.

## 🧠 Model Architecture

The CycleGAN consists of two generator networks and two discriminator networks playing a Min-Max adversarial game:

1.  **Generator $G$ (Photo $\to$ Monet):** Transforms a photo into a Monet painting.
2.  **Generator $F$ (Monet $\to$ Photo):** Transforms a Monet painting back into a photo.
3.  **Discriminator $D_X$ (Monet):** Distinguishes between real Monet paintings and generated ones.
4.  **Discriminator $D_Y$ (Photo):** Distinguishes between real photos and generated ones.

### Key Features
* **Cycle Consistency:** Ensures that if we translate an image to Monet style and back, we recover the original image ($F(G(x)) \approx x$). This prevents the model from hallucinating random content.
* **Identity Mapping:** Ensures that if we feed a real Monet into the Monet generator, it remains unchanged ($G(y) \approx y$). This preserves the color palette.
* **Instance Normalization:** Used instead of Batch Normalization to better preserve the unique artistic style of individual images.

## 🛠️ Setup & Installation

### Dependencies
* Python 3.7+
* PyTorch (CUDA enabled recommended)
* TensorFlow (for TFRecord data loading)
* Matplotlib, NumPy, PIL

### Data Structure
The project uses the standard Kaggle dataset format:
* `monet_jpg/`: 300 paintings by Claude Monet.
* `photo_jpg/`: 7,038 real-world photos.

## 🚀 Training Details

* **Image Size:** 256x256
* **Epochs:** 20 (Baseline)
* **Optimizer:** Adam (`lr=2e-4`, `beta1=0.5`)
* **Loss Weights:**
    * Cycle Weight: $\lambda_{cycle} = 10$
    * Identity Weight: $\lambda_{identity} = 5$

### EDA & Preprocessing
* **Pixel Intensity Analysis:** Verified that Monet images have a distinct "blue shift" in the histograms compared to photos, confirming the domain gap.
* **Normalization:** All images are normalized to the range $[-1, 1]$.

## 📈 Results

* **Quantitative:** The model achieved a **MiFID score of 70.40**, indicating successful style transfer with no mode collapse.
* **Qualitative:**
    * Successfully captured the impressionist brushstroke texture.
    * Learned to shift the color palette towards Monet's signature cool tones (blues/greens) in shadow areas.
    * *Limitation:* Some artifacts ("static") observed in extremely dark/nighttime photos due to the lack of dark examples in the Monet dataset.

## 🔮 Future Improvements

To improve the score further (target: <50), the following techniques are planned:
1.  **Data Augmentation:** Implement random horizontal flips and random cropping to prevent overfitting to the small 300-image Monet dataset.
2.  **Learning Rate Scheduling:** Use a linear decay of the learning rate starting from Epoch 25 to stabilize the weights.
3.  **ResNet Generator:** Switch from U-Net to a ResNet-based generator (6 or 9 blocks) to better separate content from style.
4.  **Replay Buffer:** Implement an image buffer to train the discriminator on a history of generated images, reducing oscillation.

## 📜 License
This project is open-source and available under the MIT License.
