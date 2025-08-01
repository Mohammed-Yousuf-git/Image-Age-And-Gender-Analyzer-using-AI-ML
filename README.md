# 🤖 Real-time Age & Gender AI Analyzer

An interactive web application built with **Streamlit**, using deep learning to predict **age group** and **gender** from uploaded face images. The app utilizes a custom-trained Keras/TensorFlow model for age group classification and DeepFace for robust gender analysis.

## Features

- **Easy-to-use web interface:** Upload a face image and instantly receive predictions.
- **Age group prediction:** Custom-trained model classifies into `YOUNG`, `MIDDLE`, or `OLD`.
- **Gender prediction:** Powered by DeepFace, leveraging state-of-the-art facial recognition backends.
- **Visual confidence charts:** Bar plot for age probabilities, pie chart for gender probabilities.
- **Dataset stats & feedback:** See dataset class distribution and send in-app feedback.
- **Modern, responsive UI:** Custom CSS and icons for a polished, AI-focused experience.

## Quick Start

1. **Install requirements:**

   ```bash
   pip install streamlit tensorflow keras deepface opencv-python pillow matplotlib pandas
   ```

2. **Ensure model and dataset are placed at:**
   - Model: `/Users/usufahmed/Desktop/gender_app/faces/models/age_classifier.keras`
   - Dataset CSV: `/Users/usufahmed/Desktop/gender_app/faces/train.csv`
   - Dataset images: `/Users/usufahmed/Desktop/gender_app/faces/Train/`

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

4. **Upload a face photo to get predictions!**

## Project Structure

- `app.py`  – Main Streamlit application.
- `models/age_classifier.keras` – Pretrained age classification model.
- `faces/train.csv` – Sample CSV for dataset statistics.
- `faces/Train/` – Sample images for stats display.

## Age Model Info

- Backbone: MobileNetV2 (transfer learning)
- Custom head: GlobalAveragePooling, Dense, Softmax
- Trained to classify into: `YOUNG`, `MIDDLE`, `OLD`

## Gender Model Info

- Uses DeepFace (hybrid CNNs, ensemble for gender)
- Downloads and caches weights on first use


## Acknowledgements

- Age prediction: Custom Keras/TensorFlow pipeline.
- Gender detection: [DeepFace](https://github.com/serengil/deepface).
- UI: [Streamlit](https://streamlit.io/), Matplotlib, custom CSS.

## License

This project is intended for academic and demonstration purposes only.

**Tip:** For best results, use clear, frontal face images with good lighting. Avoid occlusions (sunglasses, hats, masks).
### 🙋‍♂️ Author
 Mohammed Yousuf
AI Engineering Student | Passionate about Machine Learning, Computer Vision, and Real-World Applications 🚀
Feel free to reach out or contribute!
### 📄 License
This project is licensed under the MIT License.
### 🌟 Star this repository
If you found this helpful, give it a ⭐ on GitHub!

