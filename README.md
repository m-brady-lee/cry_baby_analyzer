CRY BABY ANALYZER

A machine learning model for predicting the reason behind baby cries using audio analysis. This project was developed as a proof-of-concept for integrating AI into baby care products, offering support to new parents by identifying potential causes for their baby's distress.


PROJECT OVERVIEW

Babies-R-Us (BRU) launched the Cry Baby Analyzer (CBA) pilot to regain market share by offering an innovative tool to help parents understand their baby's cries. The goal was to create a multi-input neural network capable of classifying baby cries based on audio data into categories like "Hungry," "Tired," "Pain," etc.


KEY RESULTS

-Multi-input deep learning model with Mel Spectrogram and MFCC feature inputs

-51.9% accuracy on pilot dataset using K-Fold Cross Validation

-End-to-end preprocessing and prediction pipeline

-Audio augmentation for real-world variability

-Working proof-of-concept with future scalability


FILE STRUCTURE

cry-baby-analyzer/

├── cry_baby_analyzer.ipynb       # Main notebook with preprocessing, modeling, and predictions

├── cry_baby_presentation.pdf     # Presentation of findings

├── prediction/                   # Sample audio files for single prediction

├── data_raw/                     # Original recordings (reference only)

├── data_categorized.split/       # Trimmed and labeled 10s clips

├── data_wav-converted/           # Augmented WAV files (generated during notebook run)

├── scalers/                      # Pickled scalers for feature normalization

├── encoders/                     # Pickled encoders for label decoding

└── README.md                     # Project overview and usage instructions


DATASET

Source: 79 manually labeled audio samples from employees and their families

Augmented: 5x increase using pitch, volume, noise, and time-shift techniques

Final Size: 1,116 audio clips

Reasons: Hungry, Tired, Unease, Discomfort, Pain

Solutions: Feed, Sleep, Adjust, Soothe


MODEL ARCHITECTURE

A multi-input neural network combining:

  -Convolutional Neural Network (CNN) for Mel Spectrograms
  
  -Fully Connected Neural Network (FCNN) for MFCC features

Both branches merge into dense layers and predict the cry reason. The solution label was excluded due to lower model accuracy when dual targeting.


TECHNOLOGIES

TensorFlow / Keras

Librosa

Scikit-learn

NumPy, Matplotlib, Pickle


PERFORMANCE

Accuracy: 51.9% (K-Fold Cross Validation)

Goal: >30% for pilot success

Confusion Matrix Insight: Model performs best with “Unease”; common confusion between “Hungry” and “Tired”


INSTALLATION & USAGE

Requirements

  Python 3.8+
  
  ffmpeg installed and added to system PATH

Setup

  -Clone the repository (in Bash)
  
  git clone https://github.com/your-username/cry-baby-analyzer.git    
  
  cd cry-baby-analyzer
  
  -Install dependencies (in Bash)
   
  pip install numpy tensorflow librosa matplotlib scikit-learn pickle-mixin jupyterlab
  
  -Install ffmpeg:
  
  Download ffmpeg (https://ffmpeg.org/)  
  
  Extract and add to your PATH
  
  -Launch the app (in Bash):
  
  jupyter lab
  
  -Open cry_baby_analyzer.ipynb and click 'Run All Cells'.


MAKING PREDICTIONS

To experiment with making a new prediction:

  -In Jupyter Notebooks, scroll down to the bottom section called ‘Single Prediction’.
  
  a. Under ‘Choose Audio File to Predict’, change the audio file to get a new prediction.
  
  -In the Project File that you downloaded, open the ‘prediction’ folder for a list of audio files to experiment with. 
  
  -Copy and paste a different audio file into the audio_file_path variable on Jupyter Notebooks.
  
  Important: When you copy and paste the new file name, make sure you only paste over the file name and not over the ‘prediction’ folder path (i.e. Only replace '2.1.3 Hungry-Feed.wav' in the following: Ex. audio_file_path = 'prediction/2.1.3 Hungry-Feed.wav')
  
  -After uploading a new file in the audio_file_path variable, run the cell (Shift + Return)
  
  -Scroll to the bottom of the cell to view prediction probabilities and category output.


ROADMAP

Next Phase Goals:

  Scale dataset to 10,000+ samples from 100+ babies
  
  Improve accuracy through better feature extraction and labeling
  
  Integrate with mobile apps, baby monitors, and wearable devices
  
  License model to third-party vendors


TIMELINE SUMMARY

Sprint	Date	          Milestone

1	      Jul 29–Aug 4	  Proof of concept

5	      Aug 26–Sep 1	  Data collection, model training begins

9–11	  Sep 23–Oct 13	  Final data prep, model retraining

13	    Oct 21–27	      Deployment


AUTHOR

Michael Lee

REFERENCES

TensorFlow & Keras Documentation

Librosa for Audio Analysis

CRISP-ML Methodology

FFmpeg
Scikit-learn Multilabel Binarizer
