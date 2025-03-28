# Fake News & Deepfake Detection

# Overview

In today's digital landscape, misinformation is a major threat, influencing public opinion, undermining trust, and causing severe societal consequences. Our project aims to combat this issue by leveraging AI-driven solutions for detecting both textual fake news and deepfake videos.

# Features

 Text-Based Fake News Detection : Uses an AI model to analyze headlines and classify them as real or fake.

Deepfake Video Detection: Identifies manipulated video content using state-of-the-art deep learning techniques.

Streamlit Frontend: A user-friendly interface allowing users to check news headlines and uploaded videos.

Confidence Score: Provides a probability score indicating the reliability of the result.

Predefined Examples: Showcases sample cases of real and fake news for better understanding.

# Approach

Our detection system follows a dual approach:

Textual Analysis: Machine learning models analyze linguistic patterns to identify misinformation.

Video Content Analysis: Deep learning models detect manipulated videos by evaluating frame inconsistencies.

# Technical Stack

Natural Language Processing (NLP): Analyzing and classifying news headlines.

Deep Learning Models: Used for both text and deepfake detection.

Streamlit: Frontend framework for an interactive user experience.

Python (Jupyter Notebook): Core development and training environment.

# Implementation Details

BiLSTM Model: Used for fake news detection.

# Performance Metrics

Fake News Detection:

Accuracy: 86.88%

Precision: 62.89%

Recall: 72.47%

F1-score: 72.68%


# How to Use

Clone the Repository

``` 
git clone https://github.com/your-repo-name.git
cd your-repo-name 

```

Install Dependencies

```
pip install -r requirements.txt
```
Run the Streamlit App
```
streamlit run app.py
```
Usage

Enter a news headline or upload a video to analyze its authenticity.

Future Enhancements

Improve deepfake detection accuracy.

Enhance scalability with distributed deployment.

Optimize text processing for real-time analysis.
