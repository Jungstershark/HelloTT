# Hello TT
[![GitHub license](https://img.shields.io/github/license/Jungstershark/HelloTT)](https://github.com/Jungstershark/HelloTT/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Jungstershark/HelloTT?style=social)](https://github.com/Jungstershark/HelloTT/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Jungstershark/HelloTT?style=social)](https://github.com/Jungstershark/HelloTT/network/members)


Welcome to the **Hello TT** chatbot project, developed for the Temasek X SUTD Gen AI Hackathon 2024. **Hello TT** is an innovative chatbot designed to assist users in resolving issues by utilizing textual information and screenshots. This chatbot creates a step-by-step guide to solving various issues, complemented by relevant screenshots for a more intuitive assistance experience.

## Features

- **Textual Issue Resolution**: Generates step-by-step guides based on textual information to resolve user queries.
- **Screenshot Integration**: Displays relevant screenshots alongside the guides to enhance understanding and provide visual assistance.
- **AI-Powered**: Utilizes OpenAI's GPT-3.5 and GPT-4-Vision models for advanced information processing and image recognition.
- **Retrieval Augmented Generation**: Employs a sophisticated mechanism to retrieve relevant information chunks from a user guide, ensuring accuracy and relevance in the solutions provided.
- **Streamlit UI**: Features an intuitive and user-friendly UI built with Streamlit, making it accessible for users with varying levels of technical expertise.

### Note

Due to NDA constraints, this chatbot has been sanitized of all training data used during the hackathon.

## Setup

Follow these steps to set up and run the **Hello TT** chatbot on your local machine.

### Prerequisites

- Python 3.9 or above

### Installation

1. **Clone the Repository**

    ```
    git clone <repository-url>
    ```
    Replace `<repository-url>` with the actual URL of this GitHub repository.

2. **Create a Virtual Environment**

    ```
    python -m venv venv
    ```

3. **Activate the Virtual Environment**

    - On Mac or WSL on Windows:

        ```
        source venv/bin/activate
        ```

    - On Windows (if not using WSL):

        ```
        .\venv\Scripts\activate
        ```

4. **Install Required Packages**

    ```
    pip install -r requirements.txt
    ```

5. **Set Up Environment Variables**

    Copy the `.env.example` file to `.env` and replace the placeholders with the actual values.

    ```
    cp -i .env.example .env
    ```

### Running the App

- To run the **Hello TT** chatbot:

    ```
    streamlit run hackathon.py
    ```

## Usage

![HelloTT_Front_Page](/screenshots/HelloTT_Front_Page.png)

![HelloTT_Feedback](/screenshots/HelloTT_Feedback.png)


