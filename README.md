# Automated Customer Reviews 🤖📊

This project automates the analysis of customer reviews using advanced Natural Language Processing (NLP) techniques. The goal is to classify reviews, cluster product categories, and generate recommendation summaries based on customer feedback. 🚀

## Project Overview 📋

1. **Review Classification**: Classifying reviews into predefined categories using a `DistilBERT` model. 
2. **Product Category Clustering**: Grouping product categories into broader categories using the `K-means` algorithm and `DistilBERT` embeddings to extract key features. 
3. **Recommendation Summaries Generation**: Generating recommendation summaries by `gpt-3.5-turbo` based on sentiment analysis of reviews, highlighting both positive and negative aspects. 

## File Descriptions 📂

Here are the key files included in the project:

1. **`app.py`**: The main application file using Streamlit, allowing users to interactively upload data and view classification, clustering, and summary results. 
   
2. **`data_cluster.csv`**: A CSV file containing the clustered product data, used in the K-means clustering process to group products into broader categories. 
   
3. **`kmeans_model.pkl`**: The trained K-means model saved for predicting cluster assignments for new data. 
   
4. **`Project_NLP_Notebook.ipynb`**: A Jupyter notebook that contains the code for NLP tasks, model training, and data preprocessing. 
   
5. **`Final_Output_all_blog_posts.txt`**: A text file containing the final output of blog posts or review summaries generated from the analysis. 

6. **`Presentation (2).pdf`**: A PDF presentation containing an overview of the project, methodologies used, and results.

7. **`requirements.txt`**: A file listing the dependencies and libraries required to run the project. Install them by running `pip install -r requirements.txt`.

## Model Files 🎯

Due to file size limitations, the trained models are hosted on Google Drive. You can download them from the following links:

- [best_sentiment_Model](https://drive.google.com/drive/folders/1_xZSgu_vHhaew1vGXSbx0iCJVFwX_hc1?usp=drive_link) 
- [best_dislibret_Model](https://drive.google.com/drive/folders/1_xZSgu_vHhaew1vGXSbx0iCJVFwX_hc1?usp=sharing)

  ## How to Run the Project 🚀

Follow these steps to set up and run the project on your local machine:

### Step 1: Clone the Repository

First, clone the repository to your local machine using Git:

```bash
git clone <repository_url>
cd <repository_directory>
```
### Step 2: Install Dependencies
Install the required dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```
### Step 3: Download the Pre-trained Models
The pre-trained models are hosted on Google Drive. Download them using the following links:
- [best_sentiment_Model](https://drive.google.com/drive/folders/1_xZSgu_vHhaew1vGXSbx0iCJVFwX_hc1?usp=drive_link) 
- [best_dislibret_Model](https://drive.google.com/drive/folders/1_xZSgu_vHhaew1vGXSbx0iCJVFwX_hc1?usp=sharing)

Once downloaded, place them in the project directory.

### Step 4: Run the Application
Run the Streamlit application using the following command:
```bash
streamlit run app.py
```
This will open a web interface where you can upload data and interact with the classification, clustering, and recommendation results.
