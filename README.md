# Automated Customer Reviews 🤖📊

This project automates the analysis of customer reviews using advanced Natural Language Processing (NLP) techniques. The goal is to classify reviews, cluster product categories, and generate recommendation summaries based on customer feedback. 🚀

## Project Overview 📋

1. **Review Classification**: Classifying reviews into predefined categories using a `DistilBERT` model. 
2. **Product Category Clustering**: Grouping product categories into broader categories using the `K-means` algorithm and `DistilBERT` embeddings to extract key features. 
3. **Recommendation Summaries Generation**: Generating recommendation summaries based on sentiment analysis of reviews, highlighting both positive and negative aspects. 

## Technologies Used ⚙️

- **Python**: The primary language for this project. 
- **Transformers**: For using pre-trained models like `DistilBERT` for text classification.
- **NLPAug**: Used for augmenting the dataset to improve the model's performance.
- **K-means Clustering**: Applied for clustering similar product categories into broader groups. 
- **Streamlit**: To create an interactive web interface for displaying the classification, clustering, and summary results. 

## File Descriptions 📂

Here are the key files included in the project:

1. **`app.py`**: The main application file using Streamlit, allowing users to interactively upload data and view classification, clustering, and summary results. 
   
2. **`data_cluster.csv`**: A CSV file containing the clustered product data, used in the K-means clustering process to group products into broader categories. 
   
3. **`kmeans_model.pkl`**: The trained K-means model saved for predicting cluster assignments for new data. 
   
4. **`Project_NLP_Notebook.ipynb`**: A Jupyter notebook that contains the code for NLP tasks, model training, and data preprocessing. 
   
5. **`Final_Output_all_blog_posts.txt`**: A text file containing the final output of blog posts or review summaries generated from the analysis. 
   
6. **`requirements.txt`**: A file listing the dependencies and libraries required to run the project. Install them by running `pip install -r requirements.txt`. 

## Model Files 🎯

Due to file size limitations, the trained models are hosted on Google Drive. You can download them from the following links:

- [best_sentiment_Model](https://drive.google.com/drive/folders/1_xZSgu_vHhaew1vGXSbx0iCJVFwX_hc1?usp=drive_link) 
- [best_dislibret_Model](https://drive.google.com/drive/folders/1_xZSgu_vHhaew1vGXSbx0iCJVFwX_hc1?usp=sharing) 
