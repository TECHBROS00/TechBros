# Data Preprocessing
import pandas as pd
import numpy as np
import nltk
from collections import Counter  # Import Counter class
from textblob import TextBlob  # Import TextBlob class
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Model
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\anokha\\amazon.csv')
pd.set_option('display.max_colwidth', 60)
df.head()

data = {
    'Feature Name': ['product_id', 'product_name', 'category', 'discounted_price', 'actual_price', 
                     'discount_percentage', 'rating', 'rating_count', 'about_product', 'user_id', 
                     'user_name', 'review_id', 'review_title', 'review_content', 'img_link', 'product_link'],
    'Data Type': ['object'] * 16,
    'Description': [
        'Unique identifier for each product',
        'Name of the product',
        'Category to which the product belongs',
        'Discounted price of the product',
        'Original price of the product before discounts',
        'Percentage of the discount provided on the product',
        'Average rating given to the product by users',
        'Number of users who have rated the product',
        'Description or details about the product',
        'Unique identifier for the user who wrote the review',
        'Name of the user who wrote the review',
        'Unique identifier for each user review',
        'Short title or summary of the user review',
        'Full content of the user review',
        'URL link to the product\'s image',
        'URL link to the product\'s page on Amazon\'s official website'
    ]
}

descriptive_df = pd.DataFrame(data)
descriptive_df

df = df.dropna()
print(df.isnull().sum())
df = df.drop_duplicates()

# Convert 'discounted_price' and 'actual_price' by removing currency symbol and converting to float
df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)

# Convert 'discount_percentage' by removing '%' and converting to float
df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)

# Convert 'rating' to float
df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')

# Convert 'rating_count' by removing commas and converting to int
df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)

# Cleaning and preprocessing text without lemmatization
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # Split text into words and rejoin without stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Assuming df is your DataFrame and it has been previously loaded
# Apply the clean_text function to the DataFrame columns
df['product_name'] = df['product_name'].apply(clean_text)
df['about_product'] = df['about_product'].apply(clean_text)
df['review_content'] = df['review_content'].apply(clean_text)
df['category'] = df['category'].apply(clean_text)

# Extracting the top-level category
df['category'] = df['category'].apply(lambda x: x.split('|')[0] if pd.notnull(x) else x)

# Sorting the data by rating_count in descending order
top_selling_products = df.sort_values(by='rating_count', ascending=False).head(10)

# Selecting relevant columns for display
top_selling_products = top_selling_products[['product_name', 'rating', 'rating_count']]
top_selling_products.reset_index(drop=True, inplace=True)
top_selling_products

categories = df['category'].str.split('|').explode()
category_counts = Counter(categories)
category_df = pd.DataFrame(category_counts.items(), columns=['Category', 'Count']).sort_values(by='Count', ascending=False)

# Display the top categories
top_categories = category_df.head(10)
top_categories.reset_index(drop=True, inplace=True)
top_categories

# Classify sentiment
def sentiment_analysis(text):
    analysis = TextBlob(text)
    # threshold for positive and negative sentiments
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Applying sentiment analysis to the review content
reviews = df['review_content']
reviews_sentiments = reviews.apply(sentiment_analysis)

# Counting the occurrences of each sentiment
sentiment_counts = reviews_sentiments.value_counts()

# Adding the sentiment labels back to the reviews
df['Sentiment'] = reviews_sentiments

# Finding examples of positive, neutral, and negative sentiments
positive_example = df[df['Sentiment'] == 'Positive'].iloc[0]['review_content']
neutral_example = df[df['Sentiment'] == 'Neutral'].iloc[0]['review_content']
negative_example = df[df['Sentiment'] == 'Negative'].iloc[0]['review_content']

print("Example of sentiment review: ")
example_reviews = pd.DataFrame({
    "Sentiment": ["Positive", "Neutral", "Negative"],
    "Review": [positive_example, neutral_example, negative_example]
})
example_reviews

drop_col = ['discounted_price', 'actual_price', 'discount_percentage', 'review_id', 'review_title',
            'user_name', 'img_link', 'product_link']
df = df.drop(columns=drop_col)

df['combined_text'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product'] + ' ' + df['review_content']
# Fill null with empty string to avoid issues
df['combined_text'] = df['combined_text'].fillna('')

# Instantiate TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

label_encoder = LabelEncoder()

# Fitting the encoder and transforming the 'Sentiment' column
df['Encoded_Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Compute the cosine similarity matrix based on the tfidf_matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Print the shape of the cosine similarity matrix to verify
cosine_sim.shape

# Create a product-user matrix with overall product ratings
product_user_matrix = df.pivot_table(index='product_id', values='rating', aggfunc='mean')

# Fill missing values with the average rating
product_user_matrix = product_user_matrix.fillna(product_user_matrix.mean())

# Display the product-user matrix
product_user_matrix.head()

def hybrid_recommendation(product_id, content_sim_matrix, product_user_matrix, products, top_n=10):
    # Get the index of the product that matches the product_id
    idx = products.index[products['product_id'] == product_id][0]

    # Content-based filtering
    # Get pairwise similarity scores
    sim_scores = list(enumerate(content_sim_matrix[idx]))
    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the top N most similar products
    content_recommendations_idx = [i[0] for i in sim_scores[1:top_n + 1]]

    # Collaborative Filtering
    # Get the rating of the current product
    if product_id in product_user_matrix.index:
        current_product_rating = product_user_matrix.loc[product_id].values[0]
        # Find products with similar ratings
        similar_rating_products = product_user_matrix.iloc[
            (product_user_matrix['rating'] - current_product_rating).abs().argsort()[:top_n]]

    # Combine content and collaborative recommendations
    # Get indices for collaborative recommendations
    collaborative_recommendations_idx = similar_rating_products.index
    # Map indices to product IDs
    collaborative_recommendations_idx = [products.index[products['product_id'] == pid].tolist()[0] for pid in
                                          collaborative_recommendations_idx]

    # Combine indices from both methods and remove duplicates
    combined_indices = list(set(content_recommendations_idx + collaborative_recommendations_idx))

    # Get recommended products details
    recommended_products = products.iloc[combined_indices].copy()
    recommended_products = recommended_products[['product_id', 'product_name', 'rating']]

    return recommended_products

sample_product_id = df['product_id'][0]
sample_product_name = df['product_name'][0]
recommended_products = hybrid_recommendation(sample_product_id, cosine_sim, product_user_matrix, df)
print("Recommendation for user who purchased product \"" + sample_product_name + "\"")
recommended_products.head(10)
# Ask the user for a category
user_category = input("Enter a category: ")

# Filter the DataFrame to include only products from the specified category
category_df = df[df['category'] == user_category]

# If there are products in the specified category
if not category_df.empty:
    # Select a sample product ID from the filtered DataFrame
    sample_product_id = category_df['product_id'].iloc[0]

    # Generate recommendations for the sample product ID within the specified category
    recommended_products = hybrid_recommendation(sample_product_id, cosine_sim, product_user_matrix, category_df)

    # Display the recommendations with links
    print("Recommendations for the category '" + user_category + "':")
    for idx, row in recommended_products.iterrows():
        product_name = row['product_name']
        product_rating = row['rating']
        product_link = category_df[category_df['product_id'] == row['product_id']]['product_link'].values[0]
        print(f"{product_name} (Rating: {product_rating}) - {product_link}")
else:
    print("No products found in the category '" + user_category + "'.")

import random

# Exclude products that were already recommended
excluded_products = set(recommended_products.index)

# Exclude top selling products
excluded_products.update(set(top_selling_products.index))

# Exclude products from the user's specified category
excluded_products.update(set(category_df.index))

# Choose random products from the dataset excluding the ones in the excluded list
random_products = df[~df.index.isin(excluded_products)]

# Select a random sample of products
random_sample = random_products.sample(n=5)

print("Randomly selected products from the dataset:")
print(random_sample[['product_name', 'rating']])
