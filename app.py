# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer

# Load the trained model
model_path = 'book_recommender.HDF5'
with open(model_path, 'rb') as file:
    model = joblib.load(file)
#data
df = pd.read_parquet('books_data.parquet')
ps = PorterStemmer()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def predict():
    # Extract data from form
    title, genre, description = [x for x in request.form.values()]
    #genre
    try :
        genre =  ' '.join([i.replace(' ','').strip().lower() for i in genre.split(',')])  # mutliple genres
    except:
        genre = genre.replace(' ','').lower().strip()
    #description
    description = re.sub(r'[^\w\s]', ' ',description.lower().strip())
    description = re.sub(r'\s+', ' ', description)

    #tag 
    tag = ' '.join([ps.stem(i) for i in (genre + ' ' + description).split(' ')])

    #get distance and indices
    distance, indices = model.named_steps['nn'].kneighbors(model.named_steps['vectorizer'].transform([tag]),return_distance=True)
    
    #getting the books
    recommended_books = df.iloc[indices.flatten()][['title','author','coverImg']].to_dict(orient='records')

    return render_template(
        'recommendations.html',
        books=recommended_books,
        query_title=title
    )


if __name__ == "__main__":
    app.run(debug=True)