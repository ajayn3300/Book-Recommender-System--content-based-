# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained model
tfidf_path = 'tfidf.joblib'
with open(tfidf_path, 'rb') as file:
    tfidf = joblib.load(file)

#stemmer
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

    #transform tag into vector
    tag = tfidf.transform([tag]).toarray().astype(np.float16)

    #let's get most similar books
    similar = list(enumerate(cosine_similarity(tag,np.load('book_vec.npz')['vec'])[0]))

    #top 10 indices of similar books
    indices = [i for i, j in sorted(similar, key = lambda x :x[1], reverse=True)[:10]]
    
    #getting the books
    recommended_books = pd.read_parquet('books_modified.parquet').iloc[indices].to_dict(orient='records')

    return render_template(
        'recommendations.html',
        books=recommended_books,
        query_title=title
    )


if __name__ == "__main__":
    app.run(debug=True)