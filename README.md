# 📚 Book Recommender System

A machine learning–powered recommender system that suggests books based on user preferences, ratings, and metadata. This project leverages a curated dataset of books with attributes such as title, author, genres, ratings, and descriptions to deliver personalized recommendations.

---

## 🚀 Features
- Content-based recommendations using book metadata (author, genres, description).
- Clean UI for browsing recommendations.
- Graceful handling of missing data (e.g. incomplete metadata).

---

## 📂 Dataset
The system uses a dataset containing detailed book information. Example entries include:

| Title | Author | Rating | Genres | Pages | Publisher | Publish Date |
|-------|--------|--------|--------|-------|-----------|--------------|
| *Mummy Told Me Not to Tell* | Cathy Glass | 4.36 | Nonfiction, Memoir, Psychology | 344 | HarperElement | July 16, 2015 |
| *A Year Down Yonder* | Richard Peck | 4.12 | Historical Fiction, Young Adult, Humor | 160 | Puffin Books | Nov 21, 2002 |

Each record includes:
- **Book metadata**: title, author, series, language, ISBN, publisher, publish dates.
- **User engagement**: ratings, number of ratings, liked percentage.
- **Content attributes**: genres, characters, description, setting.
- **Additional info**: awards, cover image, price.

---

## 🛠️ Tech Stack
- **Backend**: Python (Flask/Django)
- **Data Processing**: Pandas, NumPy
- **Recommendation Algorithms**: Scikit-learn
- **Frontend**: HTML
- **Storage**: Parquet/CSV for datasets, Git LFS for large files

---
