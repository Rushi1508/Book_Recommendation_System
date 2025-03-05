import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import norm

# Load data
ratings_df = pd.read_csv("Ratings.csv", sep=";")
users_df = pd.read_csv("Users.csv", sep=";", low_memory=False)
books_df = pd.read_csv("Books.csv", sep=";")

# Data Cleaning
users_df["Age"] = users_df["Age"].apply(pd.to_numeric, errors='coerce')
users_df["Age"].fillna(users_df["Age"].mean(), inplace=True)
users_df["Age"] = users_df["Age"].astype(int)
books_df["Author"].fillna("Unknown Author", inplace=True)
books_df["Publisher"].fillna("Unknown Publisher", inplace=True)
books_df.drop_duplicates(inplace=True)

# Create mappings
user_map = {user_id: idx for idx, user_id in enumerate(ratings_df["User-ID"].unique())}
book_map = {isbn: idx for idx, isbn in enumerate(ratings_df["ISBN"].unique())}
ratings_df["user_indx"] = ratings_df["User-ID"].map(user_map)
ratings_df["book_indx"] = ratings_df["ISBN"].map(book_map)

# Generate sparse matrix
sparse_matrix = coo_matrix(
    (ratings_df["Rating"], (ratings_df["user_indx"], ratings_df["book_indx"])),
    shape=(len(user_map), len(book_map))
).tocsr()

# Save libsvm format
def save_libsvm(sparse_matrix, filename="output.libsvm"):
    with open(filename, "w") as file:
        for user_indx in range(sparse_matrix.shape[0]):
            row = sparse_matrix.getrow(user_indx).tocoo()
            ratings = [f"{book_indx+1}:{rating}" for book_indx, rating in zip(row.col, row.data)]
            file.write(f"{user_indx} " + " ".join(ratings) + "\n")
    print(f"LibSVM file saved as {filename}")

save_libsvm(sparse_matrix)

# Load ISBN mapping
isbn_to_title = dict(zip(books_df["ISBN"], books_df["Title"]))
index_to_isbn = {v: k for k, v in book_map.items()}

# Compute cosine similarities
def calculate_top_k_similar_users(sparse_matrix, K=10):
    num_users = sparse_matrix.shape[0]
    top_k_indices = np.zeros((num_users, K), dtype=int)
    top_k_similarities = np.zeros((num_users, K), dtype=float)
    row_norms = norm(sparse_matrix, axis=1)
    
    for user_id in range(num_users):
        similarity = sparse_matrix[user_id].dot(sparse_matrix.T).toarray().flatten()
        similarity /= (row_norms[user_id] * row_norms + 1e-10)
        top_k = np.argsort(-similarity)[:K]
        top_k_indices[user_id] = top_k
        top_k_similarities[user_id] = similarity[top_k]
    
    return top_k_indices, top_k_similarities

top_k_indices, top_k_similarities = calculate_top_k_similar_users(sparse_matrix)

# Generate recommendations
def recommend_books(sparse_matrix, K=10, top_k_indices=top_k_indices, top_k_similarities=top_k_similarities):
    recommendations = []
    for user_id in range(sparse_matrix.shape[0]):
        similar_users = top_k_indices[user_id]
        similarities = top_k_similarities[user_id]
        BK = sparse_matrix[similar_users].sum(axis=0).nonzero()[1]
        book_scores = {}
        
        for book_id in BK:
            if sparse_matrix[user_id, book_id] == 0:
                numerator, denominator = 0, 0
                for idx, similar_user in enumerate(similar_users):
                    rating = sparse_matrix[similar_user, book_id]
                    if rating > 0:
                        numerator += rating * similarities[idx]
                        denominator += similarities[idx]
                if denominator > 0:
                    book_scores[book_id] = numerator / denominator
        
        top_books = sorted(book_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for book_id, score in top_books:
            isbn = index_to_isbn.get(book_id, None)
            if isbn:
                title = isbn_to_title.get(isbn, "Unknown Title")
                recommendations.append({
                    "User_ID": user_id,
                    "Book_ID": book_id,
                    "ISBN": isbn,
                    "Book_Title": title,
                    "Recommendation_Score": score
                })
    return pd.DataFrame(recommendations)

recommendations_df = recommend_books(sparse_matrix)
recommendations_df.to_csv("final_recommendations_from_libsvm.csv", index=False)
print("Recommendations saved to final_recommendations_from_libsvm.csv")
