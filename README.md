# Recommender System

This project builds a book recommender system using collaborative filtering. It processes user-book interactions, stores data efficiently in LibSVM format, and generates personalized recommendations based on user similarities.

## Features
- **Data Preprocessing:** Cleans and structures user, book, and ratings data.
- **Sparse Matrix Representation:** Converts user-book interactions into a LibSVM format for efficient storage.
- **Collaborative Filtering with Cosine Similarity:** Finds similar users and recommends books based on weighted ratings.
- **Output in CSV Format:** Generates top-5 book recommendations for each user.

## Installation
Ensure you have Python installed, then install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
1. **Prepare Data:** Place `Ratings.csv`, `Users.csv`, and `Books.csv` in the project directory.
2. **Run the Script:**
   ```bash
   python main.py
   ```
3. **Execution Time:**
   - The script takes **approximately 3 hours** to complete, depending on system performance.
   - It processes the dataset, computes user similarities, and generates book recommendations.
4. **Output Files:**
   - `output.libsvm` - Processed data in LibSVM format.
   - `final_recommendations_from_libsvm.csv` - Top-5 book recommendations for each user.

## Dependencies
- pandas
- numpy
- scipy


## Explanation of the Process
1. **Data Preprocessing**
   - Loads and cleans user, book, and ratings data.
   - Handles missing values and ensures proper data types.
   - Maps user IDs and ISBNs to numeric indices for matrix representation.

2. **Generating the Sparse Matrix**
   - Converts user-book ratings into a **LibSVM format** for efficient storage.
   - Saves the processed data as `output.libsvm`.

3. **Collaborative Filtering & Similarity Computation**
   - Computes **cosine similarity** to find similar users.
   - Identifies **top-K similar users** for personalized recommendations.

4. **Generating Recommendations**
   - Estimates book ratings based on similar users’ preferences.
   - Selects **top-5 books** per user and saves them in `final_recommendations_from_libsvm.csv`.

## License
This project is licensed under the MIT License.

