# 🎬 Netflix Recommendation System

A Machine Learning based movie recommendation engine built using **SVD (Singular Value Decomposition)** and the **Surprise Library** to provide personalized Netflix movie recommendations based on user ratings.

---

## 🔗 Project Links

### 📘 Google Colab Notebook
(https://colab.research.google.com/drive/1lqRlVyetoAqODGLvo-T0PsJEcDA1YJZO?usp=sharing)
### 💻 GitHub Repository
https://github.com/khushboo-datasci/Netflix-Recommendation-System

---

# 📌 Project Overview

Recommendation systems are widely used in platforms like Netflix, Amazon, Spotify, and YouTube to suggest personalized content to users.

This project demonstrates how collaborative filtering works using the **SVD (Singular Value Decomposition)** algorithm to recommend movies based on user preferences and historical ratings.

The system learns hidden patterns between:
- Similar users
- Similar movies
- User interests and preferences

Using these learned patterns, the model predicts ratings for movies a user has not watched and recommends the highest-rated ones.

---

# 🚀 Features

✅ Netflix-style Recommendation System  
✅ Collaborative Filtering  
✅ SVD Matrix Factorization  
✅ Personalized Movie Recommendations  
✅ Data Cleaning & Preprocessing  
✅ User & Movie Filtering  
✅ RMSE Evaluation  
✅ Surprise Library Implementation  

---

# 🛠️ Tech Stack

| Technology | Usage |
|---|---|
| Python | Programming Language |
| Pandas | Data Handling |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Seaborn | Data Visualization |
| Surprise Library | Recommendation System |
| Scikit-Learn | Machine Learning Utilities |
| Google Colab | Development Environment |

---

# 📂 Dataset Information

This project uses the **Netflix Prize Dataset**.

Dataset includes:
- User IDs
- Movie IDs
- Ratings
- Movie Titles

### 📁 Dataset Link
 (https://drive.google.com/drive/folders/1CwJCkWn0pbXsKLBTtP72h-0OribiMz4T?usp=sharing)
---

# ⚙️ Project Workflow

## 1️⃣ Data Loading

```python
netflix_dataset = pd.read_csv(
    '/content/drive/MyDrive/Datasets/NetflixRecommendationSystem/combined_data_1.txt',
    header=None,
    names=["Cust_ID", "Ratings"],
    usecols=[0,1]
)
```

---

## 2️⃣ Data Preprocessing

Performed:
- Handling missing values
- Extracting Movie IDs
- Converting datatypes
- Removing inactive users
- Filtering unpopular movies

---

## 3️⃣ Exploratory Data Analysis

Analyzed:
- Total movies
- Total customers
- Ratings distribution
- Active users
- Popular movies

### Ratings Distribution

```python
stars.plot(kind='barh', legend=False)
plt.title('Ratings Distribution')
plt.show()
```

---

# 🧠 SVD (Singular Value Decomposition)

SVD is a matrix factorization technique used in recommendation systems.

It identifies hidden relationships between:
- Similar users
- Similar movies
- User preferences

The model predicts estimated ratings for unseen movies.

---

# 🤖 Model Building

The recommendation system is implemented using the **Surprise Library**.

```python
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate
```

### Workflow:
1. Reader defines rating structure
2. Dataset converts raw data
3. SVD trains recommendation model
4. Cross-validation evaluates performance

---

# 📊 Model Evaluation

### Evaluation Metric:
✅ RMSE (Root Mean Squared Error)

```python
cross_validate(model, data, measures=['RMSE'], cv=3, verbose=True)
```

Lower RMSE indicates better recommendation accuracy.

---

# 🎯 Personalized Recommendations

```python
user_1331154['Estimated_score'] = user_1331154['Movie_ID'].apply(
    lambda x: model.predict(1331154, x).est
)
```

The system predicts ratings and recommends top movies based on estimated scores.

---

# 📁 Project Structure

```bash
Netflix-Recommendation-System/
│
├── notebooks/
│   └── Netflix_Recommendation_System.ipynb
│
├── data/
│   ├── combined_data_1.txt
│   └── movie_titles.csv
│
├── images/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/khushboo-datasci/Netflix-Recommendation-System.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Open Google Colab and run:

```bash
Netflix_Recommendation_System.ipynb
```

---

# 📌 Required Libraries

```txt
numpy==1.26.4
pandas
matplotlib
seaborn
scikit-surprise
scikit-learn
jupyter
```

---

# 🚀 Future Improvements

- Streamlit Web App
- Hybrid Recommendation System
- Deep Learning Recommendations
- Real-Time Recommendations
- FastAPI Deployment
- Docker Deployment
- Cloud Deployment (AWS/GCP)

---

# 📚 Learning Outcomes

Through this project, I learned:
- Recommendation Systems
- Collaborative Filtering
- Matrix Factorization
- Data Preprocessing
- Exploratory Data Analysis
- Model Evaluation
- Personalized Recommendation Logic

---

# 👩‍💻 Author

## Khushboo Kumari

### 🔗 GitHub Profile
https://github.com/khushboo-datasci

---

# 📜 License

This project is open-source and available under the MIT License.
