import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import re
import math
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
def extract_location(loc):
    if "New York" in loc:
        loc = "New York, NY"
    result = re.findall("\s[A-Z]{2}$", loc)
    if len(result) == 1:
        return result[0][1:]
    return loc
data = pd.read_excel("final_project.ods", dtype=str)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
x["location"] = x["location"].apply(extract_location)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
preprocessor = ColumnTransformer(
    transformers=[
        ("title_feature", TfidfVectorizer(stop_words="english"), "title"),
        ("location_feature", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ("description_feature", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
        ("function_feature", OneHotEncoder(handle_unknown="ignore"), ["function"]),
        ("industry_feature", TfidfVectorizer(stop_words="english"), "industry")
    ]
)
# pram_grid = {
#     "classifier__C":[math.pow(10, -3), math.pow(10, -2), math.pow(10, -1), math.pow(10, 0),
#                      math.pow(10, 1), math.pow(10, 2), math.pow(10, 3)],
#     "classifier__kernel":['linear', 'poly', 'sigmoid', 'rbf'],
# }
cls = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(C=1000, kernel='rbf'))
])
# cls_grid = GridSearchCV(cls, param_grid=pram_grid, verbose=0, cv=5, n_jobs=-1)
# cls_grid.fit(x_train, y_train)
# print(cls_grid.best_params_)
# print(cls_grid.best_score_)
cls.fit(x_train, y_train)
y_predicted = cls.predict(x_test)
print(classification_report(y_test, y_predicted))