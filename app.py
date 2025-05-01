{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "47382ddb-f6dc-4e33-a94d-23704c29780a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model accuracy: 1.0\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['rf_model.sav']"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "import joblib\n",
    "\n",
    "iris_df = pd.read_csv(\"iris.csv\")\n",
    "\n",
    "X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]\n",
    "y = iris_df['Class Label']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n",
    "\n",
    "clf = RandomForestClassifier(n_estimators=100)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "y_pred = clf.predict(X_test)\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Model accuracy: {accuracy}\")\n",
    "\n",
    "joblib.dump(clf, \"rf_model.sav\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "8c96a3d9-07ef-4100-b952-f46f237a3e63",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
