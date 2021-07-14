# naive_bayes_sklearn
Interactive Streamlit app exploring Naive-Bayes classification with sklearn

Find the app deployed on heroku at
[http://naive-bayes-sklearn.herokuapp.com/](http://naive-bayes-sklearn.herokuapp.com/).
(App may take up a few moments to load)

In this web app, we introduce the ideas behind Naive-Bayes classification, including a thorough mathematical background of all the ideas needed to understand how a Naive-Bayes classifier works.

We implement classification with two well known datasets, both of which are included in the `sklearn` library: the Iris dataset, comprised of numerical features, and the 20 Newsgroups dataset, comprised of textual samples. With the 20 Newsgroup dataset, we cover in detail how to use the textual feature extraction tools included with `sklearn`, such as `CountVectorizer` and `TfidfVectorizer`, and how to adequately tune their parameters.

Primary python libraries used:
  - Deployment:
    - `streamlit`

  - Visualizations:
    - `matplotlib`
    - `seaborn`

  - Data analysis:
    - `numpy`
    - `pandas`

  - Machine learning algorithms:
    - `sklearn`

  - Model performance analysis:
    - `sklearn` 