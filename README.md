# An Interactive and Educational Streamlit Webapp
Interactive Streamlit app exploring Naive-Bayes classification with sklearn

**New features:**

  >Internally, Streamlit reruns the python script from the top whenever the user interacts with any widget. This means that once the reader clicks a `Run Code` button, any previous output is erased as the app is refreshed. This update implements the new `session_state` feature, which can preserve information across reruns of the app. This allows output to be preserved, and not cleared as a reader advances along in the app. This preservation of output requires the rerunning of internal processes, and we have made heavy use of the caching `@st.cache` function decorator to store outputs of functions once they are computed, allowing large speed-ups in performance.

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