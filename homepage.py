import streamlit as st

def app_page():

    # Print title
    st.title('Naive-Bayes Classification')

    st.header('Welcome to my interactive web application developed with **Streamlit**!')

    st.markdown(
        f'''
        This application serves as a review of the mathematical ideas behind Naive-Bayes classification, as well as a thorough interactive walkthrough of this classification method as applied to the **Iris** dataset, containing numerical data describing samples of the _iris_ flower species, and **20 Newsgroups** dataset, containing textual samples of internet postings on various topics. Both datasets are included with `sklearn`. 

        The **Mathematics Background** page contains the nitty-gritty mathematics behind the Naive-Bayes classification algorithm based on _maximum a posteriori (MAP) estimation_. We begin by introducing conditional probabilities and outline a simple proof of Bayes Theorem. We illustrate these ideas by developing a very simple email spam filter that works by flagging an email given the presence of specific words, and calculate the filter's efficiency. We extend this rudimentary filter to one that can be trained on a labeled set of emails and see how one uses the trained classifier to label new emails as 'spam' or 'not spam'. Multi-class classification is covered by extending our formulation to arbitrary numbers of data classes. Finally, the typical process of implementing the classification algorithm using the `sklearn` library is outlined. The included variations of the algorithm and their properties are covered. Tools used for evaluating the performance of classifiers included in `sklearn` are covered.

        The interactive walkthoughs are contained in the **Naive-Bayes with sklearn** page. The content of this page changes depending on which dataset selected on the sidebar. For both datasets, we perform Exploratory Data Analysis to get a feel for the structure and distribution of the features in the data before attempting classification. The 'Iris' dataset contains strictly numerical data, and comes in a form ready for classification. The '20 Newsgroups' dataset contains raw text data, and must be transformed into a numerical representation conducive for classification. In doing so, we review in detail the tools available to us with `sklearn` to extract meaningful numerical features from textual data. These tools include many ways to customize how and what features are extracted, and we cover the main ones successively to see their effect and use. For both datasets, we show how we can optimize the performance of our classifiers by tuning the parameters available.

        To get started, select a page from the **Page Navigation** menu on the sidebar. Feel free to learn about how this powerful classification technique works by reading through the **Mathematics Background** pages, jump right to classification by selecting a dataset on the **Naive-Bayes with sklearn** page.

        I hope you enjoy this application and find it useful for your understanding of the simple, yet powerful tool that is the Naive-Bayes classifier.
        '''
    )
        
    st.markdown(
        f'''
        {''.join('<br>' for i in range(3))}
        ''', unsafe_allow_html=True
        )

    col1, col2 = st.beta_columns([1,3])
    with col2:
        st.write(
            r'''
            _Aside_: Throughout this application are mathematical expressions formatted with $\LaTeX$. Although [Streamlit](https://streamlit.io/) includes basic support for in-line and display-mode equations, at the time of this writing, equation numbering and hyperlinking functionality was not included. These features you see in use in this app were implemented with a custom python class that I developed specifically for use with Streamlit. To use this functionality in your own Streamlit apps, feel free to clone the repository from my [GitHub](https://github.com/evan-wes/latex-numbering-class)!
            '''
        )