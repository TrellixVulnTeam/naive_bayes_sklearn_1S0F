import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import json

# Configure matplotlib to supress warning about many open figures
#plt.rcParams.update({'figure.max_open_warning': 0})

# Import function to fetch dataset
from sklearn.datasets import fetch_20newsgroups

# Import feature extraction tools
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# Import SciPy horizontal stacking function
from scipy.sparse import hstack

# Import Multinomial Naive-Bayes classifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, GaussianNB

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import sklearn metrics for analysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import heatmap plotting function
from matrix_heatmap import matrix_heatmap

# Import custom latex display and numbering class
from latex_equation_numbering import latex_equation_numbering

# Instantiate the latex_equation_numbering class
newsgroup_equations = latex_equation_numbering()

def newsgroup_example():
    st.title('Classification with the 20 Newsgroup dataset')
    st.write(
        '''
        The 20 Newsgroup dataset is part of the `sklearn.datasets` module. There are twenty different categories of newsgroup postings that make up the dataset, each containing roughly one thousand posting samples. For more information on the dataset, see the [original website](http://qwone.com/~jason/20Newsgroups/). The different categories are roughly grouped into the following topics:
        '''
    )

    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.subheader('Computation')
        st.write(
            '''
            - comp.graphics
            - comp.os.ms-windows.misc
            - comp.sys.ibm.pc.hardware
            - comp.sys.mac.hardware
            - comp.windows.x
            '''
        )

    with col2:
        st.subheader('Science')
        st.write(
            '''
            - sci.crypt
            - sci.electronics
            - sci.med
            - sci.space
            '''
        )

    with col3:
        st.subheader('Recreation')
        st.write(
            '''
            - rec.autos
            - rec.motorcycles
            - rec.sport.baseball
            - rec.sport.hockey
            '''
        )
    
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.subheader('Politics')
        st.write(
            '''
            - talk.politics.misc
            - talk.politics.guns
            - talk.politics.mideast
            '''
        )

    with col2:
        st.subheader('Religion')
        st.write(
            '''
            - talk.religion.misc
            - alt.atheism
            - soc.religion.christian
            '''
        )

    with col3:
        st.subheader('Miscellaneous')
        st.write(
            '''
            - misc.forsale
            '''
        )
            
    st.write(
        '''   
        Categories under the same broad topic should be more difficult for a classification algorithm to distinguish from one another.
        '''
    )

    st.header('Loading the Data')
    st.write(
        '''
        To load the **20 newsgroup** dataset, we use the `fetch_20newsgroups()` function from the `sklearn.datasets` module:
        ```python
        from sklearn.datasets import fetch_20newsgroups

        newsgroups = fetch_20newsgroups()
        ```

        The data and target names are found using the `.data`, `.target_names` attributes. A description of the dataset can be returned using `.DESCR`. We can examine individual samples as well. Try out the following code by clicking the 'Run Code' button. The description and sample texts are lengthy, so they're placed inside expandable containers.
        '''
    )

    # ---------------------------------------------
    # ----- Load Newsgroup dataset code block -----
    # ---------------------------------------------
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        st.write(
            '''
            ```python
            from sklearn.datasets import fetch_20newsgroups

            newsgroups = fetch_20newsgroups()

            print('Target names:')
            print(newsgroups.target_names)

            print('Data description:')
            print(newsgroups.DESCR)

            print('First three samples:')
    
            for i in range(3):
                print('Sample '+str(i+1)+' data:')
                print(newsgroups.data[i])
                print('Sample '+str(i+1)+' class:')
                print(newsgroups.target_names[newsgroups.target[i]])
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='newsgroups_load_run_button')
    if run_button:

        newsgroups = fetch_20newsgroups()

        with output_col:
            st.write('**Target names:**')
            st.text(newsgroups.target_names)
            
            st.write('**Data description:**')
            with st.beta_expander('Expand description'):
                st.text(newsgroups.DESCR)
            st.text('\n  ')
            
            st.write('**First three samples:**')
            for i in range(3):
                with st.beta_expander(f'Expand sample {i+1}'):
                    st.write(f'**Sample {i+1} data:**')
                    st.text(newsgroups.data[i])
                    st.write(f'**Sample {i+1} class:**')
                    st.text(newsgroups.target_names[newsgroups.target[i]])

    st.write(
        '''
        The `fetch_20newsgroups()` function has several arguments. First, each class of data is already split into training and testing sets. To select only data from one set, we can use the keyword argument `subset='train'`, `subset='test'`, or `subset='all'` for both (default is `'train'`). The data can be shuffled when loaded with the `shuffle` keyword (default is `True`), and reproducable results of shuffling can be optained with the `random_state` argument.
        
        We can select only data from certain target classes using the `categories` keyword. If we wanted only the topics related to science, we could use `categories=['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']`. Since the postings are on internet boards, they contain a lot of unnecessary metadata. To force a classifier to classify based only on the post text, and not overfit the metadata, we can use the `remove` keyword argument. The value of the argument is any combination of 'headers', 'footers', and 'quotes'. Choosing 'headers' removes newsgroup headers, 'footers' removes blocks at the ends of posts that look like signatures, and 'quotes' removes lines that appear to be quoting another post. To see the effect of this choice, the following code block compares the same samples with and without the `remove` option.
        '''
    )

    # -----------------------------------------------------
    # ----- Load Newsgroup dataset options code block -----
    # -----------------------------------------------------
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        st.write(
            '''
            ```python
            from sklearn.datasets import fetch_20newsgroups

            newsgroups = fetch_20newsgroups(subset='train', 
                                            shuffle='true', 
                                            random_state=42)

            newsgroups_clean = fetch_20newsgroups(subset='train', 
                                                  shuffle='true', 
                                                  random_state=42, 
                                                  remove=('headers', 'footers', 'quotes'))

            print('First three samples:')

            for i in range(3):
                print('Sample '+str(i+1)+' data:')
                print(newsgroups.data[i])
                print('Sample '+str(i+1)+' cleaned data:')
                print(newsgroups_clean.data[i])
                print('Sample '+str(i+1)+' class:')
                print(newsgroups.target_names[newsgroups.target[i]])
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='newsgroups_load_options_run_button')
    if run_button:

        newsgroups = fetch_20newsgroups(subset='train', shuffle='true', random_state=42)
        newsgroups_clean = fetch_20newsgroups(subset='train', shuffle='true', random_state=42, remove=('headers', 'footers', 'quotes'))

        with output_col:
            st.write('**First three samples:**')
            for i in range(3):
                with st.beta_expander(f'Expand sample {i+1}'):
                    st.write(f'**Sample {i+1} data:**')
                    st.text(newsgroups.data[i])
                    st.write(f'**Sample {i+1} cleaned data:**')
                    st.text(newsgroups_clean.data[i])
                    st.write(f'**Sample {i+1} class:**')
                    st.text(newsgroups.target_names[newsgroups.target[i]])
    
    st.write(
        '''
        Comparing the samples from the data loaded with and without the `remove` option, we see a lot less of, or complete lack of, post metadata. This is important because we want to be able to classify on the textual content of the posts, and not overfit to things like the names and emails of people who were particularly prevalent on these internet boards at the time, or certain formatting of headers and footers.

        So far, we only have a method of importing the raw text samples. We need some tools to parse the text and extract useful information with which we can try to classify the samples into different content topics. Luckily, `sklearn` has several methods of doing so. We will now explore these options and review how to use each one before applying them to the newsgroup dataset.
        '''
    )

    st.header('Textual feature extraction in `sklearn`')
    st.write(
        '''
        When loaded with `fetch_20newsgroups()`, the data is entirely in textual form. To begin exploring it, we need a method of converting the raw text to a format better suited for analysis. We can do this with the help of the `sklearn.feature_extraction.text` module. Included in this module are `CountVectorizer`, `TfidfTransformer`, and `TfidfVectorizer`, imported as:
        ```python
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
        ```

        The `CountVectorizer` object takes a list of raw text document inputs and produces a corresponding list of feature vectors containing word token counts. It's usage is best explained with concrete examples, which we will do shortly. The `TfidfVectorizer` object takes a list of raw text document inputs and produces a corresponding list of TF-IDF feature vectors. The word token vocabulary-feature vector indexing is handled internally to map each unique word token to a feature index. TF-IDF stands for 'term frequency times inverse document frequency,' which is a method of reducing the weighting of words that occur very frequently in a corpus of documents, as opposed to words that occur only in subsets of documents which can be good class indicators. Finally, `TfidfTransformer` converts a count frequency matrix (the output of `CountVectorizer`) to a TF-IDF matrix (the output of `TfidfVectorizer`)

        Let us explore the use of `CountVectorizer` and `TfidfVectorizer` with some concrete examples below. We will use a simple example corpus taken straight from the documentation to illustrate how feature extraction using these function works:
        ```python
        corpus = ['This is the first document.',
                  'This document is the second document.',
                  'And this is the third one.',
                  'Is this the first document?']
        ```
        A _corpus_ is a collection of related texts. Here we have defined it as a list of strings, each one a _document_. In a text classification problem, each element in the corpus is a data sample. 

        For our convenience, let us define a function that will help us with our analysis of the effects of different options for feature extraction. It will take a feature vectorizer (initialized with whatever options we choose) and corpus and print out the vocabulary and feature vectors:
        ```python
        def print_features(vectorizer, corpus):

            # Fit the vectorizer and transform the corpus
            X = vectorizer.fit_transform(corpus)

            # Define a dataframe to hold the extracted features and values
            X_df = pd.DataFrame(data=X.toarray()).rename(columns={index: term for term, index in vectorizer.vocabulary_.items()})

            # Define a vocabulary DataFrame for easier to reference term indexing
            vocabulary_df = pd.DataFrame(data={'term': vectorizer.vocabulary_.keys(), 'feature index': vectorizer.vocabulary_.values()})
                              .set_index(pd.Index(vectorizer.vocabulary_.values()))
                              .sort_values(by=['feature index'])

            print('Feature names:')
            print(vectorizer.get_feature_names())

            print('Corpus vocabulary:')
            # Raw dictionary output
            print(vectorizer.vocabulary_)
            # Dictionary dataframe
            print(vocabulary_df)

            print('Sparse array output:')
            print(X)
            print('Matrix output:')
            print(X.toarray())
            print('DataFrame output:')
            print(X_df)

            for i in range(len(corpus)):
                print('Sample '+str(i)+' and its feature vector:')
                print(corpus[i])
                print(X_df.iloc[i].to_frame().T)
        ```

        We will use this function over and over again as we review the effects of each option for `CountVectorizer` and `TfidfVectorizer`.
        '''
    )

    st.subheader('Feature extraction: `CountVectorizer`')
    st.write(
        '''
        First we will explore the use of `CountVectorizer` to generate term-frequency feature vectors using our simple example corpus. The [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer) for this function explains the many arguments we can use to customize how the term frequencies are generated. The default values for these arguments work quite well, but we will show some of the effects of tweaking them to our benefit. The `CountVectorizer` class contains methods `.fit()`, `.transform()`, and `.fit_transform()`. The `.fit()` method learns the vocabulary of the dataset, which consists of a term: index dictionary. The word tokens (terms) are the keys and the feature vector indices are the values. The learned vocabulary can be accessed with the `vocabulary_` attribute. The feature names can be accessed with the `.get_feature_names()` method (although this is the same as the output of `.keys()` on the vocabulary dictionary). The `.transform()` method takes a dataset and uses the learned vocabulary to generate term-frequency vectors. The `fit_transform()` method is equivalent to calling `.fit()` and then `.transform()` on a dataset. One uses `.fit_transform()` on the training set and `.transform()` on the testing set.  In a feature vector, the component at a fixed index is the number of occurances of the term whose value in the vocabulary dictionary is that index. Text samples typically contain many unique words, and thus their vocabularies can be lengthy. Since most words won't appear in each sample, feature vectors will contain many components that are zero. The output of `.transform()` or `.fit_transform()` is a _sparse matrix_ and only the non-zero entries are returned in the format of `(row, column) value` entries. To convert these to actual matrices, one can use `.toarray()`. We can even use `pandas` to create a DataFrame, using an inverted form of the vocabulary dictionary to rename to columns, which can make analysis easier later on.
        
        
        Run the code below to instantiate a `CountVectorizer` class and train it on the example corpus using `.fit_transform()`. Then the vocabulary and feature vectors are printed for comparison.
        '''
    )

    # --------------------------------------
    # ----- CountVectorizer code block -----
    # --------------------------------------
    code_col, output_col = st.beta_columns(2)
    X_dict_string = '{index: term for term, index in vectorizer.vocabulary_.items()}'
    vocab_dict_string = '''{'term': vectorizer.vocabulary_.keys(), 'feature index': vectorizer.vocabulary_.values()}'''
    with code_col:
        st.subheader('Code:')
        st.write(
            f'''
            ```python
            from sklearn.feature_extraction.text import CountVectorizer

            corpus = ['This is the first document.',
                      'This document is the second document.',
                      'And this is the third one.',
                      'Is this the first document?']
            
            # Initialize a vectorizer 
            vectorizer = CountVectorizer()

            # Print the extracted features
            print_features(vectorizer, corpus)
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='newsgroups_CountVectorizer_run_button')
    if run_button:

        corpus = ['This is the first document.',
                  'This document is the second document.',
                  'And this is the third one.',
                  'Is this the first document?']
        
        vectorizer = CountVectorizer()

        X = vectorizer.fit_transform(corpus)
        X_df = pd.DataFrame(data=X.toarray()).rename(columns={index: term for term, index in vectorizer.vocabulary_.items()})
        #vocabulary_df = pd.Series(data=vectorizer.vocabulary_).sort_values()
        vocabulary_df = pd.DataFrame(data={'term': vectorizer.vocabulary_.keys(), 'feature index': vectorizer.vocabulary_.values()}).set_index(pd.Index(vectorizer.vocabulary_.values())).sort_values(by=['feature index'])
        
        with output_col:
            st.write('**Feature names:**')
            st.text(vectorizer.get_feature_names())

            st.write('**Corpus vocabulary:**')
            # Raw dictionary output
            st.text(vectorizer.vocabulary_)
            # Dictionary dataframe
            st.write(vocabulary_df)

            with st.beta_expander('Expand feature matrix output forms'):
                st.write('**Sparse array output:**')
                st.write(X)
                st.write('**Matrix output:**')
                st.write(X.toarray())
                st.write('**DataFrame output:**')
                st.write(X_df)

            for i in range(len(corpus)):
                with st.beta_expander(f'Expand sample {i+1}'):
                    st.write(f'**Sample {i+1} and its feature vector:**')
                    st.text(corpus[i])
                    st.write(X_df.iloc[i].to_frame().T)
        
    st.write(
        '''
        As seen above, each term in the corpus is assigned an index, contained in the `.vocabulary_` attribute of the `CountVectorizer` object. Then each feature vector component is the frequency of the term corresponding to the index of the vector component. `CountVectorizer` can be configured using `binary=True` (default is `False`) to instead use boolean values corresponding to the appearance or lack of appearance of each term. Another interesting usage is counting n-grams_. N-grams are sequences of N terms in order. Single terms are technically called 1-grams or unigrams. Consecutive pairs of terms are 2-grams or bigrams. Frequencies of n-grams can be counted with the assignment of the `ngram_range` keyword. It takes a tuple with values corresponding to the minimum and maximum length of n-grams. The default is `ngram_range=(1, 1)`, corresponding to only counting unigrams. N-grams can be useful when class-specific terminology includes phrases. Play around with the `ngram_range` settings below to see the n-gram frequency counts of our sample corpus.
        '''
    )

    # --------------------------------------------------
    # ----- CountVectorizer ngram_range code block -----
    # --------------------------------------------------
    st.subheader('Choose values for minimum and maximum n-gram range')
    with st.beta_columns(3)[0]:
        ngram_min = st.slider('Minimum', min_value=1, max_value=4, step=1, value=1)
        if ngram_min == 4:
            ngram_max = 4
        else:
            ngram_max = st.slider('Maximum', min_value=ngram_min, max_value=4, step=1, value=ngram_min)
    code_col, output_col = st.beta_columns(2)
    X_dict_string = '{index: term for term, index in vectorizer.vocabulary_.items()}'
    vocab_dict_string = '''{'term': vectorizer.vocabulary_.keys(), 
                                               'feature index': vectorizer.vocabulary_.values()}'''
    with code_col:
        st.subheader('Code:')
        st.write(
            f'''
            ```python
            from sklearn.feature_extraction.text import CountVectorizer

            corpus = ['This is the first document.',
                      'This document is the second document.',
                      'And this is the third one.',
                      'Is this the first document?']
            
            # Initialize a vectorizer
            vectorizer = CountVectorizer(ngram_range=({ngram_min}, {ngram_max}))

            # Print the extracted features
            print('Using ngram_range: ({ngram_min}, {ngram_max})')
            print_features(vectorizer, corpus)
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='newsgroups_CountVectorizer_ngram_range_run_button')
    if run_button:

        corpus = ['This is the first document.',
                  'This document is the second document.',
                  'And this is the third one.',
                  'Is this the first document?']
        
        vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max))

        X = vectorizer.fit_transform(corpus)
        X_df = pd.DataFrame(data=X.toarray()).rename(columns={index: term for term, index in vectorizer.vocabulary_.items()})
        #vocabulary_df = pd.Series(data=vectorizer.vocabulary_).sort_values()
        vocabulary_df = pd.DataFrame(data={'term': vectorizer.vocabulary_.keys(), 'feature index': vectorizer.vocabulary_.values()}).set_index(pd.Index(vectorizer.vocabulary_.values())).sort_values(by=['feature index'])
        
        with output_col:
            st.write(f'**Using ngram_range:** ({ngram_min}, {ngram_max})')

            st.write('**Feature names:**')
            st.text(vectorizer.get_feature_names())

            st.write('**Corpus vocabulary:**')
            # Raw dictionary output
            st.text(vectorizer.vocabulary_)
            # Dictionary dataframe
            #st.write(vocabulary_df.to_frame().T.set_index(pd.Index(['feature index'])))
            st.write(vocabulary_df)

            with st.beta_expander('Expand feature matrix output forms'):
                st.write('**Sparse array output:**')
                st.write(X)
                st.write('**Matrix output:**')
                st.write(X.toarray())
                st.write('**DataFrame output:**')
                st.write(X_df)

            for i in range(len(corpus)):
                with st.beta_expander(f'Expand sample {i+1}'):
                    st.write(f'**Sample {i+1} and its feature vector:**')
                    st.text(corpus[i])
                    st.write(X_df.iloc[i].to_frame().T)
    
    st.write(
        '''
        Setting the values for `ngram_range` is one handle on the extracted features. Another way to customize the range of features is with the `min_df`, `max_df`, and `max_features` keyword arguments. The 'df' in `min_df` and `max_df` refers to 'document frequency'. The value for `min_df` (`max_df`) is the lowest (highest) document frequency to include in the feature vocabulary. These cutoffs can be expressed as a percentage (with values as floats between `0.0` and `1.0`) or counts (with values as positive integers). The default values allow for no cutoffs: the default for `min_df` is `1` (an integer, different from `1.0` a float) meaning terms must appear in at least one sample to be included and the default for `max_df` is `1.0`, meaning terms can appear in every document. These arguments are useful for cutting out very rare words, and words that appear very frequently in the corpus. The `max_features` argument takes a positive integer and constrains the vocabulary to the top features by term frequency. Its default value is `None`. 
        
        Another tool for pruning the extracted features is the `stop_words` argument. Stop words are common, frequent words in a language that typically do not convey much information and can be ignored. Examples are 'the', 'a', 'this', etc. The values that `stop_words` accepts are either a list of custom terms, a language such as `'english'`, or the default value of `None`. While useful, employing a stop-word list must be done with some care, as predefined lists, such as using `sklearn`'s `stop_words='english'`, may include words that are actually important to the classification problem. See this [warning](https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words) in the documentation for some insight. The effect of using stop words can be replicated to an extent with a suitable value set for `max_df`. Choosing a value such as `max_df=0.75` for example, would remove words that appear in three-quarters or more of documents, which would be most common stop words.
        
        Finally, for complete custom control over the features extracted, we can use the `vocabulary` keyword. This argument takes a custom list of terms to extract from the corpus.

        Change the settings below for `min_df`, `max_df`, `max_features`, and `stop_words` below to see their effect on our example corpus.
        '''
    )

    # -------------------------------------------------------------
    # ----- CountVectorizer feature limits options code block -----
    # -------------------------------------------------------------
    st.subheader('Choose values for min_df, max_df, max_features, and stop_words')
    col1, col2, col3, col4 = st.beta_columns(4)
    corpus_length = 4 # for the simple example we are using
    with col1:
        min_df_type = st.radio("Input type for 'min_df'", options=['Integer (document counts)', 'Float (document %)', 'Default value'])
        if min_df_type == 'Integer (document counts)':
            min_df_val = st.slider('Minimum document frequency', min_value=1, max_value=corpus_length, step=1, value=1)
            min_doc_freq = min_df_val
        elif min_df_type == 'Float (document %)':
            min_df_val = st.slider('Minimum document frequency', min_value=0.0, max_value=1.0, step=0.05)
            min_doc_freq = min_df_val * corpus_length
        elif min_df_type == 'Default value':
            min_df_val = 1
            min_doc_freq = min_df_val
        st.write(f'Minimum document frequency: {min_doc_freq}')
    with col2:    
        max_df_type = st.radio("Input type for 'max_df'", options=['Integer (document counts)', 'Float (document %)', 'Default value'])
        if max_df_type == 'Integer (document counts)':
            max_df_val = st.slider('Maximum document frequency', min_value=1, max_value=corpus_length, step=1, value=1)
            max_doc_freq = max_df_val
        elif max_df_type == 'Float (document %)':
            max_df_val = st.slider('Maximum document frequency', min_value=0.0, max_value=1.0, step=0.05)
            max_doc_freq = max_df_val * corpus_length
        elif min_df_type == 'Default value':
            max_df_val = 1.0
            max_doc_freq = max_df_val * corpus_length
        st.write(f'Maximum document frequency: {max_doc_freq}')
    with col3:
        max_features_choice = st.radio("Use 'max_features'?", options=['No', 'Yes'])   
        if max_features_choice == 'No':
            max_features_val = None
        elif max_features_choice == 'Yes':
            max_features_val = st.slider('Maximum number of features', min_value=1, max_value=10, step=1)
    with col4:
        stop_words_choice = st.radio("Use 'stop_words'?", options=['No', "Yes (sklearn's 'english' list)"])   
        if stop_words_choice == 'No':
            stop_words_val = None
            stop_words_code = None
        elif stop_words_choice == "Yes (sklearn's 'english' list)":
            stop_words_val = "'english'"
            stop_words_code = 'english'
    
    # Code for testing values of min_df and max_df to avoid errors:
    if min_doc_freq > max_doc_freq:
        st.warning('Maximum document frequency larger than minimum document frequency! Will raise an error if run.')


    code_col, output_col = st.beta_columns(2)
    X_dict_string = '{index: term for term, index in vectorizer.vocabulary_.items()}'
    vocab_dict_string = '''{'term': vectorizer.vocabulary_.keys(), 
                                               'feature index': vectorizer.vocabulary_.values()}'''
    with code_col:
        st.subheader('Code:')
        st.write(
            f'''
            ```python
            from sklearn.feature_extraction.text import CountVectorizer

            corpus = ['This is the first document.',
                      'This document is the second document.',
                      'And this is the third one.',
                      'Is this the first document?']
            
            # Initialize a vectorizer
            vectorizer = CountVectorizer(min_df={min_df_val}, max_df={max_df_val}, max_features={max_features_val}, stop_words={stop_words_val}))

            # Print the extracted features
            print('Using min_df, max_df: {min_df_val}, {max_df_val}')
            print_features(vectorizer, corpus)
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='newsgroups_CountVectorizer_feature_limits_run_button')
    if run_button:

        corpus = ['This is the first document.',
                  'This document is the second document.',
                  'And this is the third one.',
                  'Is this the first document?']
        
        vectorizer = CountVectorizer(min_df=min_df_val, max_df=max_df_val, max_features=max_features_val, stop_words=stop_words_code)
        
        try:
            X = vectorizer.fit_transform(corpus)
        
        except Exception as error:
            with output_col:
                st.write(error)
            
        else:
            X_df = pd.DataFrame(data=X.toarray()).rename(columns={index: term for term, index in vectorizer.vocabulary_.items()})
            #vocabulary_df = pd.Series(data=vectorizer.vocabulary_).sort_values()
            vocabulary_df = pd.DataFrame(data={'term': vectorizer.vocabulary_.keys(), 'feature index': vectorizer.vocabulary_.values()}).set_index(pd.Index(vectorizer.vocabulary_.values())).sort_values(by=['feature index'])
            
            with output_col:
                st.write(f'**Using min_df, max_df:** {min_df_val}, {max_df_val}')

                st.write('**Feature names:**')
                st.text(vectorizer.get_feature_names())

                st.write('**Corpus vocabulary:**')
                # Raw dictionary output
                st.text(vectorizer.vocabulary_)
                # Dictionary dataframe
                #st.write(vocabulary_df.to_frame().T.set_index(pd.Index(['feature index'])))
                st.write(vocabulary_df)

                with st.beta_expander('Expand feature matrix output forms'):
                    st.write('**Sparse array output:**')
                    st.write(X)
                    st.write('**Matrix output:**')
                    st.write(X.toarray())
                    st.write('**DataFrame output:**')
                    st.write(X_df)

                for i in range(len(corpus)):
                    with st.beta_expander(f'Expand sample {i+1}'):
                        st.write(f'**Sample {i+1} and its feature vector:**')
                        st.text(corpus[i])
                        st.write(X_df.iloc[i].to_frame().T)


    st.write(
        '''
        If the values of `min_df`, `max_df`, and `max_features` are too restrictive, some of the samples get mapped to the same feature vector, making them indistinguishable to a classification algorithm. When `stop_words` is used, many of the features of the simple example corpus are removed.

        Now that we've covered the basic usage of `CountVectorizer` for textual feature extraction with `sklearn`, we will now look at the other provided option: `TfidfVectorizer`.
        '''    
    )

    st.subheader('Feature extraction: `TfidfVectorizer`')
    st.write(
        '''
        The `TfidfVectorizer` creates feature vectors whose components are values of TF-IDF: term frequency times inverse-document frequency. This vectorizer class can use the same keyword arguments for customizing the terms used in feature extraction that we covered for `CountVectorizer`, but includes some additional arguments for customizing how the TF-IDF values are calculated. Let's breifly review the basics, which are explained in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer) for `TfidfTransformer`, another class that converts the output of a `CountVectorizer` object fit on a corpus into the output that `TfidfVectorizer` would compute on the same corpus (Using `TfidfVectorizer` is the same as using `CountVectorizer` and then `TfidfTransformer`).   
        '''
        +
        r'''

        TF-IDF means multiply the term frequency of a term $t$ in a document $d$ by the inverse-document frequency of that term across all documents. Mathematically, it is computed as
        '''
    )
    newsgroup_equations.add_equation('tf-idf_def', r'\text{tf-idf}(t, d) = \text{tf}(t, d) \cdot \text{idf}(t) \;.')
    newsgroup_equations.display_equation('tf-idf_def')
    st.write(
        r'''
        The function $\text{tf}(t, d)$ is called the term frequency, and $\text{idf}(t)$ the inverse document frequency. If the `sublinear_tf` argument is set to its default value of `False`, $\text{tf}(t,d) = N_{td}$, where $N_{td}$ is simply the number of occurences of the term $t$ in the document $d$. If instead we set `sublinear_tf=True`, then $\text{tf}(t,d) = 1 + \log N_{td}$. The definition of $\text{idf}(t)$ depends on the value of the keyword argument `smooth_idf`, which controls if smoothing is applied to the document frequency. If the choice of `smooth_idf=False` is made, $\text{idf}(t)$ is defined as
        '''
    )
    newsgroup_equations.add_equation('idf_def_smooth_idf=False', r'\text{idf}(t) = 1 + \log\left(\frac{n}{\text{df}(t)}\right)\;,')
    newsgroup_equations.display_equation('idf_def_smooth_idf=False')
    st.write(
        r'''
        where $n$ is the number of documents in the training set, and $\text{df}(t)$ is the number of documents containing term $t$. The effect of adding $1$ to the logarithm is to not completely ignore terms that appear in every document, i.e. $\text{df}(t) = n$ so that $\log(n/\text{df}(t)) = 0$. This form can be problematic if documents in the testing set contain terms that do not appear in the training set, i.e. $\text{df}(t) = 0$. To prevent divisions by zero, one can use the default setting of `smooth_idf=True`, in which $\text{idf}(t)$ is defined as
        '''
    )
    newsgroup_equations.add_equation('idf_def_smooth_idf=True', r'\text{idf}(t) = 1 + \log\left(\frac{1+n}{1+\text{df}(t)}\right)\;.')
    newsgroup_equations.display_equation('idf_def_smooth_idf=True')
    st.write(
        '''
        The effect of adding $1$ to the numerator and denominator of the logarithm is essentially pretending there is an extra document in the training set that contains every term in the entire corpus (including the testing set) exactly once.

        Finally, the raw TF-IDF values in a feature vector are normalized depending on the value of the keyword argument `norm`. The options for this argument (aside from `None` for no normalization) are `'l1'` and `'l2'` corresponding to the $L^1$ and $L^2$ vector norms. The $L^1$ norm, also known as the _Manhatten_ or _taxicab_ distance since it corresponds to the number of city blocks traveled to get somewhere, is the sum of the magnitudes of the components of a vector:
        '''
    )
    newsgroup_equations.add_equation('l1-norm', r'||\bm{x}||_1 = |x_1| + |x_2| + \ldots + |x_n| \;.')
    newsgroup_equations.display_equation('l1-norm')
    st.write(
        '''
        The $L^2$ norm, also called the _Euclidean_ norm, is more familiar from linear algebra as the usual way a vector's length is calculated. It is calculated as the square root of the sum of squares of magnitudes of each component:
        '''
    )
    newsgroup_equations.add_equation('l2-norm', r'||\bm{x}||_2 = \sqrt{|x_1|^2 + |x_2|^2 + \ldots + |x_n|^2} \;.')
    newsgroup_equations.display_equation('l2-norm')
    st.write(
        r'''
        The default value is to use the Euclidean norm.

        Finally, the weighting with inverse-document frequency can be turned off by setting `use_idf=False` (default is `True`). This results in just feature vectors containing only normalized $\text{tf}(t, d)$ term-frequency counts, with the computation method controlled by the `sublinear_tf` argument, instead of $\text{tf}(t, d)\cdot \text{idf}(t)$ TF-IDF values.

        In the code block below, we compare the output of `CountVectorizer` and `TfidfVectorizer` with their default settings (in this example the printed outputs for `print_features` are grouped together for clarity and ease of comparison).
        '''
    )

    # ------------------------------------------------------
    # ----- CountVectorizer/TfidfVectorizer code block -----
    # ------------------------------------------------------
    code_col, output_col = st.beta_columns(2)
    X_count_dict_string = '{index: term for term, index in count_vectorizer.vocabulary_.items()}'
    X_tfidf_dict_string = '{index: term for term, index in tfidf_vectorizer.vocabulary_.items()}'
    vocab_dict_string = '''{'term': count_vectorizer.vocabulary_.keys(), 
                                               'feature index': count_vectorizer.vocabulary_.values()}'''
    with code_col:
        st.subheader('Code:')
        st.write(
            f'''
            ```python
            from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

            corpus = ['This is the first document.',
                      'This document is the second document.',
                      'And this is the third one.',
                      'Is this the first document?']
            
            # Initialize vectorizers
            count_vectorizer = CountVectorizer()
            tfidf_vectorizer = TfidfVectorizer()

            # Print the extracted features
            print_features(count_vectorizer, corpus)
            print_features(tfidf_vectorizer, corpus)
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='newsgroups_TfidfVectorizer_run_button')
    if run_button:

        corpus = ['This is the first document.',
                  'This document is the second document.',
                  'And this is the third one.',
                  'Is this the first document?']
        
        count_vectorizer = CountVectorizer()
        tfidf_vectorizer = TfidfVectorizer()

        X_count = count_vectorizer.fit_transform(corpus)
        X_tfidf = tfidf_vectorizer.fit_transform(corpus)

        X_count_df = pd.DataFrame(data=X_count.toarray()).rename(columns={index: term for term, index in count_vectorizer.vocabulary_.items()})
        X_tfidf_df = pd.DataFrame(data=X_tfidf.toarray()).rename(columns={index: term for term, index in tfidf_vectorizer.vocabulary_.items()})
        #vocabulary_df = pd.Series(data=vectorizer.vocabulary_).sort_values()
        vocabulary_df = pd.DataFrame(data={'term': count_vectorizer.vocabulary_.keys(), 'feature index': count_vectorizer.vocabulary_.values()}).set_index(pd.Index(count_vectorizer.vocabulary_.values())).sort_values(by=['feature index'])
        
        with output_col:
            st.write('**Corpus vocabulary:**')
            # Raw dictionary output
            st.text(count_vectorizer.vocabulary_)
            # Dictionary dataframe
            st.write(vocabulary_df)

            with st.beta_expander('Expand feature matrix output forms'):
                st.write('**Sparse count and TF-IDF array outputs:**')
                st.write(X_count)
                st.write(X_tfidf)
                st.write('**Matrix count and TF-IDF outputs:**')
                st.write(X_count.toarray())
                st.write(X_tfidf.toarray())
                st.write('**DataFrame count and TF-IDF outputs:**')
                st.write(X_count_df)
                st.write(X_tfidf_df)

            for i in range(len(corpus)):
                with st.beta_expander(f'Expand sample {i+1}'):
                    st.write(f'**Sample {i+1} and its count and TF-IDF feature vectors:**')
                    st.text(corpus[i])
                    st.write(X_count_df.iloc[i].to_frame().T)
                    st.write(X_tfidf_df.iloc[i].to_frame().T)

    st.write(
        '''
        Notice how different features with the same count values from `CountVectorizer` get mapped to different TF-IDF values from `TfidfVectorizer`. The TF-IDF use information from the entire corpus to assign values, versus only information from single documents.

        In the code block below, compare the results using different values for the keyword arguments for `TfidfVectorizer` to see their effects. The default options for these are marked.
        '''
    )

    # --------------------------------------------------------
    # ----- TfidfVectorizer keyword arguments code block -----
    # --------------------------------------------------------
    st.subheader('Choose values for norm, use_idf, smooth_idf, and sublinear_tf')
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        norm_choice = st.radio('Choose normalization', options=['L1 norm', 'L2 norm (default)', 'None'])
    with col2:
        use_idf_choice = st.radio('Use IDF weighting?', options=['Yes (default)', 'No'])
    with col3:
        smooth_idf_choice = st.radio('Use IDF smoothing?', options=['Yes (default)', 'No'])
    with col4:
        sublinear_tf_choice = st.radio('Use sublinear TF scaling?', options=['Yes', 'No (default)'])
    
    if norm_choice == 'L1 norm':
        norm_string = "'l1'"
        norm_code = 'l1'
    elif norm_choice == 'L2 norm (default)':
        norm_string = "'l2'"
        norm_code = 'l2'
    elif norm_choice == 'None':
        norm_string = None
        norm_code = None
    
    if use_idf_choice == 'Yes (default)':
        use_idf_code = True
    elif use_idf_choice == 'No':
        use_idf_code = False
    
    if smooth_idf_choice == 'Yes (default)':
        smooth_idf_code = True
    elif smooth_idf_choice == 'No':
        smooth_idf_code = False

    if sublinear_tf_choice == 'Yes':
        sublinear_tf_code = True
    elif sublinear_tf_choice == 'No (default)':
        sublinear_tf_code = False
    
    code_col, output_col = st.beta_columns(2)
    X_tfidf_dict_string = '{index: term for term, index in tfidf_vectorizer.vocabulary_.items()}'
    vocab_dict_string = '''{'term': tfidf_vectorizer.vocabulary_.keys(), 
                                               'feature index': tfidf_vectorizer.vocabulary_.values()}'''
    with code_col:
        st.subheader('Code:')
        st.write(
            f'''
            ```python
            from sklearn.feature_extraction.text import TfidfVectorizer

            corpus = ['This is the first document.',
                      'This document is the second document.',
                      'And this is the third one.',
                      'Is this the first document?']
            
            # Initialize a vectorizer
            tfidf_vectorizer = TfidfVectorizer(norm={norm_string}, use_idf={use_idf_code}, smooth_idf={smooth_idf_code}, sublinear_tf={sublinear_tf_code})
            
            # Print the extracted features
            print_features(tfidf_vectorizer, corpus)
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='newsgroups_TfidfVectorizer_keyword_arguments_run_button')
    if run_button:

        corpus = ['This is the first document.',
                  'This document is the second document.',
                  'And this is the third one.',
                  'Is this the first document?']
        
        tfidf_vectorizer = TfidfVectorizer(norm=norm_code, use_idf=use_idf_code, smooth_idf=smooth_idf_code, sublinear_tf=sublinear_tf_code)

        X_tfidf = tfidf_vectorizer.fit_transform(corpus)

        X_tfidf_df = pd.DataFrame(data=X_tfidf.toarray()).rename(columns={index: term for term, index in tfidf_vectorizer.vocabulary_.items()})
        #vocabulary_df = pd.Series(data=vectorizer.vocabulary_).sort_values()
        vocabulary_df = pd.DataFrame(data={'term': tfidf_vectorizer.vocabulary_.keys(), 'feature index': tfidf_vectorizer.vocabulary_.values()}).set_index(pd.Index(tfidf_vectorizer.vocabulary_.values())).sort_values(by=['feature index'])
        
        with output_col:
            st.write('**Corpus vocabulary:**')
            # Dictionary dataframe
            st.write(vocabulary_df)

            with st.beta_expander('Expand feature matrix output forms'):
                st.write('**Sparse TF-IDF array output:**')
                st.write(X_tfidf)
                st.write('**Matrix TF-IDF output:**')
                st.write(X_tfidf.toarray())
                st.write('**DataFrame TF-IDF output:**')
                st.write(X_tfidf_df)

            for i in range(len(corpus)):
                with st.beta_expander(f'Expand sample {i+1}'):
                    st.write(f'**Sample {i+1} and its TF-IDF feature vector:**')
                    st.text(corpus[i])
                    st.write(X_tfidf_df.iloc[i].to_frame().T)


    st.header('Exploring the 20 Newsgroups dataset')
    st.write(
        '''
        Now that we have explored the different textual feature extraction tools `sklearn` has to offer, we can start to actually use them to explore the 20 Newsgroup dataset! Since the data is labeled by newsgroup topic, we can loop over each category and fit a `CountVectorizer` on each subset of samples. By using the `max_features` keyword argument to set a limit the vocabulary to the topmost frequent terms, we can see which terms appear most frequently in each category to get an idea of how a classification algorithm would distinguish them. Choose the number of features to extract from each category and whether or not to use the `'english'` stopwords list.
        '''
    )

    # -------------------------------------------------
    # ----- CountVectorize by category code block -----
    # -------------------------------------------------
    st.subheader('Choose values for max_features and stop_words')
    col1, col2 = st.beta_columns(2)

    with col1:
        max_features_val = st.slider('Number of features:', min_value=10, max_value=50, step=1)   
    with col2:
        stop_words_choice = st.radio("Use 'stop_words='english'?", options=['Yes', 'No'])
        if stop_words_choice == 'Yes':
            stop_words_val = "'english'"
            stop_words_code = 'english'
        elif stop_words_choice == 'No':
            stop_words_val = None
            stop_words_code = None

    dict_string1 = "{'postings': newsgroups.data, 'category': newsgroups.target}"
    dict_string2 = '{idx: cat for idx, cat in enumerate(newsgroups.target_names)}'
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                f'''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.feature_extraction.text import CountVectorizer
                
                # Load the 20 newsgroup data
                newsgroups = fetch_20newsgroups(subset='all', shuffle='true', random_state=42, remove=('headers', 'footers', 'quotes'))
                
                # Create a dataframe with each text sample and category
                newsgroups_df = pd.DataFrame(data={dict_string1})

                # Replace the category value with corresponding name
                newsgroups_df.category.replace({dict_string2}, inplace=True)
                
                # Initialize top features dataframes
                top_features_df = pd.DataFrame()
                top_features_counts_df = pd.DataFrame()

                # Loop over categories
                for category in newsgroups.target_names:

                    # Initialize a vectorizer with chosen options
                    vectorizer = CountVectorizer(max_features={max_features_val}, stop_words={stop_words_val})

                    # Fit the vectorizer on the specific category
                    X = vectorizer.fit_transform(newsgroups_df[newsgroups_df.category==category].postings)

                    # Sum over the values in each column to get the total term counts
                    X_sum = X.sum(axis=0)

                    # Get the corresponding term from the vocabulary_ attribute
                    word_freq = [(word, X_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

                    # Sort the list of terms by total counts, with highest first
                    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

                    # Add a column to the top features dataframes 
                    top_features_df[category] = word_freq
                    top_features_counts_df[category] = [term[1] for term in word_freq]
                
                # Print distribution of samples by category:
                print('Distribution of newsgroups postings by category:')
                print(newsgroups_df.category.value_counts())

                # Print top features and summary statistics
                print(f'Total term counts of top {max_features_val} features by category:')
                print(top_features_df)

                print('Summary statistics of top {max_features_val} feature counts by category:')
                print(top_features_counts_df.describe()))
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_vectorize_by_category_run_button')
    st.subheader('Output:')
    if run_button:
        # Load the 20 newsgroup data
        newsgroups = fetch_20newsgroups(subset='all',
                                        shuffle='true', 
                                        random_state=42, 
                                        remove=('headers', 'footers', 'quotes'))
        
        # Create a dataframe with each text sample and category
        newsgroups_df = pd.DataFrame(data={'postings': newsgroups.data, 'category': newsgroups.target})

        # Replace the category value with corresponding name
        newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups.target_names)}, inplace=True)

        # Initialize top features dataframes
        top_features_df = pd.DataFrame()
        top_features_counts_df = pd.DataFrame()

        for category in newsgroups.target_names:

            # Initialize a vectorizer with chosen options
            vectorizer = CountVectorizer(max_features=max_features_val, stop_words=stop_words_code)

            # Fit the vectorizer on the specific category
            X = vectorizer.fit_transform(newsgroups_df[newsgroups_df.category==category].postings)

            # Sum over the values in each column to get the total term counts
            X_sum = X.sum(axis=0)

            # Get the corresponding term from the vocabulary_ attribute
            word_freq = [(word, X_sum[0, idx]) for word, idx in vectorizer.vocabulary_.items()]

            # Sort the list of terms by total counts, with highest first
            word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

            # Add a column to the top features dataframes 
            top_features_df[category] = word_freq
            top_features_counts_df[category] = [term[1] for term in word_freq]
        
        # Print distribution of samples by category:
        st.write('**Distribution of newsgroups postings by category:**')
        st.write(newsgroups_df.category.value_counts())
        
        # Print top features
        st.write(f'**Total term counts of top {max_features_val} features by category:**')
        st.write(top_features_df)

        # Print summary statistics by category:
        st.write('**Summary statistics of top feature counts by category:**')
        st.write(top_features_counts_df.describe())
    st.subheader('')

    st.write(
        '''
        By looking at the distribution of samples by category, we see the data is largely uniform and balanced. Most categories have almost 1000 postings, with two having around 800 and one about 630. This means we don't need to do any stratified sampling when we split the data into training and testing sets. 

        Notice that if `stop_words=None` is used, the most frequent terms are very similar across all categories! Category-specific features don't start appearning until after roughly the twenty-most frequent terms. This shows the power of using stop words to eliminate terms that have no category-indicitive power. When `stop_words='english'` is used, each categories most frequent terms are quite unique. This is a good sign for our classification problem. From the statistics summary output of `.describe()`, we see that each category's topmost frequent term have _roughly_ the same number of occurances. One huge outlier is the `'comp.os.ms-windows.misc'` category, where the top term `'ax'` occurs a over sixty thousand times!

        We can repeat the same type of analysis with `TfidfVectorizer`, but this time it does not make sense to split the features into categories before extracting features. This is because weighting with inverse-document frequency requires information from the entire corpus, and not each class only. We can, however, compare how the TF-IDF values are distributed compared with the raw counts. Since only the term frequency in the TF-IDF calculation depends on the individual document, summing over the samples gives the inverse-document frequency weighted _total counts_, which we can directly compare to just the total counts. Run the code below to compare feature values from `CountVectorizer` and `TfidfVectorizer` using `max_features=50` and `stop_words='english'`.
        '''
    )

    # ----------------------------------------------------------------
    # ----- CountVectorizer/TfidfVectorizer newsgroup code block -----
    # ----------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                
                # Load the 20 newsgroup data
                newsgroups = fetch_20newsgroups(subset='all', shuffle='true', random_state=42, remove=('headers', 'footers', 'quotes'))
                
                # Create a dataframe with each text sample and category
                newsgroups_df = pd.DataFrame(data={'postings': newsgroups.data, 'category': newsgroups.target})

                # Replace the category value with corresponding name
                newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups.target_names)}, inplace=True)

                # Initialize top features dataframes
                top_features_df = pd.DataFrame()

                # Instantiate CountVectorizer and TfidfVectorizer objects
                count_vectorizer = CountVectorizer(max_features=50, stop_words='english')
                tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')

                # Fit the vectorizers
                X_count = count_vectorizer.fit_transform(newsgroups_df.postings)
                X_tfidf = tfidf_vectorizer.fit_transform(newsgroups_df.postings)

                # Sum over the values in each columns
                X_count_sum = X_count.sum(axis=0)
                X_tfidf_sum = X_tfidf.sum(axis=0)

                # Get the corresponding term from the vocabulary_ attribute
                word_freq_tfidf = [(word, X_count_sum[0, idx], X_tfidf_sum[0, idx]) for word, idx in count_vectorizer.vocabulary_.items()]

                # Sort the list of terms by total counts, with highest first
                word_freq_tfidf = sorted(word_freq_tfidf, key=lambda x: x[1], reverse=True)

                # Add columns to the top features dataframes
                top_features_df['term'] = [term[0] for term in word_freq_tfidf]
                top_features_df['total count'] = [term[1] for term in word_freq_tfidf]
                top_features_df['total TF-IDF'] = [term[2] for term in word_freq_tfidf]

                # Print top features dataframe
                print('Total term counts and TF-IDF values of top 50 features:')
                print(top_features_df.T)
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_vectorize_run_button')
    st.subheader('Output:')
    if run_button:
        # Load the 20 newsgroup data
        newsgroups = fetch_20newsgroups(subset='all',
                                        shuffle='true', 
                                        random_state=42, 
                                        remove=('headers', 'footers', 'quotes'))
        
        # Create a dataframe with each text sample and category
        newsgroups_df = pd.DataFrame(data={'postings': newsgroups.data, 'category': newsgroups.target})

        # Replace the category value with corresponding name
        newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups.target_names)}, inplace=True)

        # Initialize top features dataframes
        top_features_df = pd.DataFrame()

        # Instantiate CountVectorizer and TfidfVectorizer objects
        count_vectorizer = CountVectorizer(max_features=50, stop_words='english')
        tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')

        # Fit the vectorizers
        X_count = count_vectorizer.fit_transform(newsgroups_df.postings)
        X_tfidf = tfidf_vectorizer.fit_transform(newsgroups_df.postings)

        # Sum over the values in each columns
        X_count_sum = X_count.sum(axis=0)
        X_tfidf_sum = X_tfidf.sum(axis=0)    

        # Get the corresponding term from the vocabulary_ attribute
        word_freq_tfidf = [(word, X_count_sum[0, idx], X_tfidf_sum[0, idx]) for word, idx in count_vectorizer.vocabulary_.items()]

        # Sort the list of terms by total counts, with highest first
        word_freq_tfidf = sorted(word_freq_tfidf, key=lambda x: x[1], reverse=True)

        # Add columns to the top features dataframes
        top_features_df['term'] = [term[0] for term in word_freq_tfidf]
        top_features_df['total count'] = [term[1] for term in word_freq_tfidf]
        top_features_df['total TF-IDF'] = [term[2] for term in word_freq_tfidf]
        
        
        st.write('**Total term counts and TF-IDF values of top 50 features:**')
        st.write(top_features_df.T)
    st.subheader('')
  
    st.write(
        '''
        Now that we've investigated the dataset and understand the outputs of the feature extraction tools from `sklearn`, we can move on to classification. 
        '''
    )
    

    st.header('Choosing a classifier')
    st.write(
        '''
        The choice of `True` or `False` for the `binary` argument of `CountVectorizer` determines if feature vector components are boolean, corresponding to a term appearing in a sample or not, or term frequencies. If boolean, the appropriate variation of Naive-Bayes classifier is `BernoulliNB`. If term frequencies, the appropriate classifier is `MultinomialNB`. When using `TfidfVectorizer`, the appropriate classifier is `MultinomialNB` since term frequencies are weighted by inverse-document frequencies and also normalized. One could use `binary=True`, `use_idf=False`, `sublinear_tf=False`, and `norm=None` to obtain boolean outputs, but this is simply the same as using `CountVectorizer` with `binary=True`.

        The possible Naive-Bayes variations we can use are `MultinomialNB`, `ComplementNB`, `BernoulliNB`, and `GaussianNB`. We won't use `CategoricalNB` since the data is not categorical.
        
        One may wonder if it is truly rigours to use `GaussianNB`, simply based on the fact that the feature vector components are numeric. As explored in the **Mathematics Background** page, `sklearn`'s `GaussianNB` classifier assumes the numeric features are normally distributed when estimates of conditional probabilities for each feature are computed. For term frequencies and TF-IDF values, this is certainly _not_ a good assumption. This is because feature vectors are _sparse_, containing mostly zeros. So distributions of features values, for features that are not stop words, will have huge spikes at zero and small values elsewhere. A Gaussian Naive-Bayes classifier is therefore a very poor choice here, but nonetheless we will test it with the others below.

        Starting out, let's use the default settings for `CountVectorizer` and `TfidfVectorizer` to get term-frequency and TF-IDF feature vectors. We also use a `CountVectorizer` with `binary=True` to get boolean term frequencies. We will use these with multinomial, Bernoulli, and Gaussian classifiers to compare the baseline accuracy. For smoothing of conditional probabilities for terms appearing in the test set but not the train set, we can set `alpha=0.01` in the multinomial and Bernoulli classifiers (recall that the Gaussian classifier does not need smoothing since the normal distribution is never zero). For splitting the data, we will use the `subset='train'` and `subset='test'` keyword argument in the calls to `fetch_20newsgroups`. 

        Note to train the Gaussian classifier, the sparse array output of the feature extractors has to be converted to a dense array by calling the `.toarray()` method. Since the number of features is expected to be huge, this process takes an enormous amount of memory. I have run this on my own machine once, and simply copied the accuracies for the purpose of this demonstration. When you click 'Run Code', the lines to initialize and train the Gaussian classifier have been commented out behind the scenes. Even still, if one runs this code on their own machine offline, it will reproduce similar results (or identical if one uses the same value of the `random_state` argument).
        '''
    )

    # -----------------------------------------------------------------------------
    # ----- Newsgroup classifier baseline performance with `subset` splitting -----
    # -----------------------------------------------------------------------------    
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, GaussianNB

                # Load 'train' and 'test' data subsets
                newsgroups_train = fetch_20newsgroups(subset='train', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
                newsgroups_test = fetch_20newsgroups(subset='test', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

                # Initialize feature extraction tools
                count_vectorizer = CountVectorizer()
                tfidf_vectorizer = TfidfVectorizer()
                count_vectorizer_binary = CountVectorizer(binary=True)

                # Fit the vectorizers by learning the vocabulary of the training set, then compute counts and TF-IDFs
                train_counts = count_vectorizer.fit_transform(newsgroups_train.data)
                train_tfidfs = tfidf_vectorizer.fit_transform(newsgroups_train.data)
                train_counts_binary = count_vectorizer_binary.fit_transform(newsgroups_train.data)

                # Print the number of features extracted from the data:
                print('Number of features extracted:', len(count_vectorizer.vocabulary_))

                # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
                test_counts = count_vectorizer.transform(newsgroups_test.data)
                test_tfidfs = tfidf_vectorizer.transform(newsgroups_test.data)
                test_counts_binary = count_vectorizer_binary.transform(newsgroups_test.data)

                # Initialize the classifiers
                multinomial_counts = MultinomialNB(alpha=0.01)
                multinomial_tfidfs = MultinomialNB(alpha=0.01)
                complement_counts = ComplementNB(alpha=0.01)
                complement_tfidfs = ComplementNB(alpha=0.01)
                bernoulli_counts = BernoulliNB(alpha=0.01)
                gaussian_counts = GaussianNB()
                gaussian_tfidfs = GaussianNB()  

                # Train the classifiers on the training counts and TF-IDFs
                multinomial_counts.fit(train_counts, newsgroups_train.target)
                multinomial_tfidfs.fit(train_tfidfs, newsgroups_train.target)
                complement_counts.fit(train_counts, newsgroups_train.target)
                complement_tfidfs.fit(train_tfidfs, newsgroups_train.target)
                bernoulli_counts.fit(train_counts_binary, newsgroups_train.target)
                gaussian_counts.fit(train_counts.toarray(), newsgroups_train.target)
                gaussian_tfidfs.fit(train_tfidfs.toarray(), newsgroups_train.target)

                # Calculate and print the score on the testing counts and TF-IDFs
                print('Baseline performance accuracies:')
                print('MultinomialNB')
                print('CountVectorizer:', multinomial_counts.score(test_counts, newsgroups_test.target))
                print('TfidfVectorizer:', multinomial_tfidfs.score(test_tfidfs, newsgroups_test.target))
                print('ComplementNB')
                print('CountVectorizer:', complement_counts.score(test_counts, newsgroups_test.target))
                print('TfidfVectorizer:', complement_tfidfs.score(test_tfidfs, newsgroups_test.target))
                print('BernoulliNB')
                print('CountVectorizer:', bernoulli_counts.score(test_counts_binary, newsgroups_test.target))
                print('GaussianNB')
                print('CountVectorizer:', gaussian_counts.score(test_counts, newsgroups_test.target))
                print('TfidfVectorizer:', gaussian_tfidfs.score(test_tfidfs, newsgroups_test.target))
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_classifier_baseline_performance_run_button')
    st.subheader('Output:')
    if run_button:
        # Load 'train' and 'test' data subsets
        newsgroups_train = fetch_20newsgroups(subset='train', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
        newsgroups_test = fetch_20newsgroups(subset='test', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

        # Initialize feature extraction tools
        count_vectorizer = CountVectorizer()
        tfidf_vectorizer = TfidfVectorizer()
        count_vectorizer_binary = CountVectorizer(binary=True)

        # Fit the vectorizers by learning the vocabulary of the training set, then compute counts and TF-IDFs
        train_counts = count_vectorizer.fit_transform(newsgroups_train.data)
        train_tfidfs = tfidf_vectorizer.fit_transform(newsgroups_train.data)
        train_counts_binary = count_vectorizer_binary.fit_transform(newsgroups_train.data)

        # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
        test_counts = count_vectorizer.transform(newsgroups_test.data)
        test_tfidfs = tfidf_vectorizer.transform(newsgroups_test.data)
        test_counts_binary = count_vectorizer_binary.transform(newsgroups_test.data)

        # Initialize the classifiers
        '''
        multinomial_counts = MultinomialNB(alpha=0.01)
        multinomial_tfidfs = MultinomialNB(alpha=0.01)
        complement_counts = ComplementNB(alpha=0.01)
        complement_tfidfs = ComplementNB(alpha=0.01)
        bernoulli_counts = BernoulliNB(alpha=0.01)
        #gaussian_counts = GaussianNB()
        #gaussian_tfidfs = GaussianNB()
        '''
        multinomial_counts = MultinomialNB()
        multinomial_tfidfs = MultinomialNB()
        complement_counts = ComplementNB()
        complement_tfidfs = ComplementNB()
        bernoulli_counts = BernoulliNB()
        #gaussian_counts = GaussianNB()
        #gaussian_tfidfs = GaussianNB()

        # Train the classifiers on the training counts and TF-IDFs
        multinomial_counts.fit(train_counts, newsgroups_train.target)
        multinomial_tfidfs.fit(train_tfidfs, newsgroups_train.target)
        complement_counts.fit(train_counts, newsgroups_train.target)
        complement_tfidfs.fit(train_tfidfs, newsgroups_train.target)
        bernoulli_counts.fit(train_counts_binary, newsgroups_train.target)
        #gaussian_counts.fit(train_counts.toarray(), newsgroups_train.target)
        #gaussian_tfidfs.fit(train_tfidfs.toarray(), newsgroups_train.target)

        # Print the number of features extracted from the data:
        st.write(f'**Number of features extracted:** {len(count_vectorizer.vocabulary_)}')

        # Calculate and print the score on the testing counts and TF-IDFs
        st.write(
            f'''
            **Baseline performance accuracies:**
             - `MultinomialNB`
                - `CountVectorizer`: {multinomial_counts.score(test_counts, newsgroups_test.target)}
                - `TfidfVectorizer`: {multinomial_tfidfs.score(test_tfidfs, newsgroups_test.target)}
             - `ComplementNB`
                - `CountVectorizer`: {complement_counts.score(test_counts, newsgroups_test.target)}
                - `TfidfVectorizer`: {complement_tfidfs.score(test_tfidfs, newsgroups_test.target)}
             - `BernoulliNB`
                - `CountVectorizer`: {bernoulli_counts.score(test_counts_binary, newsgroups_test.target)}
             - `GaussianNB`
                - `CountVectorizer`: 0.5533722782793414
                - `TfidfVectorizer`: 0.5509824747742963
            '''
        )
        '''
        st.write('**Score dataframe:**')
        st.write('Maximum accuracies for each feature vector type are highlighted')
        st.write(score_df.pivot(columns='Features', index='Classifier', values='Accuracy').style.highlight_max(axis=0, color="#ffff99").format(None, na_rep="-"))
        '''
    st.subheader('')
    
    st.write(
        '''
        Using the default settings for the vectorizers (and only adding `binary=True` for the Bernoulli classifier feature vectors), we see that the Multinomial and Complement classifers far outperform the Bernoulli classifier in accuracy! This may be a result of the longer form of textual posts in the dataset, since BernoulliNB classifiers can do well with short-form text content such as SMS messages or tweets, in which specific words chosen to include become much more important. The Gaussian classifier also does poorly (but surprisingly as well as the Multinomial classifier with term-frequency feature vectors). With the default settings, the Complement classifier performs better than the Multinomial classifier byroughly 10% with both term-frequency feature vectors and with TF-IDF feature vectors. For the remainer of our classification, we will consider both Multinomial and Complement classifiers.
        
        With just the default values for the feature extractors, the number of features is over 100k! It is fortunate that we will not need to use the Gaussian classifier, since a dense array of dimension 100,000 features times roughly 15,000 training samples takes a very large amount of memory to create and store. Luckily, we can stick with the sparse feature vectors for the multinomial classifier, which only store the non-zero elements.

        The next task is to optimize the training data by exploring the methods for splitting the data, and by changing how and what features are extracted with `CountVectorizer` and `TfidfVectorizer`.

        Even though TF-IDF values can be computed from term-frequency values by means of a mathematical transformation, we can pretend they are independent variables by combining feature vectors _horizontally_. This is done by essentially tacking on the TF-IDF vector to the end of the term frequency vector. Mathematically, what we are really doing is taking a **_direct sum_** of the vector space where our term frequency vector live with the vector space where our TF-IDF vectors live to arrive at a new vector space whose dimension is the sum of the dimensions of the spaces in the direct sum. In a hand-wavey explaination, consider two vectors `A` and `B` that live in their own orthogaonal vector spaces. We can write them in components as `A = [a1, a2]` and `B = [b1, b2]`. We can make a _direct sum_ of the two vector spaces, which has a combined dimension of four. Then we can write a new four-dimensional vector in this new space as `C = [a1, a2, b1, b2]`. Since our feature vectors are sparse, we can accomplish the horizontal addition using the `hstack` function from the `scipy.sparse` module, loaded as:
        ```python
        from scipy.sparse import hstack
        ```
        In tuning parameters, we will consider _three_ basic choices of feature vectors: Term frequencies (TF) extracted with `CountVectorizer`, TF-IDF features extracted with `TfidfVectorizer`, and their horizontal sum (labeled as TF + TF-IDF) created by calling `hstack` on the TF and TF-IDF vectors.

        Before jumping into all of the available options we have to tweak the feature set, we should first optimize the classifier-specific parameters.
        '''
    )

    st.header('Tuning the classifier parameters')
    st.write(
        '''
        Looking at the documentations for the [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) and [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html), we have two parameters to test out:

         - `alpha`: a parameter controlling the smoothing of conditional probabilities for terms appearing in the test set that were not learned in the training set. The default is `1.0`
         - `fit_prior`: a boolean variable (default is `True`) controlling whether a priors for class probabilities are computed or not. If 'False', then a uniform prior is used. The class priors are essentially the number of samples in the training set that are labeled with each class, divided by the number of training samples. For more informaiton, see the **Mathematics Background** page. If class priors that would be fit from the training data are not indicative of real-world probabilities, then one can use the argument `class_prior` which can take an array of class priors. This argument is ignored for `ComplementNB` except in edge cases where the training data has only one class.
         - `norm`: a boolean argument unique to `ComplementNB` that toggles whether or not a second normalization of the weights is performed (see the documentation). Its default value is `False`.
        
        We can test out these parameters by looping over the continuous `alpha` parameter, for both choices for the `fit_prior` parameter (for the Multinomial classifier) and `norm` parameter (for the Complement classifier). Run the code below to see the results.
        '''
    )

    # -----------------------------------------------------------------------------------------
    # ----- Newsgroup MultinomialNB/ComplementNB performance: tuning classifier arguments -----
    # -----------------------------------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from scipy.sparse import hstack

                # Load 'train' and 'test' data subsets
                newsgroups_train = fetch_20newsgroups(subset='train', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
                newsgroups_test = fetch_20newsgroups(subset='test', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

                # Initialize feature extraction tools
                count_vectorizer = CountVectorizer()
                tfidf_vectorizer = TfidfVectorizer()

                # Fit the vectorizers by learning the vocabulary of the training set, then compute counts and TF-IDFs
                train_counts = count_vectorizer.fit_transform(newsgroups_train.data)
                train_tfidfs = tfidf_vectorizer.fit_transform(newsgroups_train.data)

                # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
                test_counts = count_vectorizer.transform(newsgroups_test.data)
                test_tfidfs = tfidf_vectorizer.transform(newsgroups_test.data)


                # Combine the TF and TF-IDF vectors horizontally with hstack:
                train_counts_tfidfs = hstack([train_counts, train_tfidfs])
                test_counts_tfidfs = hstack([test_counts, test_tfidfs])

                # Create list of values for the alpha parameter
                alpha_list = [round(x/10000, 5) for x in range(1, 10, 1)] \\
                            + [round(x/1000, 5) for x in range(1, 10, 1)] \\
                            + [round(x/100, 5) for x in range(1, 10, 1)] \\
                            + [round(x/10, 5) for x in range(1, 11, 1)]
                
                # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
                accuracy_df = pd.DataFrame(columns=['alpha', 'fit_prior', 'norm', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

                # Loop over alpha values
                for alpha_value in alpha_list:
                    # Initialize the classifiers
                    multinomial_counts_prior = MultinomialNB(alpha=alpha_value, fit_prior=True)
                    multinomial_counts_no_prior = MultinomialNB(alpha=alpha_value, fit_prior=False)
                    multinomial_tfidfs_prior = MultinomialNB(alpha=alpha_value, fit_prior=True)
                    multinomial_tfidfs_no_prior = MultinomialNB(alpha=alpha_value, fit_prior=False)
                    multinomial_counts_tfidfs_prior = MultinomialNB(alpha=alpha_value, fit_prior=True)
                    multinomial_counts_tfidfs_no_prior = MultinomialNB(alpha=alpha_value, fit_prior=False)

                    complement_counts_norm = ComplementNB(alpha=alpha_value, norm=True)
                    complement_counts_no_norm = ComplementNB(alpha=alpha_value, norm=False)
                    complement_tfidfs_norm = ComplementNB(alpha=alpha_value, norm=True)
                    complement_tfidfs_no_norm = ComplementNB(alpha=alpha_value, norm=False)
                    complement_counts_tfidfs_norm = ComplementNB(alpha=alpha_value, norm=True)
                    complement_counts_tfidfs_no_norm = ComplementNB(alpha=alpha_value, norm=False)

                    # Train the classifiers on the training counts and TF-IDFs
                    multinomial_counts_prior.fit(train_counts, newsgroups_train.target)
                    multinomial_counts_no_prior.fit(train_counts, newsgroups_train.target)
                    multinomial_tfidfs_prior.fit(train_tfidfs, newsgroups_train.target)
                    multinomial_tfidfs_no_prior.fit(train_tfidfs, newsgroups_train.target)
                    multinomial_counts_tfidfs_prior.fit(train_counts_tfidfs, newsgroups_train.target)
                    multinomial_counts_tfidfs_no_prior.fit(train_counts_tfidfs, newsgroups_train.target)

                    complement_counts_norm.fit(train_counts, newsgroups_train.target)
                    complement_counts_no_norm.fit(train_counts, newsgroups_train.target)
                    complement_tfidfs_norm.fit(train_tfidfs, newsgroups_train.target)
                    complement_tfidfs_no_norm.fit(train_tfidfs, newsgroups_train.target)
                    complement_counts_tfidfs_norm.fit(train_counts_tfidfs, newsgroups_train.target)
                    complement_counts_tfidfs_no_norm.fit(train_counts_tfidfs, newsgroups_train.target)

                    # Append rows to accuracy dataframe
                    accuracy_df = accuracy_df.append({'alpha': alpha_value, 
                                                      'fit_prior': 'True',
                                                      'norm': '-',
                                                      'Classifier': 'Multinomial',
                                                      'TF accuracy': multinomial_counts_prior.score(test_counts, newsgroups_test.target),
                                                      'TF-IDF accuracy': multinomial_tfidfs_prior.score(test_tfidfs, newsgroups_test.target),
                                                      'TF + TF-IDF accuracy': multinomial_counts_tfidfs_prior.score(test_counts_tfidfs, newsgroups_test.target)}, 
                                                      ignore_index=True)
                    accuracy_df = accuracy_df.append({'alpha': alpha_value, 
                                                      'fit_prior': 'False',
                                                      'norm': '-',
                                                      'Classifier': 'Multinomial',
                                                      'TF accuracy': multinomial_counts_no_prior.score(test_counts, newsgroups_test.target),
                                                      'TF-IDF accuracy': multinomial_tfidfs_no_prior.score(test_tfidfs, newsgroups_test.target), 
                                                      'TF + TF-IDF accuracy': multinomial_counts_tfidfs_no_prior.score(test_counts_tfidfs, newsgroups_test.target)}, 
                                                      ignore_index=True)
                    accuracy_df = accuracy_df.append({'alpha': alpha_value, 
                                                      'fit_prior': '-',
                                                      'norm': 'True',
                                                      'Classifier': 'Complement',
                                                      'TF accuracy': complement_counts_norm.score(test_counts, newsgroups_test.target),
                                                      'TF-IDF accuracy': complement_tfidfs_norm.score(test_tfidfs, newsgroups_test.target),
                                                      'TF + TF-IDF accuracy': complement_counts_tfidfs_norm.score(test_counts_tfidfs, newsgroups_test.target)}, 
                                                      ignore_index=True)
                    accuracy_df = accuracy_df.append({'alpha': alpha_value,
                                                      'fit_prior': '-', 
                                                      'norm': 'False',
                                                      'Classifier': 'Complement',
                                                      'TF accuracy': complement_counts_no_norm.score(test_counts, newsgroups_test.target),
                                                      'TF-IDF accuracy': complement_tfidfs_no_norm.score(test_tfidfs, newsgroups_test.target), 
                                                      'TF + TF-IDF accuracy': complement_counts_tfidfs_no_norm.score(test_counts_tfidfs, newsgroups_test.target)}, 
                                                      ignore_index=True)
                
                # Melt accuracy dataframe for ease of plotting
                accuracy_melt_df = pd.melt(accuracy_df, id_vars=['alpha', 'fit_prior', 'norm', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')
                
                # Initiate plots
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                fig3, ax3 = plt.subplots()
                fig4, ax4 = plt.subplots()

                # Make line plots of accuracies vs min_df and max_df
                sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Multinomial'], x='alpha', y='Accuracy', hue='Features', style='fit_prior', ax=ax1, markers=['o', 's'])
                sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Complement'], x='alpha', y='Accuracy', hue='Features', style='norm', ax=ax2, markers=['o', 's'])
                sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Multinomial'], x='alpha', y='Accuracy', hue='Features', style='fit_prior', ax=ax3, markers=['o', 's'])
                sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Complement'], x='alpha', y='Accuracy', hue='Features', style='norm', ax=ax4, markers=['o', 's'])

                # Add titles
                ax1.set_title('Multinomial classifier accuracy vs alpha: linear scale')
                ax2.set_title('Complement classifier accuracy vs alpha: linear scale')
                ax3.set_title('Multinomial classifier accuracy vs alpha: log-linear scale')
                ax4.set_title('Complement classifier accuracy vs alpha: log-linear scale')

                # Change scaling of second plot to log-linear
                ax3.set(xscale='log', yscale='linear')
                ax4.set(xscale='log', yscale='linear')

                # Add vertical line with second legend by copying axes and hiding them
                ax1_twin = ax1.twinx()
                ax2_twin = ax2.twinx()
                ax3_twin = ax3.twinx()
                ax4_twin = ax4.twinx()
                ax1_twin.axes.yaxis.set_visible(False)
                ax2_twin.axes.yaxis.set_visible(False)
                ax3_twin.axes.yaxis.set_visible(False)
                ax4_twin.axes.yaxis.set_visible(False)

                # Add vertical lines
                ax1_twin.axvline(x=0.03, linestyle='--', color='grey', label='alpha=0.03')
                ax2_twin.axvline(x=0.3, linestyle='--', color='grey', label='alpha=0.3')
                ax3_twin.axvline(x=0.03, linestyle='--', color='grey', label='alpha=0.03')
                ax4_twin.axvline(x=0.3, linestyle='--', color='grey', label='alpha=0.3')
                
                # Add legends and adjust positions
                ax1.legend(loc=(0.1, 0.025))
                ax1_twin.legend()
                ax2_twin.legend(loc=3)
                ax3_twin.legend()
                ax4_twin.legend(loc=2)

                # Show results and plots
                print('Accuracy dataframe:')
                print(accuracy_df)

                print('Multinomial Classifier:')
                fig1.show()
                fig3.show()

                print('Complement Classifier:')
                fig2.show()
                fig4.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_tuning_classifier_run_button')
    st.subheader('Output:')
    if run_button:
        '''
        # Load 'train' and 'test' data subsets
        newsgroups_train = fetch_20newsgroups(subset='train', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
        newsgroups_test = fetch_20newsgroups(subset='test', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

        # Initialize feature extraction tools
        count_vectorizer = CountVectorizer()
        tfidf_vectorizer = TfidfVectorizer()

        # Fit the vectorizers by learning the vocabulary of the training set, then compute counts and TF-IDFs
        train_counts = count_vectorizer.fit_transform(newsgroups_train.data)
        train_tfidfs = tfidf_vectorizer.fit_transform(newsgroups_train.data)

        # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
        test_counts = count_vectorizer.transform(newsgroups_test.data)
        test_tfidfs = tfidf_vectorizer.transform(newsgroups_test.data)


        # Combine the TF and TF-IDF vectors horizontally with hstack:
        train_counts_tfidfs = hstack([train_counts, train_tfidfs])
        test_counts_tfidfs = hstack([test_counts, test_tfidfs])

        # Create list of values for the alpha parameter
        alpha_list = [round(x/10000, 5) for x in range(1, 10, 1)] \
                    + [round(x/1000, 5) for x in range(1, 10, 1)] \
                    + [round(x/100, 5) for x in range(1, 10, 1)] \
                    + [round(x/10, 5) for x in range(1, 11, 1)]
        
        # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
        accuracy_df = pd.DataFrame(columns=['alpha', 'fit_prior', 'norm', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

        # Loop over alpha values
        for alpha_value in alpha_list:
            # Initialize the classifiers
            multinomial_counts_prior = MultinomialNB(alpha=alpha_value, fit_prior=True)
            multinomial_counts_no_prior = MultinomialNB(alpha=alpha_value, fit_prior=False)
            multinomial_tfidfs_prior = MultinomialNB(alpha=alpha_value, fit_prior=True)
            multinomial_tfidfs_no_prior = MultinomialNB(alpha=alpha_value, fit_prior=False)
            multinomial_counts_tfidfs_prior = MultinomialNB(alpha=alpha_value, fit_prior=True)
            multinomial_counts_tfidfs_no_prior = MultinomialNB(alpha=alpha_value, fit_prior=False)

            complement_counts_norm = ComplementNB(alpha=alpha_value, norm=True)
            complement_counts_no_norm = ComplementNB(alpha=alpha_value, norm=False)
            complement_tfidfs_norm = ComplementNB(alpha=alpha_value, norm=True)
            complement_tfidfs_no_norm = ComplementNB(alpha=alpha_value, norm=False)
            complement_counts_tfidfs_norm = ComplementNB(alpha=alpha_value, norm=True)
            complement_counts_tfidfs_no_norm = ComplementNB(alpha=alpha_value, norm=False)

            # Train the classifiers on the training counts and TF-IDFs
            multinomial_counts_prior.fit(train_counts, newsgroups_train.target)
            multinomial_counts_no_prior.fit(train_counts, newsgroups_train.target)
            multinomial_tfidfs_prior.fit(train_tfidfs, newsgroups_train.target)
            multinomial_tfidfs_no_prior.fit(train_tfidfs, newsgroups_train.target)
            multinomial_counts_tfidfs_prior.fit(train_counts_tfidfs, newsgroups_train.target)
            multinomial_counts_tfidfs_no_prior.fit(train_counts_tfidfs, newsgroups_train.target)

            complement_counts_norm.fit(train_counts, newsgroups_train.target)
            complement_counts_no_norm.fit(train_counts, newsgroups_train.target)
            complement_tfidfs_norm.fit(train_tfidfs, newsgroups_train.target)
            complement_tfidfs_no_norm.fit(train_tfidfs, newsgroups_train.target)
            complement_counts_tfidfs_norm.fit(train_counts_tfidfs, newsgroups_train.target)
            complement_counts_tfidfs_no_norm.fit(train_counts_tfidfs, newsgroups_train.target)

            # Append rows to accuracy dataframe
            accuracy_df = accuracy_df.append({'alpha': alpha_value, 
                                              'fit_prior': 'True',
                                              'norm': '-',
                                              'Classifier': 'Multinomial',
                                              'TF accuracy': multinomial_counts_prior.score(test_counts, newsgroups_test.target),
                                              'TF-IDF accuracy': multinomial_tfidfs_prior.score(test_tfidfs, newsgroups_test.target),
                                              'TF + TF-IDF accuracy': multinomial_counts_tfidfs_prior.score(test_counts_tfidfs, newsgroups_test.target)}, 
                                              ignore_index=True)
            accuracy_df = accuracy_df.append({'alpha': alpha_value, 
                                              'fit_prior': 'False',
                                              'norm': '-',
                                              'Classifier': 'Multinomial',
                                              'TF accuracy': multinomial_counts_no_prior.score(test_counts, newsgroups_test.target),
                                              'TF-IDF accuracy': multinomial_tfidfs_no_prior.score(test_tfidfs, newsgroups_test.target), 
                                              'TF + TF-IDF accuracy': multinomial_counts_tfidfs_no_prior.score(test_counts_tfidfs, newsgroups_test.target)}, 
                                              ignore_index=True)
            accuracy_df = accuracy_df.append({'alpha': alpha_value, 
                                              'fit_prior': '-',
                                              'norm': 'True',
                                              'Classifier': 'Complement',
                                              'TF accuracy': complement_counts_norm.score(test_counts, newsgroups_test.target),
                                              'TF-IDF accuracy': complement_tfidfs_norm.score(test_tfidfs, newsgroups_test.target),
                                              'TF + TF-IDF accuracy': complement_counts_tfidfs_norm.score(test_counts_tfidfs, newsgroups_test.target)}, 
                                              ignore_index=True)
            accuracy_df = accuracy_df.append({'alpha': alpha_value,
                                              'fit_prior': '-', 
                                              'norm': 'False',
                                              'Classifier': 'Complement',
                                              'TF accuracy': complement_counts_no_norm.score(test_counts, newsgroups_test.target),
                                              'TF-IDF accuracy': complement_tfidfs_no_norm.score(test_tfidfs, newsgroups_test.target), 
                                              'TF + TF-IDF accuracy': complement_counts_tfidfs_no_norm.score(test_counts_tfidfs, newsgroups_test.target)}, 
                                              ignore_index=True)
        '''
        accuracy_df = pd.DataFrame(data={'alpha': {0: 0.0001, 1: 0.0001, 2: 0.0001, 3: 0.0001, 4: 0.0002, 5: 0.0002, 6: 0.0002, 7: 0.0002, 8: 0.0003, 9: 0.0003, 10: 0.0003, 11: 0.0003, 12: 0.0004, 13: 0.0004, 14: 0.0004, 15: 0.0004, 16: 0.0005, 17: 0.0005, 18: 0.0005, 19: 0.0005, 20: 0.0006, 21: 0.0006, 22: 0.0006, 23: 0.0006, 24: 0.0007, 25: 0.0007, 26: 0.0007, 27: 0.0007, 28: 0.0008, 29: 0.0008, 30: 0.0008, 31: 0.0008, 32: 0.0009, 33: 0.0009, 34: 0.0009, 35: 0.0009, 36: 0.001, 37: 0.001, 38: 0.001, 39: 0.001, 40: 0.002, 41: 0.002, 42: 0.002, 43: 0.002, 44: 0.003, 45: 0.003, 46: 0.003, 47: 0.003, 48: 0.004, 49: 0.004, 50: 0.004, 51: 0.004, 52: 0.005, 53: 0.005, 54: 0.005, 55: 0.005, 56: 0.006, 57: 0.006, 58: 0.006, 59: 0.006, 60: 0.007, 61: 0.007, 62: 0.007, 63: 0.007, 64: 0.008, 65: 0.008, 66: 0.008, 67: 0.008, 68: 0.009, 69: 0.009, 70: 0.009, 71: 0.009, 72: 0.01, 73: 0.01, 74: 0.01, 75: 0.01, 76: 0.02, 77: 0.02, 78: 0.02, 79: 0.02, 80: 0.03, 81: 0.03, 82: 0.03, 83: 0.03, 84: 0.04, 85: 0.04, 86: 0.04, 87: 0.04, 88: 0.05, 89: 0.05, 90: 0.05, 91: 0.05, 92: 0.06, 93: 0.06, 94: 0.06, 95: 0.06, 96: 0.07, 97: 0.07, 98: 0.07, 99: 0.07, 100: 0.08, 101: 0.08, 102: 0.08, 103: 0.08, 104: 0.09, 105: 0.09, 106: 0.09, 107: 0.09, 108: 0.1, 109: 0.1, 110: 0.1, 111: 0.1, 112: 0.2, 113: 0.2, 114: 0.2, 115: 0.2, 116: 0.3, 117: 0.3, 118: 0.3, 119: 0.3, 120: 0.4, 121: 0.4, 122: 0.4, 123: 0.4, 124: 0.5, 125: 0.5, 126: 0.5, 127: 0.5, 128: 0.6, 129: 0.6, 130: 0.6, 131: 0.6, 132: 0.7, 133: 0.7, 134: 0.7, 135: 0.7, 136: 0.8, 137: 0.8, 138: 0.8, 139: 0.8, 140: 0.9, 141: 0.9, 142: 0.9, 143: 0.9, 144: 1.0, 145: 1.0, 146: 1.0, 147: 1.0}, 'fit_prior': {0: 'True', 1: 'False', 2: '-', 3: '-', 4: 'True', 5: 'False', 6: '-', 7: '-', 8: 'True', 9: 'False', 10: '-', 11: '-', 12: 'True', 13: 'False', 14: '-', 15: '-', 16: 'True', 17: 'False', 18: '-', 19: '-', 20: 'True', 21: 'False', 22: '-', 23: '-', 24: 'True', 25: 'False', 26: '-', 27: '-', 28: 'True', 29: 'False', 30: '-', 31: '-', 32: 'True', 33: 'False', 34: '-', 35: '-', 36: 'True', 37: 'False', 38: '-', 39: '-', 40: 'True', 41: 'False', 42: '-', 43: '-', 44: 'True', 45: 'False', 46: '-', 47: '-', 48: 'True', 49: 'False', 50: '-', 51: '-', 52: 'True', 53: 'False', 54: '-', 55: '-', 56: 'True', 57: 'False', 58: '-', 59: '-', 60: 'True', 61: 'False', 62: '-', 63: '-', 64: 'True', 65: 'False', 66: '-', 67: '-', 68: 'True', 69: 'False', 70: '-', 71: '-', 72: 'True', 73: 'False', 74: '-', 75: '-', 76: 'True', 77: 'False', 78: '-', 79: '-', 80: 'True', 81: 'False', 82: '-', 83: '-', 84: 'True', 85: 'False', 86: '-', 87: '-', 88: 'True', 89: 'False', 90: '-', 91: '-', 92: 'True', 93: 'False', 94: '-', 95: '-', 96: 'True', 97: 'False', 98: '-', 99: '-', 100: 'True', 101: 'False', 102: '-', 103: '-', 104: 'True', 105: 'False', 106: '-', 107: '-', 108: 'True', 109: 'False', 110: '-', 111: '-', 112: 'True', 113: 'False', 114: '-', 115: '-', 116: 'True', 117: 'False', 118: '-', 119: '-', 120: 'True', 121: 'False', 122: '-', 123: '-', 124: 'True', 125: 'False', 126: '-', 127: '-', 128: 'True', 129: 'False', 130: '-', 131: '-', 132: 'True', 133: 'False', 134: '-', 135: '-', 136: 'True', 137: 'False', 138: '-', 139: '-', 140: 'True', 141: 'False', 142: '-', 143: '-', 144: 'True', 145: 'False', 146: '-', 147: '-'}, 'norm': {0: '-', 1: '-', 2: 'True', 3: 'False', 4: '-', 5: '-', 6: 'True', 7: 'False', 8: '-', 9: '-', 10: 'True', 11: 'False', 12: '-', 13: '-', 14: 'True', 15: 'False', 16: '-', 17: '-', 18: 'True', 19: 'False', 20: '-', 21: '-', 22: 'True', 23: 'False', 24: '-', 25: '-', 26: 'True', 27: 'False', 28: '-', 29: '-', 30: 'True', 31: 'False', 32: '-', 33: '-', 34: 'True', 35: 'False', 36: '-', 37: '-', 38: 'True', 39: 'False', 40: '-', 41: '-', 42: 'True', 43: 'False', 44: '-', 45: '-', 46: 'True', 47: 'False', 48: '-', 49: '-', 50: 'True', 51: 'False', 52: '-', 53: '-', 54: 'True', 55: 'False', 56: '-', 57: '-', 58: 'True', 59: 'False', 60: '-', 61: '-', 62: 'True', 63: 'False', 64: '-', 65: '-', 66: 'True', 67: 'False', 68: '-', 69: '-', 70: 'True', 71: 'False', 72: '-', 73: '-', 74: 'True', 75: 'False', 76: '-', 77: '-', 78: 'True', 79: 'False', 80: '-', 81: '-', 82: 'True', 83: 'False', 84: '-', 85: '-', 86: 'True', 87: 'False', 88: '-', 89: '-', 90: 'True', 91: 'False', 92: '-', 93: '-', 94: 'True', 95: 'False', 96: '-', 97: '-', 98: 'True', 99: 'False', 100: '-', 101: '-', 102: 'True', 103: 'False', 104: '-', 105: '-', 106: 'True', 107: 'False', 108: '-', 109: '-', 110: 'True', 111: 'False', 112: '-', 113: '-', 114: 'True', 115: 'False', 116: '-', 117: '-', 118: 'True', 119: 'False', 120: '-', 121: '-', 122: 'True', 123: 'False', 124: '-', 125: '-', 126: 'True', 127: 'False', 128: '-', 129: '-', 130: 'True', 131: 'False', 132: '-', 133: '-', 134: 'True', 135: 'False', 136: '-', 137: '-', 138: 'True', 139: 'False', 140: '-', 141: '-', 142: 'True', 143: 'False', 144: '-', 145: '-', 146: 'True', 147: 'False'}, 'Classifier': {0: 'Multinomial', 1: 'Multinomial', 2: 'Complement', 3: 'Complement', 4: 'Multinomial', 5: 'Multinomial', 6: 'Complement', 7: 'Complement', 8: 'Multinomial', 9: 'Multinomial', 10: 'Complement', 11: 'Complement', 12: 'Multinomial', 13: 'Multinomial', 14: 'Complement', 15: 'Complement', 16: 'Multinomial', 17: 'Multinomial', 18: 'Complement', 19: 'Complement', 20: 'Multinomial', 21: 'Multinomial', 22: 'Complement', 23: 'Complement', 24: 'Multinomial', 25: 'Multinomial', 26: 'Complement', 27: 'Complement', 28: 'Multinomial', 29: 'Multinomial', 30: 'Complement', 31: 'Complement', 32: 'Multinomial', 33: 'Multinomial', 34: 'Complement', 35: 'Complement', 36: 'Multinomial', 37: 'Multinomial', 38: 'Complement', 39: 'Complement', 40: 'Multinomial', 41: 'Multinomial', 42: 'Complement', 43: 'Complement', 44: 'Multinomial', 45: 'Multinomial', 46: 'Complement', 47: 'Complement', 48: 'Multinomial', 49: 'Multinomial', 50: 'Complement', 51: 'Complement', 52: 'Multinomial', 53: 'Multinomial', 54: 'Complement', 55: 'Complement', 56: 'Multinomial', 57: 'Multinomial', 58: 'Complement', 59: 'Complement', 60: 'Multinomial', 61: 'Multinomial', 62: 'Complement', 63: 'Complement', 64: 'Multinomial', 65: 'Multinomial', 66: 'Complement', 67: 'Complement', 68: 'Multinomial', 69: 'Multinomial', 70: 'Complement', 71: 'Complement', 72: 'Multinomial', 73: 'Multinomial', 74: 'Complement', 75: 'Complement', 76: 'Multinomial', 77: 'Multinomial', 78: 'Complement', 79: 'Complement', 80: 'Multinomial', 81: 'Multinomial', 82: 'Complement', 83: 'Complement', 84: 'Multinomial', 85: 'Multinomial', 86: 'Complement', 87: 'Complement', 88: 'Multinomial', 89: 'Multinomial', 90: 'Complement', 91: 'Complement', 92: 'Multinomial', 93: 'Multinomial', 94: 'Complement', 95: 'Complement', 96: 'Multinomial', 97: 'Multinomial', 98: 'Complement', 99: 'Complement', 100: 'Multinomial', 101: 'Multinomial', 102: 'Complement', 103: 'Complement', 104: 'Multinomial', 105: 'Multinomial', 106: 'Complement', 107: 'Complement', 108: 'Multinomial', 109: 'Multinomial', 110: 'Complement', 111: 'Complement', 112: 'Multinomial', 113: 'Multinomial', 114: 'Complement', 115: 'Complement', 116: 'Multinomial', 117: 'Multinomial', 118: 'Complement', 119: 'Complement', 120: 'Multinomial', 121: 'Multinomial', 122: 'Complement', 123: 'Complement', 124: 'Multinomial', 125: 'Multinomial', 126: 'Complement', 127: 'Complement', 128: 'Multinomial', 129: 'Multinomial', 130: 'Complement', 131: 'Complement', 132: 'Multinomial', 133: 'Multinomial', 134: 'Complement', 135: 'Complement', 136: 'Multinomial', 137: 'Multinomial', 138: 'Complement', 139: 'Complement', 140: 'Multinomial', 141: 'Multinomial', 142: 'Complement', 143: 'Complement', 144: 'Multinomial', 145: 'Multinomial', 146: 'Complement', 147: 'Complement'}, 'TF accuracy': {0: 0.6328996282527881, 1: 0.6328996282527881, 2: 0.5763409453000531, 3: 0.627854487519915, 4: 0.6350238980350504, 5: 0.6347583643122676, 6: 0.5812533191715348, 7: 0.6294476898566118, 8: 0.6355549654806161, 9: 0.6352894317578333, 10: 0.5870950610727562, 11: 0.6313064259160913, 12: 0.6370154009559214, 13: 0.6364843335103558, 14: 0.5897503983005842, 15: 0.6327668613913967, 16: 0.6386086032926181, 17: 0.6384758364312267, 18: 0.5914763674986724, 19: 0.633563462559745, 20: 0.6400690387679235, 21: 0.6391396707381838, 22: 0.5941317047265002, 23: 0.6350238980350504, 24: 0.6407328730748805, 25: 0.6400690387679235, 26: 0.5961232076473713, 27: 0.6356877323420075, 28: 0.6411311736590547, 29: 0.6409984067976633, 30: 0.597052575677111, 31: 0.6364843335103558, 32: 0.6412639405204461, 33: 0.6407328730748805, 34: 0.5986457780138078, 35: 0.6371481678173128, 36: 0.6413967073818375, 37: 0.6407328730748805, 38: 0.6005045140732873, 39: 0.6371481678173128, 40: 0.6423260754115773, 41: 0.6417950079660116, 42: 0.6082049920339884, 43: 0.6406001062134891, 44: 0.6440520446096655, 45: 0.6439192777482741, 46: 0.612453531598513, 47: 0.643255443441317, 48: 0.6448486457780138, 49: 0.6444503451938396, 50: 0.6161710037174721, 51: 0.6448486457780138, 52: 0.6448486457780138, 53: 0.6444503451938396, 54: 0.6171003717472119, 55: 0.6467073818374933, 56: 0.6449814126394052, 57: 0.6444503451938396, 58: 0.6193574083908656, 59: 0.6476367498672332, 60: 0.645910780669145, 61: 0.6453797132235793, 62: 0.6198884758364313, 63: 0.6484333510355815, 64: 0.6461763143919278, 65: 0.6460435475305364, 66: 0.6220127456186936, 67: 0.6488316516197558, 68: 0.6465746149761019, 69: 0.6464418481147106, 70: 0.6226765799256505, 71: 0.6494954859267127, 72: 0.6460435475305364, 73: 0.6460435475305364, 74: 0.6232076473712161, 75: 0.6498937865108869, 76: 0.6486988847583643, 77: 0.6481678173127987, 78: 0.6336962294211365, 79: 0.6553372278279341, 80: 0.6490971853425385, 81: 0.6490971853425385, 82: 0.6390069038767924, 83: 0.6583908656399363, 84: 0.6500265533722783, 85: 0.6493627190653213, 86: 0.6428571428571429, 87: 0.6601168348380244, 88: 0.6496282527881041, 89: 0.6492299522039299, 90: 0.645246946362188, 91: 0.661975570897504, 92: 0.6493627190653213, 93: 0.6484333510355815, 94: 0.6480350504514073, 95: 0.6625066383430696, 96: 0.6501593202336696, 97: 0.6493627190653213, 98: 0.6492299522039299, 99: 0.6644981412639405, 100: 0.6504248539564524, 101: 0.6500265533722783, 102: 0.6509559214020181, 103: 0.6658258098778544, 104: 0.6492299522039299, 105: 0.6493627190653213, 106: 0.6518852894317578, 107: 0.6660913436006373, 108: 0.6492299522039299, 109: 0.6485661178969729, 110: 0.6533457249070632, 111: 0.667020711630377, 112: 0.6476367498672332, 113: 0.6469729155602761, 114: 0.6593202336696761, 115: 0.6712692511949018, 116: 0.6407328730748805, 117: 0.6396707381837493, 118: 0.6617100371747212, 119: 0.6718003186404673, 120: 0.6253319171534785, 121: 0.6246680828465215, 122: 0.6629049389272438, 123: 0.6715347849176846, 124: 0.6155071694105151, 125: 0.6152416356877324, 126: 0.6615772703133298, 127: 0.6714020180562932, 128: 0.6023632501327668, 129: 0.6014338821030271, 130: 0.6614445034519384, 131: 0.6721986192246415, 132: 0.5884227296866702, 133: 0.588024429102496, 134: 0.6595857673924589, 135: 0.6711364843335104, 136: 0.5747477429633564, 137: 0.574614976101965, 138: 0.6583908656399363, 139: 0.6708709506107275, 140: 0.5608072225172597, 141: 0.5596123207647371, 142: 0.6566648964418481, 143: 0.6695432819968136, 144: 0.5431492299522039, 145: 0.5428836962294211, 146: 0.6549389272437599, 147: 0.6692777482740307}, 'TF-IDF accuracy': {0: 0.6711364843335104, 1: 0.669676048858205, 2: 0.6184280403611259, 3: 0.641927774827403, 4: 0.6765799256505576, 5: 0.6741901221455124, 6: 0.6237387148167818, 7: 0.6484333510355815, 8: 0.6784386617100372, 9: 0.6763143919277749, 10: 0.6287838555496548, 11: 0.6508231545406267, 12: 0.6805629314922995, 13: 0.6792352628783855, 14: 0.631837493361657, 15: 0.6537440254912373, 16: 0.6818906001062135, 17: 0.6806956983536909, 18: 0.6328996282527881, 19: 0.6554699946893255, 20: 0.6817578332448221, 21: 0.6813595326606479, 22: 0.6359532660647902, 23: 0.6570631970260223, 24: 0.6832182687201275, 25: 0.6821561338289963, 26: 0.6384758364312267, 27: 0.6579925650557621, 28: 0.6841476367498672, 29: 0.6834838024429103, 30: 0.6399362719065321, 31: 0.6587891662241104, 32: 0.6849442379182156, 33: 0.6842804036112586, 34: 0.6417950079660116, 35: 0.6597185342538502, 36: 0.6869357408390866, 37: 0.6860063728093467, 38: 0.6427243759957515, 39: 0.6602496016994158, 40: 0.6925119490175252, 41: 0.6909187466808284, 42: 0.6509559214020181, 43: 0.6676845459373341, 44: 0.694105151354222, 45: 0.6917153478491769, 46: 0.6579925650557621, 47: 0.6706054168879447, 48: 0.6960966542750929, 49: 0.6930430164630909, 50: 0.6642326075411578, 51: 0.6748539564524695, 52: 0.6968932554434413, 53: 0.694768985661179, 54: 0.6676845459373341, 55: 0.676712692511949, 56: 0.6982209240573553, 57: 0.6962294211364843, 58: 0.6707381837493361, 59: 0.6802973977695167, 60: 0.6994158258098778, 61: 0.6979553903345725, 62: 0.6728624535315985, 63: 0.6812267657992565, 64: 0.6996813595326606, 65: 0.6983536909187467, 66: 0.6748539564524695, 67: 0.6826872012745618, 68: 0.7006107275624004, 69: 0.699814126394052, 70: 0.6771109930961232, 71: 0.6848114710568242, 72: 0.7002124269782263, 73: 0.700477960701009, 74: 0.6785714285714286, 75: 0.6854753053637812, 76: 0.6995485926712692, 77: 0.6995485926712692, 78: 0.6873340414232607, 79: 0.692113648433351, 80: 0.6990175252257037, 81: 0.7003451938396177, 82: 0.694105151354222, 83: 0.6976898566117897, 84: 0.6974243228890069, 85: 0.7000796601168349, 86: 0.6988847583643123, 87: 0.701141795007966, 88: 0.695432819968136, 89: 0.6982209240573553, 90: 0.701141795007966, 91: 0.7030005310674455, 92: 0.6919808815719597, 93: 0.6960966542750929, 94: 0.7043281996813595, 95: 0.705124800849708, 96: 0.6902549123738715, 97: 0.6937068507700478, 98: 0.7061869357408391, 99: 0.7075146043547531, 100: 0.6882634094530006, 101: 0.6907859798194371, 102: 0.7080456718003186, 103: 0.7084439723844929, 104: 0.687068507700478, 105: 0.6889272437599575, 106: 0.7087095061072757, 107: 0.7100371747211895, 108: 0.6845459373340415, 109: 0.6881306425916092, 110: 0.7103027084439724, 111: 0.7120286776420606, 112: 0.6712692511949018, 113: 0.6744556558682953, 114: 0.7148167817312798, 115: 0.7169410515135423, 116: 0.659984067976633, 117: 0.6625066383430696, 118: 0.7136218799787573, 119: 0.7177376526818906, 120: 0.6492299522039299, 121: 0.6526818906001062, 122: 0.7133563462559745, 123: 0.7170738183749337, 124: 0.6408656399362719, 125: 0.6440520446096655, 126: 0.7129580456718003, 127: 0.7158789166224111, 128: 0.6321030270844398, 129: 0.6383430695698353, 130: 0.7129580456718003, 131: 0.7150823154540626, 132: 0.6249336165693044, 133: 0.630509824747743, 134: 0.7128252788104089, 135: 0.7150823154540626, 136: 0.6176314391927775, 137: 0.6225438130642592, 138: 0.7118959107806692, 139: 0.7156133828996283, 140: 0.6117896972915561, 141: 0.6180297397769516, 142: 0.712161444503452, 143: 0.7144184811471057, 144: 0.6062134891131173, 145: 0.6117896972915561, 146: 0.7112320764737121, 147: 0.7145512480084971}, 'TF + TF-IDF accuracy': {0: 0.6359532660647902, 1: 0.6360860329261816, 2: 0.598380244291025, 3: 0.6302442910249602, 4: 0.6382103027084439, 5: 0.6382103027084439, 6: 0.6055496548061604, 7: 0.6327668613913967, 8: 0.6395379713223579, 9: 0.6396707381837493, 10: 0.6091343600637281, 11: 0.6344928305894849, 12: 0.6412639405204461, 13: 0.6411311736590547, 14: 0.6116569304301647, 15: 0.6356877323420075, 16: 0.6421933085501859, 17: 0.6421933085501859, 18: 0.6129845990440786, 19: 0.6366171003717472, 20: 0.6421933085501859, 21: 0.6424588422729687, 22: 0.6145778013807753, 23: 0.6376792352628784, 24: 0.6433882103027084, 25: 0.6436537440254912, 26: 0.6153744025491238, 27: 0.6391396707381838, 28: 0.6443175783324482, 29: 0.6440520446096655, 30: 0.6157727031332979, 31: 0.6399362719065321, 32: 0.6447158789166224, 33: 0.6441848114710568, 34: 0.6172331386086033, 35: 0.6400690387679235, 36: 0.645246946362188, 37: 0.645246946362188, 38: 0.6184280403611259, 39: 0.6409984067976633, 40: 0.6484333510355815, 41: 0.6484333510355815, 42: 0.6257302177376527, 43: 0.6453797132235793, 44: 0.6480350504514073, 45: 0.6476367498672332, 46: 0.6315719596388741, 47: 0.6483005841741901, 48: 0.648964418481147, 49: 0.6484333510355815, 50: 0.6347583643122676, 51: 0.6505576208178439, 52: 0.6494954859267127, 53: 0.6496282527881041, 54: 0.6366171003717472, 55: 0.6513542219861922, 56: 0.6492299522039299, 57: 0.6496282527881041, 58: 0.6382103027084439, 59: 0.6520180562931492, 60: 0.6505576208178439, 61: 0.6498937865108869, 62: 0.6403345724907064, 63: 0.6528146574614976, 64: 0.6506903876792353, 65: 0.6506903876792353, 66: 0.6415294742432289, 67: 0.6540095592140202, 68: 0.6508231545406267, 69: 0.6510886882634095, 70: 0.6415294742432289, 71: 0.6553372278279341, 72: 0.6510886882634095, 73: 0.6513542219861922, 74: 0.6439192777482741, 75: 0.6562665958576739, 76: 0.6545406266595858, 77: 0.6544078597981944, 78: 0.650292087095061, 79: 0.661311736590547, 80: 0.6557355284121084, 81: 0.6557355284121084, 82: 0.6538767923526287, 83: 0.6642326075411578, 84: 0.6550716941051513, 85: 0.6548061603823686, 86: 0.6563993627190653, 87: 0.6678173127987255, 88: 0.6558682952734998, 89: 0.655602761550717, 90: 0.6591874668082847, 91: 0.6694105151354222, 92: 0.6565321295804567, 93: 0.6561338289962825, 94: 0.6615772703133298, 95: 0.6715347849176846, 96: 0.6558682952734998, 97: 0.6554699946893255, 98: 0.6622411046202867, 99: 0.6725969198088158, 100: 0.6558682952734998, 101: 0.6548061603823686, 102: 0.661975570897504, 103: 0.6737918215613383, 104: 0.6557355284121084, 105: 0.6552044609665427, 106: 0.6629049389272438, 107: 0.6743228890069038, 108: 0.6546733935209772, 109: 0.6548061603823686, 110: 0.6640998406797664, 111: 0.6751194901752523, 112: 0.6433882103027084, 113: 0.643255443441317, 114: 0.665693043016463, 115: 0.6798990971853426, 116: 0.6244025491237387, 117: 0.6245353159851301, 118: 0.6659585767392459, 119: 0.6797663303239512, 120: 0.6044875199150292, 121: 0.6038236856080722, 122: 0.6648964418481147, 123: 0.679368029739777, 124: 0.5835103558151885, 125: 0.5831120552310144, 126: 0.6638343069569835, 127: 0.6796335634625598, 128: 0.5586829527349974, 129: 0.5592140201805629, 130: 0.6630377057886352, 131: 0.6796335634625598, 132: 0.5394317578332448, 133: 0.5389006903876792, 134: 0.6614445034519384, 135: 0.67870419543282, 136: 0.5189856611789697, 137: 0.5188528943175783, 138: 0.6591874668082847, 139: 0.6768454593733404, 140: 0.5001327668613914, 141: 0.5002655337227828, 142: 0.6574614976101965, 143: 0.6752522570366436, 144: 0.4834041423260754, 145: 0.48353690918746683, 146: 0.6571959638874137, 147: 0.6745884227296867}})
        
        # Melt accuracy dataframe for ease of plotting
        accuracy_melt_df = pd.melt(accuracy_df, id_vars=['alpha', 'fit_prior', 'norm', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')
        
        # Initiate plots
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()

        # Make line plots of accuracies vs min_df and max_df
        sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Multinomial'], x='alpha', y='Accuracy', hue='Features', style='fit_prior', ax=ax1, markers=['o', 's'])
        sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Complement'], x='alpha', y='Accuracy', hue='Features', style='norm', ax=ax2, markers=['o', 's'])
        sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Multinomial'], x='alpha', y='Accuracy', hue='Features', style='fit_prior', ax=ax3, markers=['o', 's'])
        sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Complement'], x='alpha', y='Accuracy', hue='Features', style='norm', ax=ax4, markers=['o', 's'])

        # Add titles
        ax1.set_title('Multinomial classifier accuracy vs alpha: linear scale')
        ax2.set_title('Complement classifier accuracy vs alpha: linear scale')
        ax3.set_title('Multinomial classifier accuracy vs alpha: log-linear scale')
        ax4.set_title('Complement classifier accuracy vs alpha: log-linear scale')

        # Change scaling of second plot to log-linear
        ax3.set(xscale='log', yscale='linear')
        ax4.set(xscale='log', yscale='linear')

        # Add vertical line with second legend by copying axes and hiding them
        ax1_twin = ax1.twinx()
        ax2_twin = ax2.twinx()
        ax3_twin = ax3.twinx()
        ax4_twin = ax4.twinx()
        ax1_twin.axes.yaxis.set_visible(False)
        ax2_twin.axes.yaxis.set_visible(False)
        ax3_twin.axes.yaxis.set_visible(False)
        ax4_twin.axes.yaxis.set_visible(False)

        # Add vertical lines
        ax1_twin.axvline(x=0.01, linestyle='--', color='grey', label='alpha=0.01')
        ax2_twin.axvline(x=0.3, linestyle='--', color='grey', label='alpha=0.3')
        ax3_twin.axvline(x=0.01, linestyle='--', color='grey', label='alpha=0.01')
        ax4_twin.axvline(x=0.3, linestyle='--', color='grey', label='alpha=0.3')
        
        # Add legends and adjust positions
        ax1.legend(loc=(0.1, 0.025))
        ax1_twin.legend()
        ax2_twin.legend(loc=3)
        ax3_twin.legend()
        ax4_twin.legend(loc=2)

        # Show results and plots
        st.subheader('**Accuracy dataframe:**')
        st.write(accuracy_df)
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.subheader('**Multinomial Classifier:**')
            st.pyplot(fig1)
            st.pyplot(fig3)
        with output_col2:
            st.subheader('**Complement Classifier:**')
            st.pyplot(fig2)
            st.pyplot(fig4)       
    st.subheader('')

    st.write(
        '''
        From the dataframe holding the accuracies alone, it is hard to see the results, but the plots tell an interesting story. The data plotted in the top plots is the same as that plotted in the bottom, with the latter having log-linear axes scaling to better show the behavior at small values of `alpha`. Interestingly, the performance behavior of `MultinomialNB` and `ComplementNB` with `alpha` is very different.
        
        For the Multinomial classifier (left-hand plots), we a peak in accuracy at small values around `alpha=0.01`, followed by a steady decline towards `alpha=1` (its default value). The difference between the two choices for `fit_prior` is miniscule and only becomes noticible for the TF-IDF feature set and only at larger `alpha` values which are already sub-optimal. From this, we can conclude that optimal settings for `MultinomialNB` are `alpha=0.01` and `fit_prior=True` (which happens to be the default value).

        For the Complement classifier (right-hand plots), we see a sharp increase in small `alpha` values followed by a levelling off in accuracy around `alpha=0.3` and then a gentle decline towards `alpha=1`. The choice for the `norm` argument makes a large difference, with the default value of `False` being superior across each feature vector set. From this, we conclude that the optimal settings for `ComplementNB` are `alpha=0.3` and `norm=False` (the default value).

        Since the best choices for `fit_prior` and `norm` are their respective defaults, we will henceforth initialize the two classifiers as `MultinomialNB(alpha=0.01)` and `ComplementNB(alpha=0.3)`.

        We now move on to tweaking the training set, which consists of two parts: tuning the data splitting and optimizing the feature extraction.
        '''
    )

    st.header('Tuning the training data: data splitting')
    st.write(
        '''
        Recall that the `subset` argument in `fetch_20newsgroups()` can be set to `'train'`, '`test'`, or `'all'`. According to the [dataset website](http://qwone.com/~jason/20Newsgroups/), the `'train'` and `'test'` subsets are separated by a cut in the posting time. To see if this temporal separation has any affect on the classification performance, we can see how the baseline performance compares to using `subset='all'` and then using `sklearn`'s `train_test_split()` to generate random splits.

        The code below loads the data with all three values of `subset`, and then applies `train_test_split` on the `subset='all'` data. Multinomial and Complement classifiers are trained on feature vectors computed with both training sets to compare which performs better.
        '''
    )

    # --------------------------------------------------------------------------
    # ----- Newsgroup baseline performance: `subset` vs. sklearn splitting -----
    # --------------------------------------------------------------------------    
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from scipy.sparse import hstack

                # Load 'train' and 'test' data subsets
                newsgroups_train_subset = fetch_20newsgroups(subset='train', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
                newsgroups_test_subset = fetch_20newsgroups(subset='test', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
                
                # Create a dataframe with each text sample and category
                newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

                # Replace the category value with corresponding name
                newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

                # Split the data
                newsgroups_train_sklearn, newsgroups_test_sklearn = train_test_split(newsgroups_df, train_size=0.75, shuffle=True, random_state=42)
                
                # Initialize feature extraction tools
                count_vectorizer_subset = CountVectorizer()
                tfidf_vectorizer_subset = TfidfVectorizer()
                count_vectorizer_sklearn = CountVectorizer()
                tfidf_vectorizer_sklearn = TfidfVectorizer()

                # Fit the vectorizers by learning the vocabulary of the training set, then compute counts and TF-IDFs
                train_counts_subset = count_vectorizer_subset.fit_transform(newsgroups_train_subset.data)
                train_tfidfs_subset = tfidf_vectorizer_subset.fit_transform(newsgroups_train_subset.data)
                train_counts_sklearn = count_vectorizer_sklearn.fit_transform(newsgroups_train_sklearn.postings)
                train_tfidfs_sklearn = tfidf_vectorizer_sklearn.fit_transform(newsgroups_train_sklearn.postings)

                # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
                test_counts_subset = count_vectorizer_subset.transform(newsgroups_test_subset.data)
                test_tfidfs_subset = tfidf_vectorizer_subset.transform(newsgroups_test_subset.data)
                test_counts_sklearn = count_vectorizer_sklearn.transform(newsgroups_test_sklearn.postings)
                test_tfidfs_sklearn = tfidf_vectorizer_sklearn.transform(newsgroups_test_sklearn.postings)

                # Combine the TF and TF-IDF vectors horizontally with hstack:
                train_counts_tfidfs_subset = hstack([train_counts_subset, train_tfidfs_subset])
                train_counts_tfidfs_sklearn = hstack([train_counts_sklearn, train_tfidfs_sklearn])
                test_counts_tfidfs_subset = hstack([test_counts_subset, test_tfidfs_subset])
                test_counts_tfidfs_sklearn = hstack([test_counts_sklearn, test_tfidfs_sklearn])                

                # Initialize the classifiers
                alpha_mnb=0.01
                alpha_cnb=0.3

                multinomial_counts_subset = MultinomialNB(alpha=alpha_mnb)
                multinomial_tfidfs_subset = MultinomialNB(alpha=alpha_mnb)
                multinomial_counts_tfidfs_subset = MultinomialNB(alpha=alpha_mnb)
                multinomial_counts_sklearn = MultinomialNB(alpha=alpha_mnb)
                multinomial_tfidfs_sklearn = MultinomialNB(alpha=alpha_mnb)
                multinomial_counts_tfidfs_sklearn = MultinomialNB(alpha=alpha_mnb)

                complement_counts_subset = ComplementNB(alpha=alpha_cnb)
                complement_tfidfs_subset = ComplementNB(alpha=alpha_cnb)
                complement_counts_tfidfs_subset = ComplementNB(alpha=alpha_cnb)
                complement_counts_sklearn = ComplementNB(alpha=alpha_cnb)
                complement_tfidfs_sklearn = ComplementNB(alpha=alpha_cnb)
                complement_counts_tfidfs_sklearn = ComplementNB(alpha=alpha_cnb)

                # Train the classifiers on the training counts and TF-IDFs
                multinomial_counts_subset.fit(train_counts_subset, newsgroups_train_subset.target)
                multinomial_tfidfs_subset.fit(train_tfidfs_subset, newsgroups_train_subset.target)
                multinomial_counts_tfidfs_subset.fit(train_counts_tfidfs_subset, newsgroups_train_subset.target)
                multinomial_counts_sklearn.fit(train_counts_sklearn, newsgroups_train_sklearn.category)
                multinomial_tfidfs_sklearn.fit(train_tfidfs_sklearn, newsgroups_train_sklearn.category)
                multinomial_counts_tfidfs_sklearn.fit(train_counts_tfidfs_sklearn, newsgroups_train_sklearn.category)

                complement_counts_subset.fit(train_counts_subset, newsgroups_train_subset.target)
                complement_tfidfs_subset.fit(train_tfidfs_subset, newsgroups_train_subset.target)
                complement_counts_tfidfs_subset.fit(train_counts_tfidfs_subset, newsgroups_train_subset.target)
                complement_counts_sklearn.fit(train_counts_sklearn, newsgroups_train_sklearn.category)
                complement_tfidfs_sklearn.fit(train_tfidfs_sklearn, newsgroups_train_sklearn.category)
                complement_counts_tfidfs_sklearn.fit(train_counts_tfidfs_sklearn, newsgroups_train_sklearn.category)

                # Calculate the score on the testing counts and TF-IDFs
                print('MultinomialNB aseline performance accuracies:')
                print('Splitting with `subset` keyword argument:')
                print('TF vectors:', multinomial_counts_subset.score(test_counts_subset, newsgroups_test_subset.target))
                print('TF-IDF vectors:', multinomial_tfidfs_subset.score(test_tfidfs_subset, newsgroups_test_subset.target))
                print('TF + TF-IDF vectors:', multinomial_counts_tfidfs_subset.score(test_counts_tfidfs_subset, newsgroups_test_subset.target))
                print('Splitting with `train_test_split`:')
                print('TF vectors:', multinomial_counts_sklearn.score(test_counts_sklearn, newsgroups_test_sklearn.category))
                print('TF-IDF vectors:', multinomial_tfidfs_sklearn.score(test_tfidfs_sklearn, newsgroups_test_sklearn.category))
                print('TF + TF-IDF vectors:', multinomial_counts_tfidfs_sklearn.score(test_counts_tfidfs_sklearn, newsgroups_test_sklearn.category))

                print('ComplementNB aseline performance accuracies:')
                print('Splitting with `subset` keyword argument:')
                print('TF vectors:', complement_counts_subset.score(test_counts_subset, newsgroups_test_subset.target))
                print('TF-IDF vectors:', complement_tfidfs_subset.score(test_tfidfs_subset, newsgroups_test_subset.target))
                print('TF + TF-IDF vectors:', complement_counts_tfidfs_subset.score(test_counts_tfidfs_subset, newsgroups_test_subset.target))
                print('Splitting with `train_test_split`:')
                print('TF vectors:', complement_counts_sklearn.score(test_counts_sklearn, newsgroups_test_sklearn.category))
                print('TF-IDF vectors:', complement_tfidfs_sklearn.score(test_tfidfs_sklearn, newsgroups_test_sklearn.category))
                print('TF + TF-IDF vectors:', complement_counts_tfidfs_sklearn.score(test_counts_tfidfs_sklearn, newsgroups_test_sklearn.category))
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_baseline_splitting_method_run_button')
    st.subheader('Output:')
    if run_button:
        # Load 'train', 'test', and 'all' data subsets
        newsgroups_train_subset = fetch_20newsgroups(subset='train', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
        newsgroups_test_subset = fetch_20newsgroups(subset='test', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
        newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

        # Create a dataframe with each text sample and category
        newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})
        
        # Replace the category value with corresponding name
        newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

        # Split the data
        newsgroups_train_sklearn, newsgroups_test_sklearn = train_test_split(newsgroups_df, train_size=0.75, shuffle=True, random_state=42)

        # Initialize feature extraction tools
        count_vectorizer_subset = CountVectorizer()
        tfidf_vectorizer_subset = TfidfVectorizer()
        count_vectorizer_sklearn = CountVectorizer()
        tfidf_vectorizer_sklearn = TfidfVectorizer()

        # Fit the vectorizers by learning the vocabulary of the training set, then compute counts and TF-IDFs
        train_counts_subset = count_vectorizer_subset.fit_transform(newsgroups_train_subset.data)
        train_tfidfs_subset = tfidf_vectorizer_subset.fit_transform(newsgroups_train_subset.data)
        train_counts_sklearn = count_vectorizer_sklearn.fit_transform(newsgroups_train_sklearn.postings)
        train_tfidfs_sklearn = tfidf_vectorizer_sklearn.fit_transform(newsgroups_train_sklearn.postings)

        # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
        test_counts_subset = count_vectorizer_subset.transform(newsgroups_test_subset.data)
        test_tfidfs_subset = tfidf_vectorizer_subset.transform(newsgroups_test_subset.data)
        test_counts_sklearn = count_vectorizer_sklearn.transform(newsgroups_test_sklearn.postings)
        test_tfidfs_sklearn = tfidf_vectorizer_sklearn.transform(newsgroups_test_sklearn.postings)

        # Combine the TF and TF-IDF vectors horizontally with hstack:
        train_counts_tfidfs_subset = hstack([train_counts_subset, train_tfidfs_subset])
        train_counts_tfidfs_sklearn = hstack([train_counts_sklearn, train_tfidfs_sklearn])
        test_counts_tfidfs_subset = hstack([test_counts_subset, test_tfidfs_subset])
        test_counts_tfidfs_sklearn = hstack([test_counts_sklearn, test_tfidfs_sklearn])                

        # Initialize the classifiers
        alpha_mnb=0.01
        alpha_cnb=0.3

        multinomial_counts_subset = MultinomialNB(alpha=alpha_mnb)
        multinomial_tfidfs_subset = MultinomialNB(alpha=alpha_mnb)
        multinomial_counts_tfidfs_subset = MultinomialNB(alpha=alpha_mnb)
        multinomial_counts_sklearn = MultinomialNB(alpha=alpha_mnb)
        multinomial_tfidfs_sklearn = MultinomialNB(alpha=alpha_mnb)
        multinomial_counts_tfidfs_sklearn = MultinomialNB(alpha=alpha_mnb)

        complement_counts_subset = ComplementNB(alpha=alpha_cnb)
        complement_tfidfs_subset = ComplementNB(alpha=alpha_cnb)
        complement_counts_tfidfs_subset = ComplementNB(alpha=alpha_cnb)
        complement_counts_sklearn = ComplementNB(alpha=alpha_cnb)
        complement_tfidfs_sklearn = ComplementNB(alpha=alpha_cnb)
        complement_counts_tfidfs_sklearn = ComplementNB(alpha=alpha_cnb)

        # Train the classifiers on the training counts and TF-IDFs
        multinomial_counts_subset.fit(train_counts_subset, newsgroups_train_subset.target)
        multinomial_tfidfs_subset.fit(train_tfidfs_subset, newsgroups_train_subset.target)
        multinomial_counts_tfidfs_subset.fit(train_counts_tfidfs_subset, newsgroups_train_subset.target)
        multinomial_counts_sklearn.fit(train_counts_sklearn, newsgroups_train_sklearn.category)
        multinomial_tfidfs_sklearn.fit(train_tfidfs_sklearn, newsgroups_train_sklearn.category)
        multinomial_counts_tfidfs_sklearn.fit(train_counts_tfidfs_sklearn, newsgroups_train_sklearn.category)

        complement_counts_subset.fit(train_counts_subset, newsgroups_train_subset.target)
        complement_tfidfs_subset.fit(train_tfidfs_subset, newsgroups_train_subset.target)
        complement_counts_tfidfs_subset.fit(train_counts_tfidfs_subset, newsgroups_train_subset.target)
        complement_counts_sklearn.fit(train_counts_sklearn, newsgroups_train_sklearn.category)
        complement_tfidfs_sklearn.fit(train_tfidfs_sklearn, newsgroups_train_sklearn.category)
        complement_counts_tfidfs_sklearn.fit(train_counts_tfidfs_sklearn, newsgroups_train_sklearn.category)
       
        # Calculate the score on the testing counts and TF-IDFs
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.write(
                f'''
                **MultinomialNB baseline performance accuracies:**
                - Splitting with `subset` keyword argument:
                    - TF vectors: {multinomial_counts_subset.score(test_counts_subset, newsgroups_test_subset.target)}
                    - TF-IDF vectors: {multinomial_tfidfs_subset.score(test_tfidfs_subset, newsgroups_test_subset.target)}
                    - TF + TF-IDF vectors: {multinomial_counts_tfidfs_subset.score(test_counts_tfidfs_subset, newsgroups_test_subset.target)}
                - Splitting with `train_test_split`:
                    - TF vectors: {multinomial_counts_sklearn.score(test_counts_sklearn, newsgroups_test_sklearn.category)}
                    - TF-IDF vectors: {multinomial_tfidfs_sklearn.score(test_tfidfs_sklearn, newsgroups_test_sklearn.category)}
                    - TF + TF-IDF vectors: {multinomial_counts_tfidfs_sklearn.score(test_counts_tfidfs_sklearn, newsgroups_test_sklearn.category)}
                '''
            )
        with output_col2:
            st.write(
                f'''
                **ComplementNB baseline performance accuracies:**
                - Splitting with `subset` keyword argument:
                    - TF vectors: {complement_counts_subset.score(test_counts_subset, newsgroups_test_subset.target)}
                    - TF-IDF vectors: {complement_tfidfs_subset.score(test_tfidfs_subset, newsgroups_test_subset.target)}
                    - TF + TF-IDF vectors: {complement_counts_tfidfs_subset.score(test_counts_tfidfs_subset, newsgroups_test_subset.target)}
                - Splitting with `train_test_split`:
                    - TF vectors: {complement_counts_sklearn.score(test_counts_sklearn, newsgroups_test_sklearn.category)}
                    - TF-IDF vectors: {complement_tfidfs_sklearn.score(test_tfidfs_sklearn, newsgroups_test_sklearn.category)}
                    - TF + TF-IDF vectors: {complement_counts_tfidfs_sklearn.score(test_counts_tfidfs_sklearn, newsgroups_test_sklearn.category)}
                '''
            )
    st.subheader('')

    st.write(
        '''
        Keeping the default settings for the vectorizers and only changing how the data is split, we gain about 5% accuracies using `train_test_split` rather than the `subset` argument, for each feature vector set, for both classifiers! This demonstrates quite conclusively that the built-in method for splitting of the data, which was based on a cut-off in the posting time, actually results in poorer classification than using a simple fraction of the shuffled dataset with `train_test_split`. Henceforth, we will disregard the provided `subset` method and stick with `train_test_split`. We still have some freedom to modify the fraction of the data set aside for testing, since `train_test_split` includes a `train_size` argument.
        
        First, let's define a function to load and split the data that takes a positional argument for the training size:
        ```python
        def load_and_split_newsgroups(training_size):
            # Load all of the data
            newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test      
        ```
        when we want to tune the training size, all we need to do is modify the argument in the call to `load_and_split_newsgroups`. The process for training and scoring the classifiers can also be condensed with another custom convenience function:
        ```python
        def train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the accuracy results
            accuracy_dict = {}
            accuracy_dict['Multinomial'] = {}
            accuracy_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial_counts = MultinomialNB(alpha=alpha_mnb)
            multinomial_tfidfs = MultinomialNB(alpha=alpha_mnb)
            multinomial_counts_tfidfs = MultinomialNB(alpha=alpha_mnb)
            complement_counts = ComplementNB(alpha=alpha_cnb)
            complement_tfidfs = ComplementNB(alpha=alpha_cnb)
            complement_counts_tfidfs = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training counts and TF-IDFs
            multinomial_counts.fit(train_counts, train_labels)
            multinomial_tfidfs.fit(train_tfidfs, train_labels)
            multinomial_counts_tfidfs.fit(train_counts_tfidfs, train_labels)
            complement_counts.fit(train_counts, train_labels)
            complement_tfidfs.fit(train_tfidfs, train_labels)
            complement_counts_tfidfs.fit(train_counts_tfidfs, train_labels)

            # Add the accuracies to the dictionary:
            accuracy_dict['Multinomial']['TF accuracy'] = multinomial_counts.score(test_counts, test_labels)
            accuracy_dict['Multinomial']['TF-IDF accuracy'] = multinomial_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Multinomial']['TF + TF-IDF accuracy'] = multinomial_counts_tfidfs.score(test_counts_tfidfs, test_labels)
            accuracy_dict['Complement']['TF accuracy'] = complement_counts.score(test_counts, test_labels)
            accuracy_dict['Complement']['TF-IDF accuracy'] = complement_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Complement']['TF + TF-IDF accuracy'] = complement_counts_tfidfs.score(test_counts_tfidfs, test_labels)

            # Return the dictionary of results:
            return accuracy_dict
        ```
        
        The code below computes the accuracy of multinomial classifiers trained on term frequencies and TF-IDF values for a range of training size values.
        '''
    )

    # --------------------------------------------------------------------
    # ----- Newsgroup performance: tuning the `train_size` parameter -----
    # --------------------------------------------------------------------  
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from scipy.sparse import hstack

                # Define list of training sizes to test
                training_sizes = [round(0.05 + x/100, 3) for x in range(0, 91, 5)]
                count_accuracies = []
                tfidf_accuracies = []
                count_tfidf_accuracies = []

                # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
                accuracy_df = pd.DataFrame(columns=['train_size', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

                # Loop over the training size
                for training_size in training_sizes:
                    
                    # Load and split the data
                    newsgroups_train, newsgroups_test = load_and_split_newsgroups(training_size)

                    # Instantiate feature extractors
                    count_vectorizer = CountVectorizer()
                    tfidf_vectorizer = TfidfVectorizer()

                    # Fit the vectorizers by learning the vocabulary of the 
                    # training set, then compute counts and TF-IDFs
                    train_counts = count_vectorizer.fit_transform(newsgroups_train.postings)
                    train_tfidfs = tfidf_vectorizer.fit_transform(newsgroups_train.postings)
                    train_counts_tfidfs = hstack([train_counts, train_tfidfs])

                    # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
                    test_counts = count_vectorizer.transform(newsgroups_test.postings)
                    test_tfidfs = tfidf_vectorizer.transform(newsgroups_test.postings)
                    test_counts_tfidfs = hstack([test_counts, test_tfidfs])

                    # Train and score classifiers
                    accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

                    # Append accuracies to accuracy dataframe
                    accuracy_df = accuracy_df.append({'train_size': training_size, 
                                                      'Classifier': 'Multinomial', 
                                                      'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                                      'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                                      'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                                      ignore_index=True)
                    accuracy_df = accuracy_df.append({'train_size': training_size, 
                                                      'Classifier': 'Complement', 
                                                      'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                                      'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                                      'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                                      ignore_index=True)
                
                # Melt accuracy dataframe for ease of plotting
                accuracy_melt_df = pd.melt(accuracy_df, id_vars=['train_size', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')

                # Make line plots of accuracy vs train_size
                fig, ax = plt.subplots()

                sns.lineplot(data=accuracy_melt_df, x='train_size', y='Accuracy', hue='Features', style='Classifier', ax=ax, markers=['o', 's'])
                ax.set_title('Classifier accuracy vs training size')

                # Add vertical line with second legend by copying axes and hiding them
                ax_twin = ax.twinx()
                ax_twin.axes.yaxis.set_visible(False)

                # Add vertical line and legends
                ax_twin.axvline(x=0.75, linestyle='--', color='grey', label='train_size=0.75')
                ax_twin.legend(loc=2)
                ax.legend(loc=(0.3, 0.05))

                # Show results and plots
                print('Accuracy vs. train_size results:')
                print(accuracy_df)                
                fig.show() 
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_tuning_train_size_run_button')
    st.subheader('Output:')
    if run_button:
        '''
        def load_and_split_newsgroups(training_size):
            # Load all of the data
            newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the accuracy results
            accuracy_dict = {}
            accuracy_dict['Multinomial'] = {}
            accuracy_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial_counts = MultinomialNB(alpha=alpha_mnb)
            multinomial_tfidfs = MultinomialNB(alpha=alpha_mnb)
            multinomial_counts_tfidfs = MultinomialNB(alpha=alpha_mnb)
            complement_counts = ComplementNB(alpha=alpha_cnb)
            complement_tfidfs = ComplementNB(alpha=alpha_cnb)
            complement_counts_tfidfs = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training counts and TF-IDFs
            multinomial_counts.fit(train_counts, train_labels)
            multinomial_tfidfs.fit(train_tfidfs, train_labels)
            multinomial_counts_tfidfs.fit(train_counts_tfidfs, train_labels)
            complement_counts.fit(train_counts, train_labels)
            complement_tfidfs.fit(train_tfidfs, train_labels)
            complement_counts_tfidfs.fit(train_counts_tfidfs, train_labels)

            # Add the accuracies to the dictionary:
            accuracy_dict['Multinomial']['TF accuracy'] = multinomial_counts.score(test_counts, test_labels)
            accuracy_dict['Multinomial']['TF-IDF accuracy'] = multinomial_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Multinomial']['TF + TF-IDF accuracy'] = multinomial_counts_tfidfs.score(test_counts_tfidfs, test_labels)
            accuracy_dict['Complement']['TF accuracy'] = complement_counts.score(test_counts, test_labels)
            accuracy_dict['Complement']['TF-IDF accuracy'] = complement_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Complement']['TF + TF-IDF accuracy'] = complement_counts_tfidfs.score(test_counts_tfidfs, test_labels)

            # Return the dictionary of results:
            return accuracy_dict

        # Define list of training sizes to test
        training_sizes = [round(0.05 + x/100, 3) for x in range(0, 91, 5)]

        # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
        accuracy_df = pd.DataFrame(columns=['train_size', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

        # Loop over the training size
        for training_size in training_sizes:
            
            # Load and split the data
            newsgroups_train, newsgroups_test = load_and_split_newsgroups(training_size)

            # Instantiate feature extractors
            count_vectorizer = CountVectorizer()
            tfidf_vectorizer = TfidfVectorizer()

            # Fit the vectorizers by learning the vocabulary of the 
            # training set, then compute counts and TF-IDFs
            train_counts = count_vectorizer.fit_transform(newsgroups_train.postings)
            train_tfidfs = tfidf_vectorizer.fit_transform(newsgroups_train.postings)
            train_counts_tfidfs = hstack([train_counts, train_tfidfs])

            # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
            test_counts = count_vectorizer.transform(newsgroups_test.postings)
            test_tfidfs = tfidf_vectorizer.transform(newsgroups_test.postings)
            test_counts_tfidfs = hstack([test_counts, test_tfidfs])

            # Train and score classifiers
            accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

            # Append accuracies to accuracy dataframe
            accuracy_df = accuracy_df.append({'train_size': training_size, 
                                              'Classifier': 'Multinomial', 
                                              'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
            accuracy_df = accuracy_df.append({'train_size': training_size, 
                                              'Classifier': 'Complement', 
                                              'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
        '''            
        accuracy_df = pd.DataFrame(data={'train_size': {0: 0.05, 1: 0.05, 2: 0.1, 3: 0.1, 4: 0.15, 5: 0.15, 6: 0.2, 7: 0.2, 8: 0.25, 9: 0.25, 10: 0.3, 11: 0.3, 12: 0.35, 13: 0.35, 14: 0.4, 15: 0.4, 16: 0.45, 17: 0.45, 18: 0.5, 19: 0.5, 20: 0.55, 21: 0.55, 22: 0.6, 23: 0.6, 24: 0.65, 25: 0.65, 26: 0.7, 27: 0.7, 28: 0.75, 29: 0.75, 30: 0.8, 31: 0.8, 32: 0.85, 33: 0.85, 34: 0.9, 35: 0.9, 36: 0.95, 37: 0.95}, 'Classifier': {0: 'Multinomial', 1: 'Complement', 2: 'Multinomial', 3: 'Complement', 4: 'Multinomial', 5: 'Complement', 6: 'Multinomial', 7: 'Complement', 8: 'Multinomial', 9: 'Complement', 10: 'Multinomial', 11: 'Complement', 12: 'Multinomial', 13: 'Complement', 14: 'Multinomial', 15: 'Complement', 16: 'Multinomial', 17: 'Complement', 18: 'Multinomial', 19: 'Complement', 20: 'Multinomial', 21: 'Complement', 22: 'Multinomial', 23: 'Complement', 24: 'Multinomial', 25: 'Complement', 26: 'Multinomial', 27: 'Complement', 28: 'Multinomial', 29: 'Complement', 30: 'Multinomial', 31: 'Complement', 32: 'Multinomial', 33: 'Complement', 34: 'Multinomial', 35: 'Complement', 36: 'Multinomial', 37: 'Complement'}, 'TF accuracy': {0: 0.5096626452189454, 1: 0.5666890080428955, 2: 0.5922650630821837, 3: 0.6309987029831388, 4: 0.6185393258426967, 5: 0.6491260923845193, 6: 0.6385222524374876, 7: 0.661869072096571, 8: 0.6519278386982668, 9: 0.66947293951185, 10: 0.6636094898809975, 11: 0.6800576063063746, 12: 0.6716734693877551, 13: 0.6868571428571428, 14: 0.6802263883975946, 15: 0.697117085249381, 16: 0.6880185220914529, 17: 0.7050935751495273, 18: 0.6919240157062506, 19: 0.7076302663695214, 20: 0.6997995519396297, 21: 0.7124159886805801, 22: 0.6981031967104391, 23: 0.7164080116726357, 24: 0.7018341670456268, 25: 0.7213885099287555, 26: 0.6995047753802618, 27: 0.7258577997877609, 28: 0.7064940577249575, 29: 0.7236842105263158, 30: 0.7045092838196286, 31: 0.7228116710875332, 32: 0.708171206225681, 33: 0.7247966041740361, 34: 0.7114058355437666, 35: 0.7204244031830239, 36: 0.7073170731707317, 37: 0.7062566277836692}, 'TF-IDF accuracy': {0: 0.5451295799821269, 1: 0.5922698838248436, 2: 0.6297016861219196, 3: 0.6646032307510906, 4: 0.667103620474407, 5: 0.6929463171036204, 6: 0.690787291901572, 7: 0.7103535186044969, 8: 0.7013795542978423, 9: 0.7213300318358684, 10: 0.7147729856742212, 11: 0.7321306753581445, 12: 0.7235918367346938, 13: 0.7372244897959184, 14: 0.7290413866289353, 15: 0.7432790944464096, 16: 0.7354813814393208, 17: 0.7471541578236542, 18: 0.7404223707948636, 19: 0.7544306484134564, 20: 0.750147388279684, 21: 0.7592265063082184, 22: 0.7523544236636158, 23: 0.7611089003846664, 24: 0.7592845232681522, 25: 0.764893133242383, 26: 0.7589317297488504, 27: 0.7647683056243367, 28: 0.7650679117147708, 29: 0.7644312393887945, 30: 0.7649867374005305, 31: 0.7628647214854112, 32: 0.77431906614786, 33: 0.7714892111779271, 34: 0.7708222811671087, 35: 0.7649867374005305, 36: 0.767762460233298, 37: 0.7518557794273595}, 'TF + TF-IDF accuracy': {0: 0.5144660411081322, 1: 0.5729445933869526, 2: 0.5969814880320717, 3: 0.6389576700860747, 4: 0.6256554307116104, 5: 0.6560549313358303, 6: 0.6446242621211117, 7: 0.6720833056974199, 8: 0.6568800848956491, 9: 0.6793066855323665, 10: 0.6675509740013643, 11: 0.6902903054650193, 12: 0.6772244897959183, 13: 0.6966530612244898, 14: 0.6861513972408914, 15: 0.7063141139016625, 16: 0.6925525757283426, 17: 0.7143546208759406, 18: 0.6950015918497294, 19: 0.7179242279528812, 20: 0.704044334394529, 21: 0.7233816766890697, 22: 0.7046027324578856, 23: 0.7254277755670513, 24: 0.7082006972866455, 25: 0.7301803850234956, 26: 0.7064025468694729, 27: 0.7338167668906969, 28: 0.7098896434634975, 29: 0.7313242784380306, 30: 0.7108753315649867, 31: 0.7281167108753316, 32: 0.7155995755217545, 33: 0.7357622921825256, 34: 0.716710875331565, 35: 0.7315649867374006, 36: 0.7179215270413574, 37: 0.7136797454931071}})

        # Melt accuracy dataframe for ease of plotting
        accuracy_melt_df = pd.melt(accuracy_df, id_vars=['train_size', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')

        # Make line plots of accuracy vs train_size
        fig, ax = plt.subplots()

        sns.lineplot(data=accuracy_melt_df, x='train_size', y='Accuracy', hue='Features', style='Classifier', ax=ax, markers=['o', 's'])
        ax.set_title('Classifier accuracy vs training size')

        # Add vertical line with second legend by copying axes and hiding them
        ax_twin = ax.twinx()
        ax_twin.axes.yaxis.set_visible(False)

        # Add vertical line and legends
        ax_twin.axvline(x=0.75, linestyle='--', color='grey', label='train_size=0.75')
        ax_twin.legend(loc=2)
        ax.legend(loc=(0.3, 0.05))

        # Show results and plots
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.subheader('**Accuracy vs. train_size results:**')
            st.write(accuracy_df)
        with output_col2:
            st.pyplot(fig)        
    st.subheader('')

    st.write(
        '''
        As expected, as the proportion of data used in training increases, the accuracy of the classifiers increases. Below 20%, the accuracies are very poor, but increase rapidly as the learned vocabulary expands. However, we see diminishing returns with more and more training data, and the accuracies start leveling off, becoming flat around 70-80%. We will use a value of `train_size=0.75` as our optimal choice.

        Next, we will try to find the optimal training features by tweaking the parameters of `CountVectorizer` and `TfidfVectorizer`. As we saw in the above plot, the accuracy for TF feature vectors was always lower than for TF-IDF feature vectors, across all values of `train_size`. The combination of TF and TF-IDF vectors resulted in moderate improvement over TF vectors alone, but was still significantly below the performance of TF-IDF vectors alone. As explored above, there is a lot of control over the features that are extracted using `CountVectorizer` and `TfidfVectorizer`, and we should see what their affect is on the classification accuracy.
        '''
    )

    st.header('Tuning the training data: feature engineering')
    st.write(
        '''
        Now that we've compared different classifiers and identitified the best option (the multinomial classifier), and an optimal training size, let's try to find the optimal features to classify with. These choices are made with the initialization of the `CountVectorizer` and `TfidfVectorizer` objects, which both return sparse feature vectors. Recall the options we have at our disposal:
        
        **Options shared by `CountVectorizer` and `TfidfVectorizer`:**
         - `ngram_range`: takes a tuple in the form of `(min, max)`, default is `(1, 1)`
         - `min_df` and `max_df`: takes a float between `0.0` and `1.0` or an integer. Sets a lower limit (`min_df`) or upper limit (`max_df`) on the document frequency of features extracted in the vocabulary. The defaults are the integer `1` for `min_df` corresponding to a minimum document frequency of one and the float `1.0` for `max_df` corresponding to the maximum document frequency of 100%
         - `max_features`: takes an integer (or `None`, the default) corresponding to the number of most frequently occuring terms in the corpus allowed in the vocabulary.
         - 'stop_words': takes either a list of words to exclude, or the string 'english' for a preset list of common English language words. The default is `None`.
        
        **Options unique to `TfidfVectorizer`:**
         - `norm`: option to choose either `'l1'` or `'l2'` (default) normalization, or `None` for no normalization
         - `use_idf`: boolean to toggle inverse document frequency weighting (default is `True`)
         - `smooth_idf`: boolean to toggle smoothing of inverse document frequency value (default is `True`)
         - `sublinear_tf`: boolean to apply logarithmic term frequency scaling (default is `False`)
        
        We will work under the assumption that the values for these arguments are, to a large degree, independent. This means we can tune one at a time, finding the optimal choice before moving to the next, and fixing the parameters to their optimal value as we find them. This greatly cuts down the overall number of points in _parameter space_ we need to test: if we have four parameters, and test out ten choices for each, to test every combination would require 10000 trials (ten to the fourth power). If those features are independent, we only need to test out 40 points, (four parameters times ten points each).
        
        We first explore the numeric options shared by the feature extraction tools, and later explore the extra options provided with `TfidfVectorizer`. To do so, let's define another convenience function for extracting features. We can use a keyword argument variable that takes a dictionary to hold the various options, and set its default values to those of the feature extractors:

        ```python
        def extract_features(training_data, testing_data, opts={'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0, 'max_features': None, 'stop_words': 'english'}):
            # Instantiate feature extractors
            count_vectorizer = CountVectorizer(ngram_range=opts['ngram_range'], min_df=opts['min_df'], max_df=opts['max_df'], max_features=opts['max_features', stop_words=opts['stop_words'])
            tfidf_vectorizer = TfidfVectorizer(ngram_range=opts['ngram_range'], min_df=opts['min_df'], max_df=opts['max_df'], max_features=opts['max_features'], stop_words=opts['stop_words'])

            # Fit the vectorizers by learning the vocabulary of the training set, then compute counts and TF-IDFs
            train_counts = count_vectorizer.fit_transform(training_data)
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
            test_counts = count_vectorizer.transform(testing_data)
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_counts, test_counts, train_tfidfs, test_tfidfs
        ```
        '''
    )
    st.subheader('Enlarging the feature space: tuning `ngram_range`')
    st.write(
        '''
        First, let's try different values for `ngram_range`. Below we test out all of the possible values of `ngram_range=(ngram_min, ngram_max)`, from `ngram_min = 1` up to `ngram_max = 4` and all combinations in-between. For each of these values, we split the data using our custom `load_and_split_newsgroups()` function, keeping 75% of the data for training. Then we extract term frequencies and TF-IDF values using our `extract_features()` function. We also compute their horizontal combination using `hstack` to test out a third set of features. For each of the three feature sets, we train a `MultinomialNB` classifier and compute the accuracy on the held-out testing sets. The number of features extracted in each of the three feature sets is included in a column in the `accuracy_df`. As you will see when you run the code below, when `ngram_range` includes a small lower bound and large upper bound, the number of terms in the vocabulary is very large. This computation was completed once, and the code to do so on your own machine is shown in the code block below. For this demonstration, the resulting `accuracy_df` dataframe is simply copied and displayed. 

        We can better display the data in the dataframe by manipulating it with `numpy` into matrix forms, and then plotting heatmaps of the matrix values for each of the three sets of features. The code block below uses a custom function `matrix_heatmap` that wraps the `seaborn` `heatmap` plotting function for easier control of the various options. Expand the below code to see how `matrix_heatmap` is defined:
        '''
    )
    with st.beta_expander('Expand definition of custom `matrix_heatmap` visualization function'):
        st.write(
            '''
            ```python
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            import seaborn as sns
            import numpy as np

            def matrix_heatmap(matrix, options={'x_labels': [], 'y_labels': [], 'annotation_format': 'd', 'color_map': 'Blues', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0, 'title_axis_labels': ('Default Title', 'Default x-axis label', 'Default y-axis label'), 'rotate x_tick_labels': False}):

                # Create matrix figure
                fig, ax = plt.subplots()

                # Resize the figure if dimension is larger than a cutoff so that heatmap annotations
                # do not overflow their cells (chosen via testing to be 7)
                max_dimension = max(len(options['x_labels']), len(options['y_labels']))
                if max_dimension >= 7:
                    fig.set_size_inches(max_dimension, max_dimension)
                
                # ----------------------------------
                # ----- Create seaborn heatmap -----
                # ----------------------------------

                # For custom colorbar, note cbar=False keyword is used to prevent duplicate colorbars

                # Set custom vmin, vmax if 'custom_range' option is True
                if options['custom_range']:
                    ax = sns.heatmap(matrix, annot=True, fmt=options['annotation_format'], ax = ax, cmap=options['color_map'], vmin=options['vmin_vmax'][0], vmax=options['vmin_vmax'][1], center=options['center'], square=True, cbar=False)
                else:
                    ax = sns.heatmap(matrix, annot=True, fmt=options['annotation_format'], ax = ax, cmap=options['color_map'], center=options['center'], square=True, cbar=False)
                
                # Format colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.25)
                cbar = plt.colorbar(ax.collections[0], cax=cax)
                cbar.outline.set_edgecolor('black')

                # make heatmap frame visible
                for _, spine in ax.spines.items():
                    spine.set_visible(True)
                
                # title, axis labels, and ticks
                ax.set_title(options['title_axis_labels'][0])
                ax.set_xlabel(options['title_axis_labels'][1])
                ax.set_ylabel(options['title_axis_labels'][2])
                
                # Make y-axis labels horizontal
                ax.yaxis.set_tick_params(rotation=0)

                ax.xaxis.set_ticklabels(options['x_labels'])
                ax.yaxis.set_ticklabels(options['y_labels'])

                # Rotate x-axis ticks and align
                if options['rotate x_tick_labels']:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha="right", rotation_mode="anchor")

                # Return the figure
                return fig
            ```
            '''
        )

    # -----------------------------------------------------
    # ----- Newsgroup performance: tuning ngram_range -----
    # -----------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from scipy.sparse import hstack

                # Define values of ngram_range to loop over
                ngram_ranges = []
                max_ngram = 4
                for ngram_max in range(1, max_ngram+1):
                    for ngram_min in range(1, ngram_max+1):
                        ngram_ranges.append((ngram_min, ngram_max))

                # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
                accuracy_df = pd.DataFrame(columns=['ngram_range', 'num features', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

                # Load and split the data
                newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)

                # Loop over ngram_ranges
                for ngrams in ngram_ranges:
                    
                    # Extract features
                    extract_options = {'ngram_range': ngrams, 'min_df': 1, 'max_df': 1.0, 'max_features': None, 'stop_words': 'english'}
                    train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
                
                    # Horizontally stack TF and TF-IDF vectors
                    train_counts_tfidfs = hstack([train_counts, train_tfidfs])
                    test_counts_tfidfs = hstack([test_counts, test_tfidfs])

                    # Train and score classifiers
                    accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

                    # Append accuracies to accuracy dataframe
                    accuracy_df = accuracy_df.append({'ngram_range': ngrams,
                                                      'num features': train_counts.shape[1],
                                                      'Classifier': 'Multinomial', 
                                                      'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                                      'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                                      'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                                      ignore_index=True)
                    accuracy_df = accuracy_df.append({'ngram_range': ngrams,
                                                      'num features': train_counts.shape[1], 
                                                      'Classifier': 'Complement', 
                                                      'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                                      'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                                      'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                                      ignore_index=True)
                
                # Make numpy arrays for heatmap plots, initialized with np.nan values
                accuracy_arrays = {}
                accuracy_arrays['Multinomial'] = {'TF': np.empty((4,4))*np.nan, 'TF-IDF': np.empty((4,4))*np.nan, 'TF + TF-IDF': np.empty((4,4))*np.nan}
                accuracy_arrays['Complement'] = {'TF': np.empty((4,4))*np.nan, 'TF-IDF': np.empty((4,4))*np.nan, 'TF + TF-IDF': np.empty((4,4))*np.nan}
                num_features_array = np.empty((4,4))*np.nan
                
                # Split the accuracy dataframe by classifier
                # Loop over index in accuracy dataframe to get access to different ngram_range values
                # Use the values in ngram_range for indexing, need to subtract 1 from them since numpy arrays are zero-indexed
                accuracy_subset = accuracy_df[accuracy_df['Classifier']=='Multinomial']
                for idx in range(len(accuracy_subset)):
                    ngram = accuracy_subset['ngram_range'].iloc[idx]
                    accuracy_arrays['Multinomial']['TF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF accuracy'].iloc[idx]
                    accuracy_arrays['Multinomial']['TF-IDF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF-IDF accuracy'].iloc[idx]
                    accuracy_arrays['Multinomial']['TF + TF-IDF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF + TF-IDF accuracy'].iloc[idx]
                accuracy_subset = accuracy_df[accuracy_df['Classifier']=='Complement']   
                for idx in range(len(accuracy_subset)):
                    ngram = accuracy_subset['ngram_range'].iloc[idx]
                    accuracy_arrays['Complement']['TF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF accuracy'].iloc[idx]
                    accuracy_arrays['Complement']['TF-IDF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF-IDF accuracy'].iloc[idx]
                    accuracy_arrays['Complement']['TF + TF-IDF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF + TF-IDF accuracy'].iloc[idx]
                    num_features_array[ngram[0]-1][ngram[1]-1] = accuracy_subset['num features'].iloc[idx]
                
                # Flip arrays horizontally, then take transpose to get indexing right
                accuracy_arrays['Multinomial']['TF'] = np.fliplr(accuracy_arrays['Multinomial']['TF']).T
                accuracy_arrays['Multinomial']['TF-IDF'] = np.fliplr(accuracy_arrays['Multinomial']['TF-IDF']).T
                accuracy_arrays['Multinomial']['TF + TF-IDF'] = np.fliplr(accuracy_arrays['Multinomial']['TF + TF-IDF']).T
                accuracy_arrays['Complement']['TF'] = np.fliplr(accuracy_arrays['Complement']['TF']).T
                accuracy_arrays['Complement']['TF-IDF'] = np.fliplr(accuracy_arrays['Complement']['TF-IDF']).T
                accuracy_arrays['Complement']['TF + TF-IDF'] = np.fliplr(accuracy_arrays['Complement']['TF + TF-IDF']).T
                num_features_array = np.fliplr(num_features_array).T
                
                # Melt accuracy dataframe for ease of plotting
                accuracy_melt_df = pd.melt(accuracy_df, id_vars=['ngram_range', 'num features', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')

                # Initialize figures and axes objects 
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                fig3, ax3 = plt.subplots()
                fig4, ax4 = plt.subplots()
                fig5, ax5 = plt.subplots()
                fig6, ax6 = plt.subplots()
                fig7, ax7 = plt.subplots()
                fig8, ax8 = plt.subplots()

                # Make line plot of accuracies vs ngram size
                # Extract rows where ngram_min = ngram_max for plot
                ngram_accuracy_df = accuracy_melt_df[accuracy_melt_df['ngram_range'].apply(lambda x: True if x[0]==x[1] else False)].reset_index(drop=True)
                ngram_accuracy_df['ngram size'] = ngram_accuracy_df['ngram_range'].apply(lambda x: x[0])

                sns.lineplot(data=ngram_accuracy_df, x='ngram size', y='Accuracy', hue='Features', style='Classifier', ax=ax1, markers=['o', 's'])
                
                ax1.set_title('Classifier accuracy vs single ngram size')
                ax1.set_xticks(ngram_accuracy_df['ngram size'])
                
                # Make heatmaps of the num_features_array and accuracy arrays using custom matrix_heatmap function
                fig2 = matrix_heatmap(num_features_array.tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'bone_r', 'custom_range': False, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Number of extracted features', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

                fig3 = matrix_heatmap(accuracy_arrays['Multinomial']['TF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Multinomial accuracy array heatmap:\n  TF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

                fig4 = matrix_heatmap(accuracy_arrays['Multinomial']['TF-IDF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Multinomial accuracy array heatmap:\n  TF-IDF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

                fig5 = matrix_heatmap(accuracy_arrays['Multinomial']['TF + TF-IDF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Multinomial accuracy array heatmap:\n  TF + TF-IDF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

                fig6 = matrix_heatmap(accuracy_arrays['Complement']['TF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Complement accuracy array heatmap:\n  TF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

                fig7 = matrix_heatmap(accuracy_arrays['Complement']['TF-IDF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Complement accuracy array heatmap:\n  TF-IDF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

                fig8 = matrix_heatmap(accuracy_arrays['Complement']['TF + TF-IDF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Complement accuracy array heatmap:\n  TF + TF-IDF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})
            
                print('Accuracies vs. `ngram_range` results:')
                print('Accuracy vs `ngram_range` dataframe:')
                print(accuracy_df)
                fig1.show()
                fig2.show()

                print('Heat maps of accuracies: Multinomial Classifier')
                output_col1, output_col2, output_col3 = st.beta_columns(3)
                fig3.show()
                fig4.show()
                fig5.show()

                print('Heat maps of accuracies: Complement Classifier')
                output_col1, output_col2, output_col3 = st.beta_columns(3)
                fig6.show()
                fig7.show()
                fig8.show()  
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_tuning_ngram_range_run_button')
    st.subheader('Output:')
    if run_button:
        '''
        def load_and_split_newsgroups(training_size):
            # Load all of the data
            newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def extract_features(training_data, testing_data, opts={'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0, 'max_features': None, 'stop_words': None}):
            # Instantiate feature extractors
            count_vectorizer = CountVectorizer(ngram_range=opts['ngram_range'], min_df=opts['min_df'], max_df=opts['max_df'], max_features=opts['max_features'], stop_words=opts['stop_words'])
            tfidf_vectorizer = TfidfVectorizer(ngram_range=opts['ngram_range'], min_df=opts['min_df'], max_df=opts['max_df'], max_features=opts['max_features'], stop_words=opts['stop_words'])

            # Fit the vectorizers by learning the vocabulary of the 
            # training set, then compute counts and TF-IDFs
            train_counts = count_vectorizer.fit_transform(training_data)
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
            test_counts = count_vectorizer.transform(testing_data)
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_counts, test_counts, train_tfidfs, test_tfidfs
        
        def train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the accuracy results
            accuracy_dict = {}
            accuracy_dict['Multinomial'] = {}
            accuracy_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial_counts = MultinomialNB(alpha=alpha_mnb)
            multinomial_tfidfs = MultinomialNB(alpha=alpha_mnb)
            multinomial_counts_tfidfs = MultinomialNB(alpha=alpha_mnb)
            complement_counts = ComplementNB(alpha=alpha_cnb)
            complement_tfidfs = ComplementNB(alpha=alpha_cnb)
            complement_counts_tfidfs = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training counts and TF-IDFs
            multinomial_counts.fit(train_counts, train_labels)
            multinomial_tfidfs.fit(train_tfidfs, train_labels)
            multinomial_counts_tfidfs.fit(train_counts_tfidfs, train_labels)
            complement_counts.fit(train_counts, train_labels)
            complement_tfidfs.fit(train_tfidfs, train_labels)
            complement_counts_tfidfs.fit(train_counts_tfidfs, train_labels)

            # Add the accuracies to the dictionary:
            accuracy_dict['Multinomial']['TF accuracy'] = multinomial_counts.score(test_counts, test_labels)
            accuracy_dict['Multinomial']['TF-IDF accuracy'] = multinomial_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Multinomial']['TF + TF-IDF accuracy'] = multinomial_counts_tfidfs.score(test_counts_tfidfs, test_labels)
            accuracy_dict['Complement']['TF accuracy'] = complement_counts.score(test_counts, test_labels)
            accuracy_dict['Complement']['TF-IDF accuracy'] = complement_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Complement']['TF + TF-IDF accuracy'] = complement_counts_tfidfs.score(test_counts_tfidfs, test_labels)

            # Return the dictionary of results:
            return accuracy_dict
        
        # Define values of ngram_range to loop over
        ngram_ranges = []
        max_ngram = 4
        for ngram_max in range(1, max_ngram+1):
            for ngram_min in range(1, ngram_max+1):
                ngram_ranges.append((ngram_min, ngram_max))

        # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
        accuracy_df = pd.DataFrame(columns=['ngram_range', 'num features', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

        # Load and split the data
        newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)

        # Loop over ngram_ranges
        for ngrams in ngram_ranges:
            
            # Extract features
            extract_options = {'ngram_range': ngrams, 'min_df': 1, 'max_df': 1.0, 'max_features': None, 'stop_words': 'english'}
            train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
        
            # Horizontally stack TF and TF-IDF vectors
            train_counts_tfidfs = hstack([train_counts, train_tfidfs])
            test_counts_tfidfs = hstack([test_counts, test_tfidfs])

            # Train and score classifiers
            accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

            # Append accuracies to accuracy dataframe
            accuracy_df = accuracy_df.append({'ngram_range': ngrams,
                                              'num features': train_counts.shape[1],
                                              'Classifier': 'Multinomial', 
                                              'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
            accuracy_df = accuracy_df.append({'ngram_range': ngrams,
                                              'num features': train_counts.shape[1], 
                                              'Classifier': 'Complement', 
                                              'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
        '''
        accuracy_df = pd.DataFrame(data={'ngram_range': {0: (1, 1), 1: (1, 1), 2: (1, 2), 3: (1, 2), 4: (2, 2), 5: (2, 2), 6: (1, 3), 7: (1, 3), 8: (2, 3), 9: (2, 3), 10: (3, 3), 11: (3, 3), 12: (1, 4), 13: (1, 4), 14: (2, 4), 15: (2, 4), 16: (3, 4), 17: (3, 4), 18: (4, 4), 19: (4, 4)}, 'num features': {0: 108779, 1: 108779, 2: 1069606, 3: 1069606, 4: 960827, 5: 960827, 6: 2252271, 7: 2252271, 8: 2143492, 9: 2143492, 10: 1182665, 11: 1182665, 12: 3456572, 13: 3456572, 14: 3347793, 15: 3347793, 16: 2386966, 17: 2386966, 18: 1204301, 19: 1204301}, 'Classifier': {0: 'Multinomial', 1: 'Complement', 2: 'Multinomial', 3: 'Complement', 4: 'Multinomial', 5: 'Complement', 6: 'Multinomial', 7: 'Complement', 8: 'Multinomial', 9: 'Complement', 10: 'Multinomial', 11: 'Complement', 12: 'Multinomial', 13: 'Complement', 14: 'Multinomial', 15: 'Complement', 16: 'Multinomial', 17: 'Complement', 18: 'Multinomial', 19: 'Complement'}, 'TF accuracy': {0: 0.708616298811545, 1: 0.7245331069609507, 2: 0.7368421052631579, 3: 0.7635823429541596, 4: 0.6443123938879457, 5: 0.6808149405772496, 6: 0.7387521222410866, 7: 0.7627334465195246, 8: 0.6411290322580645, 9: 0.6733870967741935, 10: 0.4093803056027165, 11: 0.4144736842105263, 12: 0.7374787775891342, 13: 0.7618845500848896, 14: 0.6394312393887945, 15: 0.6716893039049237, 16: 0.4083191850594228, 17: 0.41362478777589134, 18: 0.28204584040747027, 19: 0.27695246179966043}, 'TF-IDF accuracy': {0: 0.767402376910017, 1: 0.766553480475382, 2: 0.7769524617996605, 3: 0.7828947368421053, 4: 0.6714770797962648, 5: 0.6803904923599321, 6: 0.7735568760611206, 7: 0.7828947368421053, 8: 0.6642614601018676, 9: 0.6735993208828522, 10: 0.4121392190152801, 11: 0.40598471986417656, 12: 0.7710101867572157, 13: 0.780560271646859, 14: 0.6632003395585738, 15: 0.6695670628183361, 16: 0.41086587436332767, 17: 0.40407470288624786, 18: 0.2814091680814941, 19: 0.2767402376910017}, 'TF + TF-IDF accuracy': {0: 0.7166808149405772, 1: 0.7298387096774194, 2: 0.7423599320882852, 3: 0.7671901528013583, 4: 0.6502546689303905, 5: 0.6816638370118846, 6: 0.7412988115449916, 7: 0.7663412563667232, 8: 0.6443123938879457, 9: 0.6765704584040747, 10: 0.4100169779286927, 11: 0.4142614601018676, 12: 0.7432088285229203, 13: 0.764855687606112, 14: 0.6434634974533107, 15: 0.6740237691001698, 16: 0.4085314091680815, 17: 0.41277589134125636, 18: 0.28204584040747027, 19: 0.27716468590831916}})

        # Make numpy arrays for heatmap plots, initialized with np.nan values
        accuracy_arrays = {}
        accuracy_arrays['Multinomial'] = {'TF': np.empty((4,4))*np.nan, 'TF-IDF': np.empty((4,4))*np.nan, 'TF + TF-IDF': np.empty((4,4))*np.nan}
        accuracy_arrays['Complement'] = {'TF': np.empty((4,4))*np.nan, 'TF-IDF': np.empty((4,4))*np.nan, 'TF + TF-IDF': np.empty((4,4))*np.nan}
        num_features_array = np.empty((4,4))*np.nan
        
        # Split the accuracy dataframe by classifier
        # Loop over index in accuracy dataframe to get access to different ngram_range values
        # Use the values in ngram_range for indexing, need to subtract 1 from them since numpy arrays are zero-indexed
        accuracy_subset = accuracy_df[accuracy_df['Classifier']=='Multinomial']
        for idx in range(len(accuracy_subset)):
            ngram = accuracy_subset['ngram_range'].iloc[idx]
            accuracy_arrays['Multinomial']['TF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF accuracy'].iloc[idx]
            accuracy_arrays['Multinomial']['TF-IDF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF-IDF accuracy'].iloc[idx]
            accuracy_arrays['Multinomial']['TF + TF-IDF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF + TF-IDF accuracy'].iloc[idx]
        accuracy_subset = accuracy_df[accuracy_df['Classifier']=='Complement']   
        for idx in range(len(accuracy_subset)):
            ngram = accuracy_subset['ngram_range'].iloc[idx]
            accuracy_arrays['Complement']['TF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF accuracy'].iloc[idx]
            accuracy_arrays['Complement']['TF-IDF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF-IDF accuracy'].iloc[idx]
            accuracy_arrays['Complement']['TF + TF-IDF'][ngram[0]-1][ngram[1]-1] = accuracy_subset['TF + TF-IDF accuracy'].iloc[idx]
            num_features_array[ngram[0]-1][ngram[1]-1] = accuracy_subset['num features'].iloc[idx]
        
        # Flip arrays horizontally, then take transpose to get indexing right
        accuracy_arrays['Multinomial']['TF'] = np.fliplr(accuracy_arrays['Multinomial']['TF']).T
        accuracy_arrays['Multinomial']['TF-IDF'] = np.fliplr(accuracy_arrays['Multinomial']['TF-IDF']).T
        accuracy_arrays['Multinomial']['TF + TF-IDF'] = np.fliplr(accuracy_arrays['Multinomial']['TF + TF-IDF']).T
        accuracy_arrays['Complement']['TF'] = np.fliplr(accuracy_arrays['Complement']['TF']).T
        accuracy_arrays['Complement']['TF-IDF'] = np.fliplr(accuracy_arrays['Complement']['TF-IDF']).T
        accuracy_arrays['Complement']['TF + TF-IDF'] = np.fliplr(accuracy_arrays['Complement']['TF + TF-IDF']).T
        num_features_array = np.fliplr(num_features_array).T
        
        # Melt accuracy dataframe for ease of plotting
        accuracy_melt_df = pd.melt(accuracy_df, id_vars=['ngram_range', 'num features', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')

        # Initialize figures and axes objects 
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        fig7, ax7 = plt.subplots()
        fig8, ax8 = plt.subplots()

        # Make line plot of accuracies vs ngram size
        # Extract rows where ngram_min = ngram_max for plot
        ngram_accuracy_df = accuracy_melt_df[accuracy_melt_df['ngram_range'].apply(lambda x: True if x[0]==x[1] else False)].reset_index(drop=True)
        ngram_accuracy_df['ngram size'] = ngram_accuracy_df['ngram_range'].apply(lambda x: x[0])

        sns.lineplot(data=ngram_accuracy_df, x='ngram size', y='Accuracy', hue='Features', style='Classifier', ax=ax1, markers=['o', 's'])
        
        ax1.set_title('Classifier accuracy vs single ngram size')
        ax1.set_xticks(ngram_accuracy_df['ngram size'])
        
        # Make heatmaps of the num_features_array and accuracy arrays using custom matrix_heatmap function
        fig2 = matrix_heatmap(num_features_array.tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'bone_r', 'custom_range': False, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Number of extracted features', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

        fig3 = matrix_heatmap(accuracy_arrays['Multinomial']['TF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Multinomial accuracy array heatmap:\n  TF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

        fig4 = matrix_heatmap(accuracy_arrays['Multinomial']['TF-IDF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Multinomial accuracy array heatmap:\n  TF-IDF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

        fig5 = matrix_heatmap(accuracy_arrays['Multinomial']['TF + TF-IDF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Multinomial accuracy array heatmap:\n  TF + TF-IDF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

        fig6 = matrix_heatmap(accuracy_arrays['Complement']['TF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Complement accuracy array heatmap:\n  TF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

        fig7 = matrix_heatmap(accuracy_arrays['Complement']['TF-IDF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Complement accuracy array heatmap:\n  TF-IDF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})

        fig8 = matrix_heatmap(accuracy_arrays['Complement']['TF + TF-IDF'].tolist(), options={'x_labels': [str(i) for i in range(1,5)], 'y_labels': [str(i) for i in range(1,5)][::-1], 'annotation_format': '.4g', 'color_map': 'inferno', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0.5, 'title_axis_labels': ('Complement accuracy array heatmap:\n  TF + TF-IDF feature vectors', 'ngram_min', 'ngram_max'), 'rotate x_tick_labels': False})
       
        st.header('Accuracies vs. `ngram_range` results:')
        st.write('**Accuracy vs** `ngram_range` **dataframe:**')
        st.write(accuracy_df)
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.pyplot(fig1)
        with output_col2:
            st.pyplot(fig2)

        st.subheader('**Heat maps of accuracies: Multinomial Classifier**')
        output_col1, output_col2, output_col3 = st.beta_columns(3)
        with output_col1:
            st.pyplot(fig3)
        with output_col2:
            st.pyplot(fig4)
        with output_col3:
            st.pyplot(fig5)
        st.subheader('**Heat maps of accuracies: Complement Classifier**')
        output_col1, output_col2, output_col3 = st.beta_columns(3)
        with output_col1:
            st.pyplot(fig6)
        with output_col2:
            st.pyplot(fig7)
        with output_col3:
            st.pyplot(fig8)       
    st.subheader('')
        
    st.write(
        '''
        From the accuracy dataframe, we generally do not see an increase in accuracy when `ngram_range` allows for more features to be extracted. In fact, we actually see a _decrease_ in classification performance, as shown in the first plot. It shows essentially the diagonal components of the six heatmaps below. It shows a stark decrease in accuracy as the number of terms in the ngram features increases from 1 to 4. The data in the plot correspond to just the values of `ngram_range` where the minimum and maximum sizes are equal, and is therefore a one-dimensional visualization. The heatmap next to it shows the number of features extracted depending on the value of `ngram_range`. The default value `ngram_range=(1,1)` is the lower-left cell with about 100k features, and is the smallest number in the plot. The largest number of features corresponds to `ngram_range=(1,4)` which allows for features with one, two, three, and four consecutive terms.

        The six heatmaps below the top two plots show two-dimensional views on the effect of changing `ngram_range` on classifier accuracy. The cell in the bottom-left corner corresponds to unigram features (`ngram_range=(1,1)`). Moving upward along the vertical axis increases the maximum ngram size. Moving horizontally increases the minimum ngram size. Of course, there is a bult in limitation in that the minimum size cannot be larger than the maximum size, so those cells are blank.

        A key observation in all of these heatmaps is the relative lack of dependence of accuracy on the value of the upper limit of `ngram_range`: in each heatmap, values in the same column are very similar. However, as the lower limit is increased, the accuracy steeply declines. This leads to the conclusion that smaller ngrams are more key to classification, so much so that one could either stick with the default value of `ngram_range=(1,1)` without losing much on performance, but gaining significantly on computation speed due to the reduction in the size of the feature space, or, for optimal performance, one can also include bi-grams, by setting `ngram_range=(1,2)`.
        '''
    )

    st.subheader('Limiting the feature space: tuning `min_df`, `max_df`, `max_features`, and `stop_words`')
    st.write(
        '''
        The next set of parameters to tune are ones that set limits on the learned vocabularies. These are `min_df`, `max_df`, and `max_features`. Recall that `min_df` (`max_df`) set a lower (upper) limit on the document frequencies of features included in the vocabulary, and `max_features=N` results in a 'Top-N' list of features when ranked by overall document frequency. 
        
        The arguments `min_df` and `max_df` take either integers, for document-frequency counts, or floats, for percentage of documents containing the terms. Their values are constrained to that `min_df` cannot represent more documents in the training set than `max_df`. If the inputs are floats, the number of documents is calculated by percent value times length of training corpus. The argument `max_features` takes an integer, and since our dataset has hundreds of thousands of features, we should space out the points to try accordingly.

        We want to choose values of `min_df` and `max_df` that slightly restrict the vocabulary, so we can see their effect better. We will use integer values of `min_df` to gain access to the smallest values different from the default. For `max_df`, we will use floats to restrict the document frequency down from 100%.

        The code block below loops over values of `min_df` and `max_df`, for `stop_words='english` and the default `None`.
        '''
    )

    # ---------------------------------------------------------------------
    # ----- Newsgroup performance: tuning min_df/max_df vs stop_words -----
    # ---------------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from scipy.sparse import hstack

                # Define values of min_df and max_df to loop over
                min_dfs = [x for x in range(1, 11, 1)]
                #max_dfs = [round(0.01+0.005*x, 3) for x in range(0, 29, 1)]
                max_dfs = [round(0.01*x, 3) for x in range(1, 16, 1)]+[round(0.2*x,2) for x in range(1,5)]

                # Create the pairs of min_df and max_df
                min_max_dfs = list(zip(min_dfs + [1]*len(max_dfs), [1.0]*len(min_dfs) + max_dfs))

                # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
                accuracy_df = pd.DataFrame(columns=['min_df', 'max_df', 'stop_words', 'num_features', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

                # Load and split the data
                newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)
                
                # Loop over min_df and max_df values
                for min_max_df in min_max_dfs:
                    
                    # Extract features with stop_words=None
                    extract_options = {'ngram_range': (1, 1), 'min_df': min_max_df[0], 'max_df': min_max_df[1], 'max_features': None, 'stop_words': None}
                    train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
                
                    # Horizontally stack TF and TF-IDF vectors
                    train_counts_tfidfs = hstack([train_counts, train_tfidfs])
                    test_counts_tfidfs = hstack([test_counts, test_tfidfs])

                    # Train and score classifiers
                    accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

                    # Append rows to accuracy dataframe
                    accuracy_df = accuracy_df.append({'min_df': min_max_df[0], 
                                                      'max_df': min_max_df[1], 
                                                      'stop_words': 'None', 
                                                      'num_features': train_counts.shape[1], 
                                                      'Classifier': 'Multinomial', 
                                                      'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                                      'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                                      'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                                      ignore_index=True)
                    accuracy_df = accuracy_df.append({'min_df': min_max_df[0], 
                                                      'max_df': min_max_df[1], 
                                                      'stop_words': 'None', 
                                                      'num_features': train_counts.shape[1], 
                                                      'Classifier': 'Complement', 
                                                      'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                                      'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                                      'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                                      ignore_index=True)
                    
                    # Extract features with stop_words='english'
                    extract_options = {'ngram_range': (1, 1), 'min_df': min_max_df[0], 'max_df': min_max_df[1], 'max_features': None, 'stop_words': 'english'}
                    train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
                
                    # Horizontally stack TF and TF-IDF vectors
                    train_counts_tfidfs = hstack([train_counts, train_tfidfs])
                    test_counts_tfidfs = hstack([test_counts, test_tfidfs])

                    # Train and score classifiers
                    accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

                    # Append row to accuracy dataframe
                    accuracy_df = accuracy_df.append({'min_df': min_max_df[0], 
                                                      'max_df': min_max_df[1], 
                                                      'stop_words': 'english', 
                                                      'num_features': train_counts.shape[1], 
                                                      'Classifier': 'Multinomial', 
                                                      'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                                      'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                                      'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                                      ignore_index=True)
                    accuracy_df = accuracy_df.append({'min_df': min_max_df[0], 
                                                      'max_df': min_max_df[1], 
                                                      'stop_words': 'english', 
                                                      'num_features': train_counts.shape[1], 
                                                      'Classifier': 'Complement', 
                                                      'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                                      'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                                      'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                                      ignore_index=True)
                
                # Melt accuracy dataframe for ease of plotting
                accuracy_melt_df = pd.melt(accuracy_df, id_vars=['min_df', 'max_df', 'stop_words', 'num_features', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')
                
                # Initiate plots
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()
                fig3, ax3 = plt.subplots()
                fig4, ax4 = plt.subplots()

                # Make line plots of accuracies vs min_df and max_df
                default_min_df = accuracy_melt_df[accuracy_melt_df['min_df']==1]
                default_max_df = accuracy_melt_df[accuracy_melt_df['max_df']==1.0]
                sns.lineplot(data=default_max_df[default_max_df['Classifier']=='Multinomial'], x='min_df', y='Accuracy', hue='Features', style='stop_words', ax=ax1, err_style=None, markers=['o', 's'])
                sns.lineplot(data=default_min_df[default_min_df['Classifier']=='Multinomial'], x='max_df', y='Accuracy', hue='Features', style='stop_words', ax=ax2, err_style=None, markers=['o', 's'])
                sns.lineplot(data=default_max_df[default_max_df['Classifier']=='Complement'], x='min_df', y='Accuracy', hue='Features', style='stop_words', ax=ax3, err_style=None, markers=['o', 's'])
                sns.lineplot(data=default_min_df[default_min_df['Classifier']=='Complement'], x='max_df', y='Accuracy', hue='Features', style='stop_words', ax=ax4, err_style=None, markers=['o', 's'])
                
                # Fix x-ticks of min_df plot
                ax1.set_xticks([x for x in range(1, 11, 1)])
                ax3.set_xticks([x for x in range(1, 11, 1)])

                # Change scaling of max_df plot to log-linear
                ax2.set(xscale='log', yscale='linear')
                ax4.set(xscale='log', yscale='linear')

                # Set plot titles
                ax1.set_title('Multinomial classifier accuracy vs. min_df value')
                ax2.set_title('Multinomial classifier accuracy vs. max_df value')
                ax3.set_title('Complement classifier accuracy vs. min_df value')
                ax4.set_title('Complement classifier accuracy vs. max_df value')

                # Show results and plots
                print('Accuracy dataframe:')
                print(accuracy_df.style.format({'max_df': '{:.2f}'}))

                print('Accuracies vs `min_df` with default `max_df=1.0`:')
                print(default_max_df)
                fig1.show()
                fig3.show()

                print('Accuracies vs `max_df` with default `min_df=1`:')
                print(default_min_df)
                fig2.show()
                fig4.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_tuning_mindf_maxdf_run_button')
    st.subheader('Output:')
    #output_expander = st.beta_expander('Expand output')
    if run_button:
        '''
        def load_and_split_newsgroups(training_size):
            # Load all of the data
            newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def extract_features(training_data, testing_data, opts={'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0, 'max_features': None, 'stop_words': None}):
            # Instantiate feature extractors
            count_vectorizer = CountVectorizer(ngram_range=opts['ngram_range'], min_df=opts['min_df'], max_df=opts['max_df'], max_features=opts['max_features'], stop_words=opts['stop_words'])
            tfidf_vectorizer = TfidfVectorizer(ngram_range=opts['ngram_range'], min_df=opts['min_df'], max_df=opts['max_df'], max_features=opts['max_features'], stop_words=opts['stop_words'])

            # Fit the vectorizers by learning the vocabulary of the 
            # training set, then compute counts and TF-IDFs
            train_counts = count_vectorizer.fit_transform(training_data)
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
            test_counts = count_vectorizer.transform(testing_data)
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_counts, test_counts, train_tfidfs, test_tfidfs
        
        def train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the accuracy results
            accuracy_dict = {}
            accuracy_dict['Multinomial'] = {}
            accuracy_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial_counts = MultinomialNB(alpha=alpha_mnb)
            multinomial_tfidfs = MultinomialNB(alpha=alpha_mnb)
            multinomial_counts_tfidfs = MultinomialNB(alpha=alpha_mnb)
            complement_counts = ComplementNB(alpha=alpha_cnb)
            complement_tfidfs = ComplementNB(alpha=alpha_cnb)
            complement_counts_tfidfs = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training counts and TF-IDFs
            multinomial_counts.fit(train_counts, train_labels)
            multinomial_tfidfs.fit(train_tfidfs, train_labels)
            multinomial_counts_tfidfs.fit(train_counts_tfidfs, train_labels)
            complement_counts.fit(train_counts, train_labels)
            complement_tfidfs.fit(train_tfidfs, train_labels)
            complement_counts_tfidfs.fit(train_counts_tfidfs, train_labels)

            # Add the accuracies to the dictionary:
            accuracy_dict['Multinomial']['TF accuracy'] = multinomial_counts.score(test_counts, test_labels)
            accuracy_dict['Multinomial']['TF-IDF accuracy'] = multinomial_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Multinomial']['TF + TF-IDF accuracy'] = multinomial_counts_tfidfs.score(test_counts_tfidfs, test_labels)
            accuracy_dict['Complement']['TF accuracy'] = complement_counts.score(test_counts, test_labels)
            accuracy_dict['Complement']['TF-IDF accuracy'] = complement_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Complement']['TF + TF-IDF accuracy'] = complement_counts_tfidfs.score(test_counts_tfidfs, test_labels)

            # Return the dictionary of results:
            return accuracy_dict
        
        # Define values of min_df and max_df to loop over
        min_dfs = [x for x in range(1, 11, 1)]
        #max_dfs = [round(0.01+0.005*x, 3) for x in range(0, 29, 1)]
        max_dfs = [round(0.01*x, 3) for x in range(1, 16, 1)]+[round(0.2*x,2) for x in range(1,5)]

        # Create the pairs of min_df and max_df
        min_max_dfs = list(zip(min_dfs + [1]*len(max_dfs), [1.0]*len(min_dfs) + max_dfs))

        # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
        accuracy_df = pd.DataFrame(columns=['min_df', 'max_df', 'stop_words', 'num_features', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

        # Load and split the data
        newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)
        
        # Loop over min_df and max_df values
        for min_max_df in min_max_dfs:
            
            # Extract features with stop_words=None
            extract_options = {'ngram_range': (1, 1), 'min_df': min_max_df[0], 'max_df': min_max_df[1], 'max_features': None, 'stop_words': None}
            train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
        
            # Horizontally stack TF and TF-IDF vectors
            train_counts_tfidfs = hstack([train_counts, train_tfidfs])
            test_counts_tfidfs = hstack([test_counts, test_tfidfs])

            # Train and score classifiers
            accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

            # Append rows to accuracy dataframe
            accuracy_df = accuracy_df.append({'min_df': min_max_df[0], 
                                              'max_df': min_max_df[1], 
                                              'stop_words': 'None', 
                                              'num_features': train_counts.shape[1], 
                                              'Classifier': 'Multinomial', 
                                              'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
            accuracy_df = accuracy_df.append({'min_df': min_max_df[0], 
                                              'max_df': min_max_df[1], 
                                              'stop_words': 'None', 
                                              'num_features': train_counts.shape[1], 
                                              'Classifier': 'Complement', 
                                              'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
            
            # Extract features with stop_words='english'
            extract_options = {'ngram_range': (1, 1), 'min_df': min_max_df[0], 'max_df': min_max_df[1], 'max_features': None, 'stop_words': 'english'}
            train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
        
            # Horizontally stack TF and TF-IDF vectors
            train_counts_tfidfs = hstack([train_counts, train_tfidfs])
            test_counts_tfidfs = hstack([test_counts, test_tfidfs])

            # Train and score classifiers
            accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

            # Append row to accuracy dataframe
            accuracy_df = accuracy_df.append({'min_df': min_max_df[0], 
                                              'max_df': min_max_df[1], 
                                              'stop_words': 'english', 
                                              'num_features': train_counts.shape[1], 
                                              'Classifier': 'Multinomial', 
                                              'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
            accuracy_df = accuracy_df.append({'min_df': min_max_df[0], 
                                              'max_df': min_max_df[1], 
                                              'stop_words': 'english', 
                                              'num_features': train_counts.shape[1], 
                                              'Classifier': 'Complement', 
                                              'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
        '''        
        accuracy_df = pd.DataFrame(data={'min_df': {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3, 12: 4, 13: 4, 14: 4, 15: 4, 16: 5, 17: 5, 18: 5, 19: 5, 20: 6, 21: 6, 22: 6, 23: 6, 24: 7, 25: 7, 26: 7, 27: 7, 28: 8, 29: 8, 30: 8, 31: 8, 32: 9, 33: 9, 34: 9, 35: 9, 36: 10, 37: 10, 38: 10, 39: 10, 40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1, 50: 1, 51: 1, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 1, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 66: 1, 67: 1, 68: 1, 69: 1, 70: 1, 71: 1, 72: 1, 73: 1, 74: 1, 75: 1, 76: 1, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 1, 84: 1, 85: 1, 86: 1, 87: 1, 88: 1, 89: 1, 90: 1, 91: 1, 92: 1, 93: 1, 94: 1, 95: 1, 96: 1, 97: 1, 98: 1, 99: 1, 100: 1, 101: 1, 102: 1, 103: 1, 104: 1, 105: 1, 106: 1, 107: 1, 108: 1, 109: 1, 110: 1, 111: 1, 112: 1, 113: 1, 114: 1, 115: 1}, 'max_df': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0, 29: 1.0, 30: 1.0, 31: 1.0, 32: 1.0, 33: 1.0, 34: 1.0, 35: 1.0, 36: 1.0, 37: 1.0, 38: 1.0, 39: 1.0, 40: 0.01, 41: 0.01, 42: 0.01, 43: 0.01, 44: 0.02, 45: 0.02, 46: 0.02, 47: 0.02, 48: 0.03, 49: 0.03, 50: 0.03, 51: 0.03, 52: 0.04, 53: 0.04, 54: 0.04, 55: 0.04, 56: 0.05, 57: 0.05, 58: 0.05, 59: 0.05, 60: 0.06, 61: 0.06, 62: 0.06, 63: 0.06, 64: 0.07, 65: 0.07, 66: 0.07, 67: 0.07, 68: 0.08, 69: 0.08, 70: 0.08, 71: 0.08, 72: 0.09, 73: 0.09, 74: 0.09, 75: 0.09, 76: 0.1, 77: 0.1, 78: 0.1, 79: 0.1, 80: 0.11, 81: 0.11, 82: 0.11, 83: 0.11, 84: 0.12, 85: 0.12, 86: 0.12, 87: 0.12, 88: 0.13, 89: 0.13, 90: 0.13, 91: 0.13, 92: 0.14, 93: 0.14, 94: 0.14, 95: 0.14, 96: 0.15, 97: 0.15, 98: 0.15, 99: 0.15, 100: 0.2, 101: 0.2, 102: 0.2, 103: 0.2, 104: 0.4, 105: 0.4, 106: 0.4, 107: 0.4, 108: 0.6, 109: 0.6, 110: 0.6, 111: 0.6, 112: 0.8, 113: 0.8, 114: 0.8, 115: 0.8}, 'stop_words': {0: 'None', 1: 'None', 2: 'english', 3: 'english', 4: 'None', 5: 'None', 6: 'english', 7: 'english', 8: 'None', 9: 'None', 10: 'english', 11: 'english', 12: 'None', 13: 'None', 14: 'english', 15: 'english', 16: 'None', 17: 'None', 18: 'english', 19: 'english', 20: 'None', 21: 'None', 22: 'english', 23: 'english', 24: 'None', 25: 'None', 26: 'english', 27: 'english', 28: 'None', 29: 'None', 30: 'english', 31: 'english', 32: 'None', 33: 'None', 34: 'english', 35: 'english', 36: 'None', 37: 'None', 38: 'english', 39: 'english', 40: 'None', 41: 'None', 42: 'english', 43: 'english', 44: 'None', 45: 'None', 46: 'english', 47: 'english', 48: 'None', 49: 'None', 50: 'english', 51: 'english', 52: 'None', 53: 'None', 54: 'english', 55: 'english', 56: 'None', 57: 'None', 58: 'english', 59: 'english', 60: 'None', 61: 'None', 62: 'english', 63: 'english', 64: 'None', 65: 'None', 66: 'english', 67: 'english', 68: 'None', 69: 'None', 70: 'english', 71: 'english', 72: 'None', 73: 'None', 74: 'english', 75: 'english', 76: 'None', 77: 'None', 78: 'english', 79: 'english', 80: 'None', 81: 'None', 82: 'english', 83: 'english', 84: 'None', 85: 'None', 86: 'english', 87: 'english', 88: 'None', 89: 'None', 90: 'english', 91: 'english', 92: 'None', 93: 'None', 94: 'english', 95: 'english', 96: 'None', 97: 'None', 98: 'english', 99: 'english', 100: 'None', 101: 'None', 102: 'english', 103: 'english', 104: 'None', 105: 'None', 106: 'english', 107: 'english', 108: 'None', 109: 'None', 110: 'english', 111: 'english', 112: 'None', 113: 'None', 114: 'english', 115: 'english'}, 'num_features': {0: 109088, 1: 109088, 2: 108779, 3: 108779, 4: 42199, 5: 42199, 6: 41891, 7: 41891, 8: 29381, 9: 29381, 10: 29074, 11: 29074, 12: 23854, 13: 23854, 14: 23547, 15: 23547, 16: 20264, 17: 20264, 18: 19958, 19: 19958, 20: 17546, 21: 17546, 22: 17241, 23: 17241, 24: 15606, 25: 15606, 26: 15302, 27: 15302, 28: 14210, 29: 14210, 30: 13909, 31: 13909, 32: 12967, 33: 12967, 34: 12666, 35: 12666, 36: 12025, 37: 12025, 38: 11725, 39: 11725, 40: 107740, 41: 107740, 42: 107668, 43: 107668, 44: 108448, 45: 108448, 46: 108344, 47: 108344, 48: 108689, 49: 108689, 50: 108555, 51: 108555, 52: 108787, 53: 108787, 54: 108641, 55: 108641, 56: 108856, 57: 108856, 58: 108693, 59: 108693, 60: 108894, 61: 108894, 62: 108722, 63: 108722, 64: 108928, 65: 108928, 66: 108740, 67: 108740, 68: 108952, 69: 108952, 70: 108752, 71: 108752, 72: 108966, 73: 108966, 74: 108757, 75: 108757, 76: 108978, 77: 108978, 78: 108762, 79: 108762, 80: 108992, 81: 108992, 82: 108766, 83: 108766, 84: 108999, 85: 108999, 86: 108767, 87: 108767, 88: 109009, 89: 109009, 90: 108769, 91: 108769, 92: 109014, 93: 109014, 94: 108770, 95: 108770, 96: 109019, 97: 109019, 98: 108771, 99: 108771, 100: 109039, 101: 109039, 102: 108776, 103: 108776, 104: 109072, 105: 109072, 106: 108779, 107: 108779, 108: 109082, 109: 109082, 110: 108779, 111: 108779, 112: 109087, 113: 109087, 114: 108779, 115: 108779}, 'Classifier': {0: 'Multinomial', 1: 'Complement', 2: 'Multinomial', 3: 'Complement', 4: 'Multinomial', 5: 'Complement', 6: 'Multinomial', 7: 'Complement', 8: 'Multinomial', 9: 'Complement', 10: 'Multinomial', 11: 'Complement', 12: 'Multinomial', 13: 'Complement', 14: 'Multinomial', 15: 'Complement', 16: 'Multinomial', 17: 'Complement', 18: 'Multinomial', 19: 'Complement', 20: 'Multinomial', 21: 'Complement', 22: 'Multinomial', 23: 'Complement', 24: 'Multinomial', 25: 'Complement', 26: 'Multinomial', 27: 'Complement', 28: 'Multinomial', 29: 'Complement', 30: 'Multinomial', 31: 'Complement', 32: 'Multinomial', 33: 'Complement', 34: 'Multinomial', 35: 'Complement', 36: 'Multinomial', 37: 'Complement', 38: 'Multinomial', 39: 'Complement', 40: 'Multinomial', 41: 'Complement', 42: 'Multinomial', 43: 'Complement', 44: 'Multinomial', 45: 'Complement', 46: 'Multinomial', 47: 'Complement', 48: 'Multinomial', 49: 'Complement', 50: 'Multinomial', 51: 'Complement', 52: 'Multinomial', 53: 'Complement', 54: 'Multinomial', 55: 'Complement', 56: 'Multinomial', 57: 'Complement', 58: 'Multinomial', 59: 'Complement', 60: 'Multinomial', 61: 'Complement', 62: 'Multinomial', 63: 'Complement', 64: 'Multinomial', 65: 'Complement', 66: 'Multinomial', 67: 'Complement', 68: 'Multinomial', 69: 'Complement', 70: 'Multinomial', 71: 'Complement', 72: 'Multinomial', 73: 'Complement', 74: 'Multinomial', 75: 'Complement', 76: 'Multinomial', 77: 'Complement', 78: 'Multinomial', 79: 'Complement', 80: 'Multinomial', 81: 'Complement', 82: 'Multinomial', 83: 'Complement', 84: 'Multinomial', 85: 'Complement', 86: 'Multinomial', 87: 'Complement', 88: 'Multinomial', 89: 'Complement', 90: 'Multinomial', 91: 'Complement', 92: 'Multinomial', 93: 'Complement', 94: 'Multinomial', 95: 'Complement', 96: 'Multinomial', 97: 'Complement', 98: 'Multinomial', 99: 'Complement', 100: 'Multinomial', 101: 'Complement', 102: 'Multinomial', 103: 'Complement', 104: 'Multinomial', 105: 'Complement', 106: 'Multinomial', 107: 'Complement', 108: 'Multinomial', 109: 'Complement', 110: 'Multinomial', 111: 'Complement', 112: 'Multinomial', 113: 'Complement', 114: 'Multinomial', 115: 'Complement'}, 'TF accuracy': {0: 0.7064940577249575, 1: 0.7236842105263158, 2: 0.708616298811545, 3: 0.7245331069609507, 4: 0.6967317487266553, 5: 0.7143463497453311, 6: 0.7022495755517827, 7: 0.7124363327674024, 8: 0.6935483870967742, 9: 0.70776740237691, 10: 0.6969439728353141, 11: 0.7084040747028862, 12: 0.6893039049235993, 13: 0.7005517826825127, 14: 0.6926994906621392, 15: 0.7011884550084889, 16: 0.6833616298811545, 17: 0.6929117147707979, 18: 0.6869694397283531, 19: 0.6941850594227504, 20: 0.6784804753820034, 21: 0.6882427843803056, 22: 0.6842105263157895, 23: 0.6907894736842105, 24: 0.6793293718166383, 25: 0.6839983022071308, 26: 0.6816638370118846, 27: 0.6859083191850595, 28: 0.6772071307300509, 29: 0.6791171477079796, 30: 0.6772071307300509, 31: 0.6797538200339559, 32: 0.6725382003395586, 33: 0.6769949066213922, 34: 0.6719015280135824, 35: 0.6778438030560272, 36: 0.669779286926995, 37: 0.6731748726655348, 38: 0.6693548387096774, 39: 0.672962648556876, 40: 0.6960950764006791, 41: 0.6988539898132428, 42: 0.6975806451612904, 43: 0.6988539898132428, 44: 0.7035229202037352, 45: 0.7141341256366723, 46: 0.7050084889643463, 47: 0.7128607809847198, 48: 0.7088285229202037, 49: 0.7215619694397284, 50: 0.7107385398981324, 51: 0.7202886247877759, 52: 0.7117996604414262, 53: 0.7241086587436333, 54: 0.7124363327674024, 55: 0.7228353140916808, 56: 0.7117996604414262, 57: 0.7255942275042445, 58: 0.7111629881154499, 59: 0.724320882852292, 60: 0.7122241086587436, 61: 0.7247453310696095, 62: 0.7120118845500849, 63: 0.7234719864176571, 64: 0.7105263157894737, 65: 0.7247453310696095, 66: 0.7113752122241087, 67: 0.7241086587436333, 68: 0.7096774193548387, 69: 0.7238964346349746, 70: 0.7107385398981324, 71: 0.7234719864176571, 72: 0.7098896434634975, 73: 0.7247453310696095, 74: 0.7107385398981324, 75: 0.7236842105263158, 76: 0.7090407470288624, 77: 0.7249575551782682, 78: 0.7105263157894737, 79: 0.7241086587436333, 80: 0.708616298811545, 81: 0.7258064516129032, 82: 0.7101018675721562, 83: 0.724320882852292, 84: 0.7092529711375212, 85: 0.7258064516129032, 86: 0.7107385398981324, 87: 0.7251697792869269, 88: 0.708616298811545, 89: 0.7266553480475382, 90: 0.7107385398981324, 91: 0.7251697792869269, 92: 0.7090407470288624, 93: 0.7266553480475382, 94: 0.7109507640067911, 95: 0.7253820033955858, 96: 0.708616298811545, 97: 0.7262308998302207, 98: 0.7109507640067911, 99: 0.7253820033955858, 100: 0.70946519524618, 101: 0.7255942275042445, 102: 0.7101018675721562, 103: 0.7241086587436333, 104: 0.7079796264855688, 105: 0.7253820033955858, 106: 0.708616298811545, 107: 0.7245331069609507, 108: 0.7058573853989814, 109: 0.7238964346349746, 110: 0.708616298811545, 111: 0.7245331069609507, 112: 0.7067062818336163, 113: 0.7236842105263158, 114: 0.708616298811545, 115: 0.7245331069609507}, 'TF-IDF accuracy': {0: 0.7650679117147708, 1: 0.7644312393887945, 2: 0.767402376910017, 3: 0.766553480475382, 4: 0.7593378607809848, 5: 0.7570033955857386, 6: 0.7606112054329371, 7: 0.7570033955857386, 8: 0.754881154499151, 9: 0.7525466893039049, 10: 0.7523344651952462, 11: 0.7508488964346349, 12: 0.7508488964346349, 13: 0.7463921901528013, 14: 0.7474533106960951, 15: 0.7436332767402377, 16: 0.7423599320882852, 17: 0.7396010186757216, 18: 0.7412988115449916, 19: 0.7376910016977929, 20: 0.7359932088285229, 21: 0.7338709677419355, 22: 0.7359932088285229, 23: 0.732597623089983, 24: 0.7349320882852292, 25: 0.7289898132427843, 26: 0.7349320882852292, 27: 0.7277164685908319, 28: 0.7349320882852292, 29: 0.7253820033955858, 30: 0.7321731748726655, 31: 0.7245331069609507, 32: 0.7315365025466893, 33: 0.7215619694397284, 34: 0.7272920203735145, 35: 0.7200764006791172, 36: 0.7264431239388794, 37: 0.7154074702886248, 38: 0.7241086587436333, 39: 0.7141341256366723, 40: 0.7449066213921901, 41: 0.7444821731748726, 42: 0.7453310696095077, 43: 0.7444821731748726, 44: 0.7561544991511036, 45: 0.7536078098471987, 46: 0.7567911714770797, 47: 0.7540322580645161, 48: 0.7635823429541596, 49: 0.7610356536502547, 50: 0.7640067911714771, 51: 0.7620967741935484, 52: 0.7635823429541596, 53: 0.7631578947368421, 54: 0.7635823429541596, 55: 0.7644312393887945, 56: 0.7642190152801358, 57: 0.7646434634974533, 58: 0.7633701188455009, 59: 0.7642190152801358, 60: 0.7646434634974533, 61: 0.7654923599320883, 62: 0.7659168081494058, 63: 0.765704584040747, 64: 0.7652801358234296, 65: 0.7659168081494058, 66: 0.7659168081494058, 67: 0.765704584040747, 68: 0.7661290322580645, 69: 0.766553480475382, 70: 0.7669779286926995, 71: 0.7659168081494058, 72: 0.7659168081494058, 73: 0.7663412563667232, 74: 0.7669779286926995, 75: 0.766553480475382, 76: 0.7654923599320883, 77: 0.7663412563667232, 78: 0.7659168081494058, 79: 0.7661290322580645, 80: 0.765704584040747, 81: 0.7659168081494058, 82: 0.7661290322580645, 83: 0.7667657045840407, 84: 0.7652801358234296, 85: 0.7661290322580645, 86: 0.7661290322580645, 87: 0.766553480475382, 88: 0.7652801358234296, 89: 0.7663412563667232, 90: 0.7661290322580645, 91: 0.7667657045840407, 92: 0.7652801358234296, 93: 0.766553480475382, 94: 0.7661290322580645, 95: 0.766553480475382, 96: 0.7659168081494058, 97: 0.7669779286926995, 98: 0.7661290322580645, 99: 0.7663412563667232, 100: 0.7659168081494058, 101: 0.7661290322580645, 102: 0.7676146010186757, 103: 0.7663412563667232, 104: 0.7663412563667232, 105: 0.7654923599320883, 106: 0.767402376910017, 107: 0.766553480475382, 108: 0.765704584040747, 109: 0.7644312393887945, 110: 0.767402376910017, 111: 0.766553480475382, 112: 0.7652801358234296, 113: 0.764855687606112, 114: 0.767402376910017, 115: 0.766553480475382}, 'TF + TF-IDF accuracy': {0: 0.7098896434634975, 1: 0.7313242784380306, 2: 0.7166808149405772, 3: 0.7298387096774194, 4: 0.7071307300509337, 5: 0.7213497453310697, 6: 0.7090407470288624, 7: 0.7211375212224108, 8: 0.7033106960950763, 9: 0.7173174872665535, 10: 0.705220713073005, 11: 0.7173174872665535, 12: 0.6982173174872666, 13: 0.7115874363327674, 14: 0.7001273344651953, 15: 0.7120118845500849, 16: 0.6922750424448217, 17: 0.7043718166383701, 18: 0.6952461799660441, 19: 0.7058573853989814, 20: 0.6880305602716469, 21: 0.6994906621392191, 22: 0.6931239388794567, 23: 0.7011884550084889, 24: 0.6880305602716469, 25: 0.6977928692699491, 26: 0.6916383701188455, 27: 0.6975806451612904, 28: 0.6854838709677419, 29: 0.6914261460101867, 30: 0.6873938879456706, 31: 0.6922750424448217, 32: 0.6801782682512734, 33: 0.6878183361629882, 34: 0.6833616298811545, 35: 0.6890916808149405, 36: 0.6767826825127334, 37: 0.6854838709677419, 38: 0.6810271646859083, 39: 0.6842105263157895, 40: 0.7060696095076401, 41: 0.7092529711375212, 42: 0.7067062818336163, 43: 0.7092529711375212, 44: 0.7101018675721562, 45: 0.7221986417657046, 46: 0.7111629881154499, 47: 0.7215619694397284, 48: 0.7166808149405772, 49: 0.730899830220713, 50: 0.7173174872665535, 51: 0.7304753820033956, 52: 0.7173174872665535, 53: 0.730899830220713, 54: 0.7202886247877759, 55: 0.730899830220713, 56: 0.717741935483871, 57: 0.7315365025466893, 58: 0.7183786078098472, 59: 0.7311120543293718, 60: 0.717741935483871, 61: 0.7311120543293718, 62: 0.7179541595925297, 63: 0.730899830220713, 64: 0.7158319185059423, 65: 0.730899830220713, 66: 0.7158319185059423, 67: 0.7304753820033956, 68: 0.716044142614601, 69: 0.7319609507640068, 70: 0.7164685908319185, 71: 0.7298387096774194, 72: 0.7171052631578947, 73: 0.7315365025466893, 74: 0.7164685908319185, 75: 0.7296264855687606, 76: 0.7164685908319185, 77: 0.7315365025466893, 78: 0.7162563667232598, 79: 0.7302631578947368, 80: 0.7162563667232598, 81: 0.7323853989813243, 82: 0.7171052631578947, 83: 0.7302631578947368, 84: 0.716893039049236, 85: 0.7330220713073005, 86: 0.7175297113752123, 87: 0.7304753820033956, 88: 0.7171052631578947, 89: 0.7321731748726655, 90: 0.7175297113752123, 91: 0.7302631578947368, 92: 0.7171052631578947, 93: 0.7319609507640068, 94: 0.7164685908319185, 95: 0.7306876061120543, 96: 0.717741935483871, 97: 0.7323853989813243, 98: 0.7164685908319185, 99: 0.7311120543293718, 100: 0.716893039049236, 101: 0.7313242784380306, 102: 0.716893039049236, 103: 0.7302631578947368, 104: 0.7141341256366723, 105: 0.7317487266553481, 106: 0.7166808149405772, 107: 0.7298387096774194, 108: 0.7107385398981324, 109: 0.7311120543293718, 110: 0.7166808149405772, 111: 0.7298387096774194, 112: 0.7109507640067911, 113: 0.7311120543293718, 114: 0.7166808149405772, 115: 0.7298387096774194}})
               
        # Melt accuracy dataframe for ease of plotting
        accuracy_melt_df = pd.melt(accuracy_df, id_vars=['min_df', 'max_df', 'stop_words', 'num_features', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')
        
        # Initiate plots
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()

        # Make line plots of accuracies vs min_df and max_df
        default_min_df = accuracy_melt_df[accuracy_melt_df['min_df']==1]
        default_max_df = accuracy_melt_df[accuracy_melt_df['max_df']==1.0]
        sns.lineplot(data=default_max_df[default_max_df['Classifier']=='Multinomial'], x='min_df', y='Accuracy', hue='Features', style='stop_words', ax=ax1, err_style=None, markers=['o', 's'])
        sns.lineplot(data=default_min_df[default_min_df['Classifier']=='Multinomial'], x='max_df', y='Accuracy', hue='Features', style='stop_words', ax=ax2, err_style=None, markers=['o', 's'])
        sns.lineplot(data=default_max_df[default_max_df['Classifier']=='Complement'], x='min_df', y='Accuracy', hue='Features', style='stop_words', ax=ax3, err_style=None, markers=['o', 's'])
        sns.lineplot(data=default_min_df[default_min_df['Classifier']=='Complement'], x='max_df', y='Accuracy', hue='Features', style='stop_words', ax=ax4, err_style=None, markers=['o', 's'])
        
        # Fix x-ticks of min_df plot
        ax1.set_xticks([x for x in range(1, 11, 1)])
        ax3.set_xticks([x for x in range(1, 11, 1)])

        # Change scaling of max_df plot to log-linear
        ax2.set(xscale='log', yscale='linear')
        ax4.set(xscale='log', yscale='linear')

        # Set plot titles
        ax1.set_title('Multinomial classifier accuracy vs. min_df value')
        ax2.set_title('Multinomial classifier accuracy vs. max_df value')
        ax3.set_title('Complement classifier accuracy vs. min_df value')
        ax4.set_title('Complement classifier accuracy vs. max_df value')

        # Show results and plots
        st.subheader('**Accuracy dataframe:**')
        st.write(accuracy_df.style.format({'max_df': '{:.2f}'}))
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.subheader('**Accuracies vs** `min_df` **with default** `max_df=1.0`:')
            st.write(default_max_df)
            st.pyplot(fig1)
            st.pyplot(fig3)
        with output_col2:
            st.subheader('**Accuracies vs** `max_df` **with default** `min_df=1`:')
            st.write(default_min_df)
            st.pyplot(fig2)
            st.pyplot(fig4)
    st.subheader('')        

    st.write(
        '''
        The plots of the classifier accuracy vs the values of `min_df` and `max_df` allow for easy drawing of some conclusions. From the left-hand side plots, we immediately see that increasing `min_df` results in deterioration of classifier accuracies across all types of feature vectors and choice of `stop_words`, for both the Multinomial and Complement classifiers. This makes sense since the dataset consists of twenty different categories, and requiring _every_ extracted feature to appear in multiple samples (the effect of using `min_df` greater than `1`) greatly reduces the vocabulary. The accuracies for the two `stop_words` choices follow each other as `min_df` increases.

        In the right-hand side plots, we've used a log-linear scaling to better show the effects of small `max_df` values. As `max_df` increases, we see an increase, and then saturation in the classifier accuracy when `stop_words='english`'. Notably, the accuracies for `stop_words=None` decreases as `max_df` increases to `1.0`. This is because the common English words excluded by choosing `stop_words='english'` are reintroduced to the vocabulary.

        For the default values of `min_df=1` and `max_df=1.0`, using `stop_words='english'` typically achieves higher accuracy than the default choice of `stop_words=None`. Despite our efforts to tune `min_df` and `max_df` to achieve a better performance, simply using the default value results in essentially optimal performance already, and so we will retain that choice as we continue to tune the parameters.

        The last free parameter we have is the value of the `max_features` argument. The code below tests different choices of the parameter against the same two choices of `stop_words`.
        '''
    )

    # --------------------------------------------------------------------
    # ----- Newsgroup performance: tuning max_features vs stop_words -----
    # --------------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from scipy.sparse import hstack

                # Define values of max_features to loop over
                max_features_list = range(5000, 110001, 5000)

                # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
                accuracy_df = pd.DataFrame(columns=['max_features', 'stop_words', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

                # Load and split the data
                newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)
                
                # Loop over max_features values
                for max_features_value in max_features_list:
                    
                    # Extract features with stop_words=None
                    extract_options = {'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0, 'max_features': max_features_value, 'stop_words': None}
                    train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
                
                    # Horizontally stack TF and TF-IDF vectors
                    train_counts_tfidfs = hstack([train_counts, train_tfidfs])
                    test_counts_tfidfs = hstack([test_counts, test_tfidfs])

                    # Train and score classifiers
                    accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

                    # Append rows to accuracy dataframe
                    accuracy_df = accuracy_df.append({'max_features': max_features_value,
                                                    'stop_words': 'None', 
                                                    'Classifier': 'Multinomial', 
                                                    'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                                    'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                                    'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                                    ignore_index=True)
                    accuracy_df = accuracy_df.append({'max_features': max_features_value,
                                                    'stop_words': 'None', 
                                                    'Classifier': 'Complement', 
                                                    'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                                    'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                                    'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                                    ignore_index=True)
                    
                    # Extract features with stop_words='english'
                    extract_options = {'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0, 'max_features': max_features_value, 'stop_words': 'english'}
                    train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
                
                    # Horizontally stack TF and TF-IDF vectors
                    train_counts_tfidfs = hstack([train_counts, train_tfidfs])
                    test_counts_tfidfs = hstack([test_counts, test_tfidfs])

                    # Train and score classifiers
                    accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

                    # Append rows to accuracy dataframe
                    accuracy_df = accuracy_df.append({'max_features': max_features_value,
                                                    'stop_words': 'english', 
                                                    'Classifier': 'Multinomial', 
                                                    'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                                    'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                                    'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                                    ignore_index=True)
                    accuracy_df = accuracy_df.append({'max_features': max_features_value,
                                                    'stop_words': 'english', 
                                                    'Classifier': 'Complement', 
                                                    'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                                    'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                                    'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                                    ignore_index=True)
                
                # Melt accuracy dataframe for ease of plotting
                accuracy_melt_df = pd.melt(accuracy_df, id_vars=['max_features', 'stop_words', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')
                
                # Initiate plots
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()

                # Make line plots of accuracies vs min_df and max_df
                sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Multinomial'], x='max_features', y='Accuracy', hue='Features', style='stop_words', ax=ax1, markers=['o', 's'])
                sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Complement'], x='max_features', y='Accuracy', hue='Features', style='stop_words', ax=ax2, markers=['o', 's'])

                # Set plot titles
                ax1.set_title('Multinomial classifier accuracy vs. max_features value')
                ax2.set_title('Complement classifier accuracy vs. max_features value')

                # Show results and plots
                print('Accuracy dataframe:')
                print(accuracy_df)
                fig1.show()
                fig2.show()      
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_tuning_max_features_run_button')
    st.subheader('Output:')
    #output_expander = st.beta_expander('Expand output')
    if run_button:
        '''
        def load_and_split_newsgroups(training_size):
            # Load all of the data
            newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def extract_features(training_data, testing_data, opts={'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0, 'max_features': None, 'stop_words': None}):
            # Instantiate feature extractors
            count_vectorizer = CountVectorizer(ngram_range=opts['ngram_range'], min_df=opts['min_df'], max_df=opts['max_df'], max_features=opts['max_features'], stop_words=opts['stop_words'])
            tfidf_vectorizer = TfidfVectorizer(ngram_range=opts['ngram_range'], min_df=opts['min_df'], max_df=opts['max_df'], max_features=opts['max_features'], stop_words=opts['stop_words'])

            # Fit the vectorizers by learning the vocabulary of the 
            # training set, then compute counts and TF-IDFs
            train_counts = count_vectorizer.fit_transform(training_data)
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into counts and TF-IDFs
            test_counts = count_vectorizer.transform(testing_data)
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_counts, test_counts, train_tfidfs, test_tfidfs
        
        def train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the accuracy results
            accuracy_dict = {}
            accuracy_dict['Multinomial'] = {}
            accuracy_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial_counts = MultinomialNB(alpha=alpha_mnb)
            multinomial_tfidfs = MultinomialNB(alpha=alpha_mnb)
            multinomial_counts_tfidfs = MultinomialNB(alpha=alpha_mnb)
            complement_counts = ComplementNB(alpha=alpha_cnb)
            complement_tfidfs = ComplementNB(alpha=alpha_cnb)
            complement_counts_tfidfs = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training counts and TF-IDFs
            multinomial_counts.fit(train_counts, train_labels)
            multinomial_tfidfs.fit(train_tfidfs, train_labels)
            multinomial_counts_tfidfs.fit(train_counts_tfidfs, train_labels)
            complement_counts.fit(train_counts, train_labels)
            complement_tfidfs.fit(train_tfidfs, train_labels)
            complement_counts_tfidfs.fit(train_counts_tfidfs, train_labels)

            # Add the accuracies to the dictionary:
            accuracy_dict['Multinomial']['TF accuracy'] = multinomial_counts.score(test_counts, test_labels)
            accuracy_dict['Multinomial']['TF-IDF accuracy'] = multinomial_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Multinomial']['TF + TF-IDF accuracy'] = multinomial_counts_tfidfs.score(test_counts_tfidfs, test_labels)
            accuracy_dict['Complement']['TF accuracy'] = complement_counts.score(test_counts, test_labels)
            accuracy_dict['Complement']['TF-IDF accuracy'] = complement_tfidfs.score(test_tfidfs, test_labels)
            accuracy_dict['Complement']['TF + TF-IDF accuracy'] = complement_counts_tfidfs.score(test_counts_tfidfs, test_labels)

            # Return the dictionary of results:
            return accuracy_dict
        
        # Define values of max_features to loop over
        max_features_list = range(5000, 110001, 5000)

        # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
        accuracy_df = pd.DataFrame(columns=['max_features', 'stop_words', 'Classifier', 'TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'])

        # Load and split the data
        newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)
        
        # Loop over max_features values
        for max_features_value in max_features_list:
            
            # Extract features with stop_words=None
            extract_options = {'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0, 'max_features': max_features_value, 'stop_words': None}
            train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
        
            # Horizontally stack TF and TF-IDF vectors
            train_counts_tfidfs = hstack([train_counts, train_tfidfs])
            test_counts_tfidfs = hstack([test_counts, test_tfidfs])

            # Train and score classifiers
            accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

            # Append rows to accuracy dataframe
            accuracy_df = accuracy_df.append({'max_features': max_features_value,
                                              'stop_words': 'None', 
                                              'Classifier': 'Multinomial', 
                                              'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
            accuracy_df = accuracy_df.append({'max_features': max_features_value,
                                              'stop_words': 'None', 
                                              'Classifier': 'Complement', 
                                              'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
            
            # Extract features with stop_words='english'
            extract_options = {'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0, 'max_features': max_features_value, 'stop_words': 'english'}
            train_counts, test_counts, train_tfidfs, test_tfidfs = extract_features(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
        
            # Horizontally stack TF and TF-IDF vectors
            train_counts_tfidfs = hstack([train_counts, train_tfidfs])
            test_counts_tfidfs = hstack([test_counts, test_tfidfs])

            # Train and score classifiers
            accuracies = train_and_score_classifiers(train_counts, test_counts, train_tfidfs, test_tfidfs, train_counts_tfidfs, test_counts_tfidfs, newsgroups_train.category, newsgroups_test.category)

            # Append rows to accuracy dataframe
            accuracy_df = accuracy_df.append({'max_features': max_features_value,
                                              'stop_words': 'english', 
                                              'Classifier': 'Multinomial', 
                                              'TF accuracy': accuracies['Multinomial']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Multinomial']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Multinomial']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
            accuracy_df = accuracy_df.append({'max_features': max_features_value,
                                              'stop_words': 'english', 
                                              'Classifier': 'Complement', 
                                              'TF accuracy': accuracies['Complement']['TF accuracy'], 
                                              'TF-IDF accuracy': accuracies['Complement']['TF-IDF accuracy'], 
                                              'TF + TF-IDF accuracy': accuracies['Complement']['TF + TF-IDF accuracy']},
                                            ignore_index=True)
        '''
        accuracy_df = pd.DataFrame(data={'max_features': {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 10000, 5: 10000, 6: 10000, 7: 10000, 8: 15000, 9: 15000, 10: 15000, 11: 15000, 12: 20000, 13: 20000, 14: 20000, 15: 20000, 16: 25000, 17: 25000, 18: 25000, 19: 25000, 20: 30000, 21: 30000, 22: 30000, 23: 30000, 24: 35000, 25: 35000, 26: 35000, 27: 35000, 28: 40000, 29: 40000, 30: 40000, 31: 40000, 32: 45000, 33: 45000, 34: 45000, 35: 45000, 36: 50000, 37: 50000, 38: 50000, 39: 50000, 40: 55000, 41: 55000, 42: 55000, 43: 55000, 44: 60000, 45: 60000, 46: 60000, 47: 60000, 48: 65000, 49: 65000, 50: 65000, 51: 65000, 52: 70000, 53: 70000, 54: 70000, 55: 70000, 56: 75000, 57: 75000, 58: 75000, 59: 75000, 60: 80000, 61: 80000, 62: 80000, 63: 80000, 64: 85000, 65: 85000, 66: 85000, 67: 85000, 68: 90000, 69: 90000, 70: 90000, 71: 90000, 72: 95000, 73: 95000, 74: 95000, 75: 95000, 76: 100000, 77: 100000, 78: 100000, 79: 100000, 80: 105000, 81: 105000, 82: 105000, 83: 105000, 84: 110000, 85: 110000, 86: 110000, 87: 110000}, 'stop_words': {0: 'None', 1: 'None', 2: 'english', 3: 'english', 4: 'None', 5: 'None', 6: 'english', 7: 'english', 8: 'None', 9: 'None', 10: 'english', 11: 'english', 12: 'None', 13: 'None', 14: 'english', 15: 'english', 16: 'None', 17: 'None', 18: 'english', 19: 'english', 20: 'None', 21: 'None', 22: 'english', 23: 'english', 24: 'None', 25: 'None', 26: 'english', 27: 'english', 28: 'None', 29: 'None', 30: 'english', 31: 'english', 32: 'None', 33: 'None', 34: 'english', 35: 'english', 36: 'None', 37: 'None', 38: 'english', 39: 'english', 40: 'None', 41: 'None', 42: 'english', 43: 'english', 44: 'None', 45: 'None', 46: 'english', 47: 'english', 48: 'None', 49: 'None', 50: 'english', 51: 'english', 52: 'None', 53: 'None', 54: 'english', 55: 'english', 56: 'None', 57: 'None', 58: 'english', 59: 'english', 60: 'None', 61: 'None', 62: 'english', 63: 'english', 64: 'None', 65: 'None', 66: 'english', 67: 'english', 68: 'None', 69: 'None', 70: 'english', 71: 'english', 72: 'None', 73: 'None', 74: 'english', 75: 'english', 76: 'None', 77: 'None', 78: 'english', 79: 'english', 80: 'None', 81: 'None', 82: 'english', 83: 'english', 84: 'None', 85: 'None', 86: 'english', 87: 'english'}, 'Classifier': {0: 'Multinomial', 1: 'Complement', 2: 'Multinomial', 3: 'Complement', 4: 'Multinomial', 5: 'Complement', 6: 'Multinomial', 7: 'Complement', 8: 'Multinomial', 9: 'Complement', 10: 'Multinomial', 11: 'Complement', 12: 'Multinomial', 13: 'Complement', 14: 'Multinomial', 15: 'Complement', 16: 'Multinomial', 17: 'Complement', 18: 'Multinomial', 19: 'Complement', 20: 'Multinomial', 21: 'Complement', 22: 'Multinomial', 23: 'Complement', 24: 'Multinomial', 25: 'Complement', 26: 'Multinomial', 27: 'Complement', 28: 'Multinomial', 29: 'Complement', 30: 'Multinomial', 31: 'Complement', 32: 'Multinomial', 33: 'Complement', 34: 'Multinomial', 35: 'Complement', 36: 'Multinomial', 37: 'Complement', 38: 'Multinomial', 39: 'Complement', 40: 'Multinomial', 41: 'Complement', 42: 'Multinomial', 43: 'Complement', 44: 'Multinomial', 45: 'Complement', 46: 'Multinomial', 47: 'Complement', 48: 'Multinomial', 49: 'Complement', 50: 'Multinomial', 51: 'Complement', 52: 'Multinomial', 53: 'Complement', 54: 'Multinomial', 55: 'Complement', 56: 'Multinomial', 57: 'Complement', 58: 'Multinomial', 59: 'Complement', 60: 'Multinomial', 61: 'Complement', 62: 'Multinomial', 63: 'Complement', 64: 'Multinomial', 65: 'Complement', 66: 'Multinomial', 67: 'Complement', 68: 'Multinomial', 69: 'Complement', 70: 'Multinomial', 71: 'Complement', 72: 'Multinomial', 73: 'Complement', 74: 'Multinomial', 75: 'Complement', 76: 'Multinomial', 77: 'Complement', 78: 'Multinomial', 79: 'Complement', 80: 'Multinomial', 81: 'Complement', 82: 'Multinomial', 83: 'Complement', 84: 'Multinomial', 85: 'Complement', 86: 'Multinomial', 87: 'Complement'}, 'TF accuracy': {0: 0.6279711375212224, 1: 0.6073853989813243, 2: 0.634125636672326, 3: 0.6186332767402377, 4: 0.6623514431239389, 5: 0.6574702886247877, 6: 0.6625636672325976, 7: 0.6591680814940577, 8: 0.6784804753820034, 9: 0.6795415959252971, 10: 0.6806027164685908, 11: 0.6814516129032258, 12: 0.6833616298811545, 13: 0.6929117147707979, 14: 0.6886672325976231, 15: 0.6960950764006791, 16: 0.6888794567062818, 17: 0.7035229202037352, 18: 0.6941850594227504, 19: 0.7037351443123939, 20: 0.693760611205433, 21: 0.7101018675721562, 22: 0.6960950764006791, 23: 0.7103140916808149, 24: 0.6946095076400679, 25: 0.7115874363327674, 26: 0.698641765704584, 27: 0.7117996604414262, 28: 0.6967317487266553, 29: 0.7145585738539898, 30: 0.7011884550084889, 31: 0.7130730050933786, 32: 0.6980050933786078, 33: 0.7173174872665535, 34: 0.7039473684210527, 35: 0.7164685908319185, 36: 0.6992784380305602, 37: 0.7188030560271647, 38: 0.705220713073005, 39: 0.7166808149405772, 40: 0.7007640067911715, 41: 0.7200764006791172, 42: 0.7054329371816639, 43: 0.7175297113752123, 44: 0.7005517826825127, 45: 0.719439728353141, 46: 0.7064940577249575, 47: 0.7183786078098472, 48: 0.7009762308998302, 49: 0.7200764006791172, 50: 0.706918505942275, 51: 0.7188030560271647, 52: 0.7014006791171478, 53: 0.7207130730050934, 54: 0.7084040747028862, 55: 0.7190152801358234, 56: 0.7018251273344652, 57: 0.7207130730050934, 58: 0.7081918505942275, 59: 0.7202886247877759, 60: 0.7039473684210527, 61: 0.7221986417657046, 62: 0.7090407470288624, 63: 0.7213497453310697, 64: 0.7047962648556876, 65: 0.7228353140916808, 66: 0.7088285229202037, 67: 0.7221986417657046, 68: 0.7045840407470289, 69: 0.722623089983022, 70: 0.7096774193548387, 71: 0.7234719864176571, 72: 0.705220713073005, 73: 0.722623089983022, 74: 0.7090407470288624, 75: 0.7241086587436333, 76: 0.7054329371816639, 77: 0.7219864176570459, 78: 0.7090407470288624, 79: 0.7245331069609507, 80: 0.7058573853989814, 81: 0.7238964346349746, 82: 0.70946519524618, 83: 0.724320882852292, 84: 0.7064940577249575, 85: 0.7236842105263158, 86: 0.708616298811545, 87: 0.7245331069609507}, 'TF-IDF accuracy': {0: 0.6793293718166383, 1: 0.6716893039049237, 2: 0.6871816638370118, 3: 0.6723259762308998, 4: 0.7221986417657046, 5: 0.7101018675721562, 6: 0.7205008488964346, 7: 0.7088285229202037, 8: 0.7381154499151104, 9: 0.7311120543293718, 10: 0.7355687606112055, 11: 0.7311120543293718, 12: 0.7449066213921901, 13: 0.7427843803056027, 14: 0.7423599320882852, 15: 0.7396010186757216, 16: 0.7521222410865874, 17: 0.7474533106960951, 18: 0.7504244482173175, 19: 0.745118845500849, 20: 0.7559422750424448, 21: 0.7531833616298812, 22: 0.7536078098471987, 23: 0.7506366723259762, 24: 0.7587011884550084, 25: 0.754881154499151, 26: 0.7572156196943973, 27: 0.7521222410865874, 28: 0.7608234295415959, 29: 0.7570033955857386, 30: 0.7587011884550084, 31: 0.7542444821731749, 32: 0.7614601018675722, 33: 0.7580645161290323, 34: 0.7614601018675722, 35: 0.7576400679117148, 36: 0.7618845500848896, 37: 0.759974533106961, 38: 0.7631578947368421, 39: 0.7589134125636672, 40: 0.7631578947368421, 41: 0.7606112054329371, 42: 0.7631578947368421, 43: 0.7608234295415959, 44: 0.7644312393887945, 45: 0.7610356536502547, 46: 0.7631578947368421, 47: 0.7610356536502547, 48: 0.764855687606112, 49: 0.7618845500848896, 50: 0.7627334465195246, 51: 0.7620967741935484, 52: 0.7661290322580645, 53: 0.7627334465195246, 54: 0.7631578947368421, 55: 0.7618845500848896, 56: 0.7654923599320883, 57: 0.7625212224108658, 58: 0.7650679117147708, 59: 0.7635823429541596, 60: 0.7669779286926995, 61: 0.7635823429541596, 62: 0.765704584040747, 63: 0.7640067911714771, 64: 0.765704584040747, 65: 0.7633701188455009, 66: 0.7654923599320883, 67: 0.7640067911714771, 68: 0.7650679117147708, 69: 0.7633701188455009, 70: 0.766553480475382, 71: 0.765704584040747, 72: 0.764855687606112, 73: 0.7637945670628183, 74: 0.766553480475382, 75: 0.7659168081494058, 76: 0.7646434634974533, 77: 0.7646434634974533, 78: 0.7669779286926995, 79: 0.7659168081494058, 80: 0.7646434634974533, 81: 0.7642190152801358, 82: 0.767402376910017, 83: 0.766553480475382, 84: 0.7650679117147708, 85: 0.7644312393887945, 86: 0.767402376910017, 87: 0.766553480475382}, 'TF + TF-IDF accuracy': {0: 0.634974533106961, 1: 0.6245755517826825, 2: 0.6419779286926995, 3: 0.6324278438030561, 4: 0.6691426146010186, 5: 0.672962648556876, 6: 0.6714770797962648, 7: 0.6727504244482173, 8: 0.6848471986417657, 9: 0.6943972835314092, 10: 0.6899405772495756, 11: 0.6935483870967742, 12: 0.692062818336163, 13: 0.7058573853989814, 14: 0.6980050933786078, 15: 0.7062818336162988, 16: 0.6988539898132428, 17: 0.7143463497453311, 18: 0.7022495755517827, 19: 0.7139219015280136, 20: 0.7022495755517827, 21: 0.7202886247877759, 22: 0.7050084889643463, 23: 0.7183786078098472, 24: 0.7043718166383701, 25: 0.7200764006791172, 26: 0.7067062818336163, 27: 0.719439728353141, 28: 0.7058573853989814, 29: 0.7221986417657046, 30: 0.7075551782682513, 31: 0.7200764006791172, 32: 0.7071307300509337, 33: 0.7234719864176571, 34: 0.7098896434634975, 35: 0.7241086587436333, 36: 0.70776740237691, 37: 0.7249575551782682, 38: 0.7120118845500849, 39: 0.7245331069609507, 40: 0.70776740237691, 41: 0.7262308998302207, 42: 0.7120118845500849, 43: 0.7253820033955858, 44: 0.7075551782682513, 45: 0.7262308998302207, 46: 0.7124363327674024, 47: 0.726018675721562, 48: 0.7075551782682513, 49: 0.7275042444821732, 50: 0.7130730050933786, 51: 0.726018675721562, 52: 0.7084040747028862, 53: 0.7279286926994907, 54: 0.7143463497453311, 55: 0.7264431239388794, 56: 0.70776740237691, 57: 0.7275042444821732, 58: 0.7134974533106961, 59: 0.7275042444821732, 60: 0.7088285229202037, 61: 0.7292020373514432, 62: 0.7141341256366723, 63: 0.7281409168081494, 64: 0.7098896434634975, 65: 0.7296264855687606, 66: 0.7141341256366723, 67: 0.7298387096774194, 68: 0.7092529711375212, 69: 0.7296264855687606, 70: 0.715195246179966, 71: 0.7304753820033956, 72: 0.7090407470288624, 73: 0.7298387096774194, 74: 0.7158319185059423, 75: 0.730899830220713, 76: 0.7092529711375212, 77: 0.7296264855687606, 78: 0.7156196943972836, 79: 0.730899830220713, 80: 0.70946519524618, 81: 0.7306876061120543, 82: 0.7162563667232598, 83: 0.7298387096774194, 84: 0.7098896434634975, 85: 0.7313242784380306, 86: 0.7166808149405772, 87: 0.7298387096774194}})
        
        # Melt accuracy dataframe for ease of plotting
        accuracy_melt_df = pd.melt(accuracy_df, id_vars=['max_features', 'stop_words', 'Classifier'], value_vars=['TF accuracy', 'TF-IDF accuracy', 'TF + TF-IDF accuracy'], var_name='Features', value_name='Accuracy')
        
        # Initiate plots
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        # Make line plots of accuracies vs min_df and max_df
        sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Multinomial'], x='max_features', y='Accuracy', hue='Features', style='stop_words', ax=ax1, markers=['o', 's'])
        sns.lineplot(data=accuracy_melt_df[accuracy_melt_df['Classifier']=='Complement'], x='max_features', y='Accuracy', hue='Features', style='stop_words', ax=ax2, markers=['o', 's'])

        # Set plot titles
        ax1.set_title('Multinomial classifier accuracy vs. max_features value')
        ax2.set_title('Complement classifier accuracy vs. max_features value')

        # Show results and plots
        st.subheader('**Accuracy dataframe:**')
        st.write(accuracy_df)
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.pyplot(fig1)
        with output_col2:
            st.pyplot(fig2)        
    st.subheader('')

    st.write(
        '''
        Looking at the plots of classifier accuracy vs. `max_features`, we see as `max_features` increases, the accuracies of both classifier types increase before becoming saturated around a value of `max_features=50000`. For the Multinomial classifier, we also see that using `stop_words='english'` is a better choice in most cases, and notably for the right-most points. For the Complement classifier, the choice of `stop_words` has marginal effect. Since the right-most point has a `max_feature` value larger than the number of extracted features, it is equivalent to using the default setting of `max_features=None`.

        Having verified that the default choices for `ngram_range`, `min_df`, `max_df`, and `max_features` are already optimal for the most part, along with performance boosts seen when using `stop_words='english'`, we will now focus on the additional options available using `TfidfVectorizer` to see if we can squeeze out any additional performance. Moving forward, we will abandon term frequency and TF + TF-IDF feature vector sets in favor of TF-IDF vectors.
        '''
    )

    st.subheader('Further tuning `TfidfVectorizer` arguments')
    st.write(
        '''
        Recall the list of options we have access to for `TfidfVectorizer`:
        **Options unique to `TfidfVectorizer`:**
         - `norm`: option to choose either `'l1'` or `'l2'` (default) normalization, or `None` for no normalization
         - `use_idf`: boolean to toggle inverse document frequency weighting (default is `True`)
         - `smooth_idf`: boolean to toggle smoothing of inverse document frequency value (default is `True`)
         - `sublinear_tf`: boolean to apply logarithmic term frequency scaling (default is `False`)

        Since they are all discrete variables, with three being boolean and one having only three choices, we can simply run through all of them and make a table showing the accuracies. In all there are 24 combinations to test. To test these choices, let's take our `extract_features` convenience function and modify it as
        ```python
        def extract_tfidfs(training_data, testing_data, opts={'norm': 'l2', 'use_idf': True, 'smooth_idf': True, 'sublinear_tf': False}):
            # Instantiate feature extractors
            tfidf_vectorizer = TfidfVectorizer(norm=opts['norm'], use_idf=opts['use_idf'], smooth_idf=opts['smooth_idf'], sublinear_tf=opts['sublinear_tf'], stop_words='english')

            # Fit the vectorizers by learning the vocabulary of the training set, then compute TF-IDFs
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into TF-IDFs
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_tfidfs, test_tfidfs
        ```
        Note that we are using defaults for `ngram_range`, `min_df`, `max_df`, and `max_features`. For `stop_words`, we use `'english'` even though it is not the default choice. Our custom `train_and_score_classifiers` function is simplified to only handle TF-IDF vectors:
        ```python
        def train_and_score_classifiers(train_features, test_features, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the accuracy results
            accuracy_dict = {}
            accuracy_dict['Multinomial'] = {}
            accuracy_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial = MultinomialNB(alpha=alpha_mnb)
            complement = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training features
            multinomial.fit(train_features, train_labels)
            complement.fit(train_features, train_labels)

            # Add the accuracies to the dictionary:
            accuracy_dict['Multinomial']['Accuracy'] = multinomial.score(test_features, test_labels)
            accuracy_dict['Complement']['Accuracy'] = complement.score(test_features, test_labels)

            # Return the dictionary of results:
            return accuracy_dict
        ```

        Run the code below to test out options unique to `TfidfVectorizer`.
        '''
    )

    # -----------------------------------------------------------------
    # ----- Newsgroup performance: tuning TfidfVectorizer options -----
    # -----------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB

                # Define values of max_features to loop over
                norm_list = ['l1', 'l2', None]
                use_idf_list = [True, False]
                smooth_idf_list = [True, False]
                sublinear_tf_list = [True, False]

                # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
                accuracy_df = pd.DataFrame(columns=['norm', 'use_idf', 'smooth_idf', 'sublinear_tf', 'Classifier', 'TF-IDF accuracy'])

                # Load and split the data
                newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)
                
                # Create nested loops to test all combinations
                for norm_choice in norm_list:
                    for use_idf_choice in use_idf_list:
                        for smooth_idf_choice in smooth_idf_list:
                            for sublinear_tf_choice in sublinear_tf_list:

                                # Extract features
                                extract_options = {'norm': norm_choice, 'use_idf': use_idf_choice, 'smooth_idf': smooth_idf_choice, 'sublinear_tf': sublinear_tf_choice}
                                train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
                
                                # Train and score classifiers
                                accuracies = train_and_score_classifiers(train_tfidfs, test_tfidfs, newsgroups_train.category, newsgroups_test.category)

                                # Append row to accuracy dataframe
                                accuracy_df = accuracy_df.append({'norm': str(norm_choice),
                                                                'use_idf': str(use_idf_choice),
                                                                'smooth_idf': str(smooth_idf_choice),
                                                                'sublinear_tf': str(sublinear_tf_choice),
                                                                'Classifier': 'Multinomial',
                                                                'TF-IDF accuracy': accuracies['Multinomial']['Accuracy']},
                                                                ignore_index=True)
                                accuracy_df = accuracy_df.append({'norm': str(norm_choice),
                                                                'use_idf': str(use_idf_choice),
                                                                'smooth_idf': str(smooth_idf_choice),
                                                                'sublinear_tf': str(sublinear_tf_choice),
                                                                'Classifier': 'Complement',
                                                                'TF-IDF accuracy': accuracies['Complement']['Accuracy']},
                                                                ignore_index=True)

                print('Accuracy dataframe:')
                print(accuracy_df[accuracy_df['Classifier']=='Multinomial'].sort_values(by='TF-IDF accuracy', ascending=False))
                print(accuracy_df[accuracy_df['Classifier']=='Complement'].sort_values(by='TF-IDF accuracy', ascending=False))
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroups_tuning_tfidfvectorizer_run_button')
    st.subheader('Output:')
    #output_expander = st.beta_expander('Expand output')
    if run_button:
        '''
        def load_and_split_newsgroups(training_size):
            # Load all of the data
            newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def extract_tfidfs(training_data, testing_data, opts={'norm': 'l2', 'use_idf': True, 'smooth_idf': True, 'sublinear_tf': False}):
            # Instantiate feature extractors
            tfidf_vectorizer = TfidfVectorizer(norm=opts['norm'], use_idf=opts['use_idf'], smooth_idf=opts['smooth_idf'], sublinear_tf=opts['sublinear_tf'], stop_words='english')

            # Fit the vectorizers by learning the vocabulary of the training set, then compute TF-IDFs
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into TF-IDFs
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_tfidfs, test_tfidfs
        
        def train_and_score_classifiers(train_features, test_features, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the accuracy results
            accuracy_dict = {}
            accuracy_dict['Multinomial'] = {}
            accuracy_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial = MultinomialNB(alpha=alpha_mnb)
            complement = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training features
            multinomial.fit(train_features, train_labels)
            complement.fit(train_features, train_labels)

            # Add the accuracies to the dictionary:
            accuracy_dict['Multinomial']['Accuracy'] = multinomial.score(test_features, test_labels)
            accuracy_dict['Complement']['Accuracy'] = complement.score(test_features, test_labels)

            # Return the dictionary of results:
            return accuracy_dict
        
        # Define values of max_features to loop over
        norm_list = ['l1', 'l2', None]
        use_idf_list = [True, False]
        smooth_idf_list = [True, False]
        sublinear_tf_list = [True, False]

        # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
        accuracy_df = pd.DataFrame(columns=['norm', 'use_idf', 'smooth_idf', 'sublinear_tf', 'Classifier', 'TF-IDF accuracy'])

        # Load and split the data
        newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)
        
        # Create nested loops to test all combinations
        for norm_choice in norm_list:
            for use_idf_choice in use_idf_list:
                for smooth_idf_choice in smooth_idf_list:
                    for sublinear_tf_choice in sublinear_tf_list:

                        # Extract features
                        extract_options = {'norm': norm_choice, 'use_idf': use_idf_choice, 'smooth_idf': smooth_idf_choice, 'sublinear_tf': sublinear_tf_choice}
                        train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings, opts=extract_options)
        
                        # Train and score classifiers
                        accuracies = train_and_score_classifiers(train_tfidfs, test_tfidfs, newsgroups_train.category, newsgroups_test.category)

                        # Append row to accuracy dataframe
                        accuracy_df = accuracy_df.append({'norm': str(norm_choice),
                                                          'use_idf': str(use_idf_choice),
                                                          'smooth_idf': str(smooth_idf_choice),
                                                          'sublinear_tf': str(sublinear_tf_choice),
                                                          'Classifier': 'Multinomial',
                                                          'TF-IDF accuracy': accuracies['Multinomial']['Accuracy']},
                                                          ignore_index=True)
                        accuracy_df = accuracy_df.append({'norm': str(norm_choice),
                                                          'use_idf': str(use_idf_choice),
                                                          'smooth_idf': str(smooth_idf_choice),
                                                          'sublinear_tf': str(sublinear_tf_choice),
                                                          'Classifier': 'Complement',
                                                          'TF-IDF accuracy': accuracies['Complement']['Accuracy']},
                                                          ignore_index=True)
        '''
        accuracy_df = pd.DataFrame(data={'norm': {0: 'l1', 1: 'l1', 2: 'l1', 3: 'l1', 4: 'l1', 5: 'l1', 6: 'l1', 7: 'l1', 8: 'l1', 9: 'l1', 10: 'l1', 11: 'l1', 12: 'l1', 13: 'l1', 14: 'l1', 15: 'l1', 16: 'l2', 17: 'l2', 18: 'l2', 19: 'l2', 20: 'l2', 21: 'l2', 22: 'l2', 23: 'l2', 24: 'l2', 25: 'l2', 26: 'l2', 27: 'l2', 28: 'l2', 29: 'l2', 30: 'l2', 31: 'l2', 32: 'None', 33: 'None', 34: 'None', 35: 'None', 36: 'None', 37: 'None', 38: 'None', 39: 'None', 40: 'None', 41: 'None', 42: 'None', 43: 'None', 44: 'None', 45: 'None', 46: 'None', 47: 'None'}, 'use_idf': {0: 'True', 1: 'True', 2: 'True', 3: 'True', 4: 'True', 5: 'True', 6: 'True', 7: 'True', 8: 'False', 9: 'False', 10: 'False', 11: 'False', 12: 'False', 13: 'False', 14: 'False', 15: 'False', 16: 'True', 17: 'True', 18: 'True', 19: 'True', 20: 'True', 21: 'True', 22: 'True', 23: 'True', 24: 'False', 25: 'False', 26: 'False', 27: 'False', 28: 'False', 29: 'False', 30: 'False', 31: 'False', 32: 'True', 33: 'True', 34: 'True', 35: 'True', 36: 'True', 37: 'True', 38: 'True', 39: 'True', 40: 'False', 41: 'False', 42: 'False', 43: 'False', 44: 'False', 45: 'False', 46: 'False', 47: 'False'}, 'smooth_idf': {0: 'True', 1: 'True', 2: 'True', 3: 'True', 4: 'False', 5: 'False', 6: 'False', 7: 'False', 8: 'True', 9: 'True', 10: 'True', 11: 'True', 12: 'False', 13: 'False', 14: 'False', 15: 'False', 16: 'True', 17: 'True', 18: 'True', 19: 'True', 20: 'False', 21: 'False', 22: 'False', 23: 'False', 24: 'True', 25: 'True', 26: 'True', 27: 'True', 28: 'False', 29: 'False', 30: 'False', 31: 'False', 32: 'True', 33: 'True', 34: 'True', 35: 'True', 36: 'False', 37: 'False', 38: 'False', 39: 'False', 40: 'True', 41: 'True', 42: 'True', 43: 'True', 44: 'False', 45: 'False', 46: 'False', 47: 'False'}, 'sublinear_tf': {0: 'True', 1: 'True', 2: 'False', 3: 'False', 4: 'True', 5: 'True', 6: 'False', 7: 'False', 8: 'True', 9: 'True', 10: 'False', 11: 'False', 12: 'True', 13: 'True', 14: 'False', 15: 'False', 16: 'True', 17: 'True', 18: 'False', 19: 'False', 20: 'True', 21: 'True', 22: 'False', 23: 'False', 24: 'True', 25: 'True', 26: 'False', 27: 'False', 28: 'True', 29: 'True', 30: 'False', 31: 'False', 32: 'True', 33: 'True', 34: 'False', 35: 'False', 36: 'True', 37: 'True', 38: 'False', 39: 'False', 40: 'True', 41: 'True', 42: 'False', 43: 'False', 44: 'True', 45: 'True', 46: 'False', 47: 'False'}, 'Classifier': {0: 'Multinomial', 1: 'Complement', 2: 'Multinomial', 3: 'Complement', 4: 'Multinomial', 5: 'Complement', 6: 'Multinomial', 7: 'Complement', 8: 'Multinomial', 9: 'Complement', 10: 'Multinomial', 11: 'Complement', 12: 'Multinomial', 13: 'Complement', 14: 'Multinomial', 15: 'Complement', 16: 'Multinomial', 17: 'Complement', 18: 'Multinomial', 19: 'Complement', 20: 'Multinomial', 21: 'Complement', 22: 'Multinomial', 23: 'Complement', 24: 'Multinomial', 25: 'Complement', 26: 'Multinomial', 27: 'Complement', 28: 'Multinomial', 29: 'Complement', 30: 'Multinomial', 31: 'Complement', 32: 'Multinomial', 33: 'Complement', 34: 'Multinomial', 35: 'Complement', 36: 'Multinomial', 37: 'Complement', 38: 'Multinomial', 39: 'Complement', 40: 'Multinomial', 41: 'Complement', 42: 'Multinomial', 43: 'Complement', 44: 'Multinomial', 45: 'Complement', 46: 'Multinomial', 47: 'Complement'}, 'TF-IDF accuracy': {0: 0.7338709677419355, 1: 0.7487266553480475, 2: 0.740025466893039, 3: 0.7493633276740238, 4: 0.7349320882852292, 5: 0.7497877758913413, 6: 0.741723259762309, 7: 0.75, 8: 0.7134974533106961, 9: 0.734295415959253, 10: 0.716893039049236, 11: 0.7338709677419355, 12: 0.7134974533106961, 13: 0.734295415959253, 14: 0.716893039049236, 15: 0.7338709677419355, 16: 0.7603989813242784, 17: 0.765704584040747, 18: 0.767402376910017, 19: 0.766553480475382, 20: 0.7612478777589134, 21: 0.767402376910017, 22: 0.7676146010186757, 23: 0.7669779286926995, 24: 0.7531833616298812, 25: 0.7618845500848896, 26: 0.7559422750424448, 27: 0.759125636672326, 28: 0.7531833616298812, 29: 0.7618845500848896, 30: 0.7559422750424448, 31: 0.759125636672326, 32: 0.706918505942275, 33: 0.6943972835314092, 34: 0.7090407470288624, 35: 0.6956706281833617, 36: 0.7079796264855688, 37: 0.6926994906621392, 38: 0.708616298811545, 39: 0.6933361629881154, 40: 0.7166808149405772, 41: 0.7389643463497453, 42: 0.708616298811545, 43: 0.7245331069609507, 44: 0.7166808149405772, 45: 0.7389643463497453, 46: 0.708616298811545, 47: 0.7245331069609507}})

        
        # Create custom style function
        def custom_style_old(row):
            color = 'white'
            if row['norm'] == 'l2' and row['use_idf'] == 'True' and row['smooth_idf'] == 'True' and row['sublinear_tf'] == 'False':
                color = '#ffb3ba'
            elif row['norm'] != 'l2' and row['use_idf'] == 'True' and row['smooth_idf'] == 'True' and row['sublinear_tf'] == 'False':
                color = '#ffdfba'
                return ['background-color: %s' % color, 'background-color: white', 'background-color: white', 'background-color: white', 'background-color: %s' % color]
            elif row['norm'] == 'l2' and row['use_idf'] != 'True' and row['smooth_idf'] == 'True' and row['sublinear_tf'] == 'False':
                color = '#baffc9'
                return ['background-color: white', 'background-color: %s' % color, 'background-color: white', 'background-color: white', 'background-color: %s' % color]
            elif row['norm'] == 'l2' and row['use_idf'] == 'True' and row['smooth_idf'] != 'True' and row['sublinear_tf'] == 'False':
                color = '#bae1ff'
                return ['background-color: white', 'background-color: white', 'background-color: %s' % color, 'background-color: white', 'background-color: %s' % color]
            elif row['norm'] == 'l2' and row['use_idf'] == 'True' and row['smooth_idf'] == 'True' and row['sublinear_tf'] != 'False':
                color = '#E0BBE4'
                return ['background-color: white', 'background-color: white', 'background-color: white', 'background-color: %s' % color, 'background-color: %s' % color]

            return ['background-color: %s' % color]*len(row.values)
        
        def custom_style(row):
            default_color = '#ffb3ba'
            styles = []
            if row['norm'] == 'l2':
                styles.append('background-color: %s' % default_color)
            else:
                styles.append('background-color: #ffdfba')
            
            if row['use_idf'] == 'True':
                styles.append('background-color: %s' % default_color)
            else:
                styles.append('background-color: #ffffba')
            
            if row['smooth_idf'] == 'True':
                styles.append('background-color: %s' % default_color)
            else:
                styles.append('background-color: #baffc9')
            
            if row['sublinear_tf'] == 'False':
                styles.append('background-color: %s' % default_color)
            else:
                styles.append('background-color: #bae1ff')
            
            styles.append('background-color: white')
            styles.append('background-color: white')
            return styles

        
        def color_guide_style(row):
            if row['Color'] == 'Red':
                color = '#ffb3ba'
            elif row['Color'] == 'Orange':
                color = '#ffdfba'
            elif row['Color'] == 'Yellow':
                color = '#ffffba'
            elif row['Color'] == 'Green':
                color = '#baffc9'
            elif row['Color'] == 'Blue':
                color = '#bae1ff'
            return ['background-color: %s' % color]*len(row.values)

        color_df = pd.DataFrame(data={'Property': ['Default', 'change norm', 'change use_idf', 'change smooth_idf', 'change sublinear_tf'], 'Color': ['Red', 'Orange', 'Yellow', 'Green', 'Blue']})
        
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.subheader('**Multinomial classifier accuracy dataframe:**')
            #st.dataframe(accuracy_df.style.apply(custom_style, axis=1).highlight_max(axis=0, subset='TF-IDF accuracy', color="#ffff99"), height=1000)
            st.dataframe(accuracy_df[accuracy_df['Classifier']=='Multinomial'].sort_values(by='TF-IDF accuracy', ascending=False).style.apply(custom_style, axis=1).background_gradient(cmap='viridis', axis=0, subset='TF-IDF accuracy'), height=1000)
        with output_col2:
            st.subheader('**Complement classifier accuracy dataframe:**')
            st.dataframe(accuracy_df[accuracy_df['Classifier']=='Complement'].sort_values(by='TF-IDF accuracy', ascending=False).style.apply(custom_style, axis=1).background_gradient(cmap='viridis', axis=0, subset='TF-IDF accuracy'), height=1000)
        st.subheader('**Color guide:**')
        st.write(color_df.style.apply(color_guide_style, axis=1))
    st.subheader('')

    st.write(
        '''
        The DataFrames above are sorted by descending accuracy. We've used a custom style function to color the cells according to whether a given option is at its default value or not. For both the Multinomial and Complement classifiers, the default choices for `TfidfVectorizer` resulted in the second-highest and third-highest performance, respectively. The differences between the default settings and the top-performing settings is in the fourth (Multinomial classifier) and third (Complement classifier) decimal place respectfully. These miniscule differences may very well be a statistical fluctuation with the initial shuffling of the data, and we can expect slightly different results without using the `random_state` parameter. This is strong verification that sticking with the default values for the arguments of `CountVectorizer` and `TfidfVectorizer` is probably a safe bet and is a choice we will continue to make going forward.
        '''
    )
    

    st.header('Intermediate conclusions and outlook')
    st.write(
        '''
        We've now explored in depth the options available to us for feature extraction that are shared by both `CountVectorizer` and `TfidfVectorizer`. The key take away is that for the 20 newsgroups dataset, **the default settings are typically best**, with a marginal improvement of including bi-grams by setting `ngram_range=(1,2)`. We also see in every case that the performance of classifiers trained on TF-IDF feature vectors is higher than that of classifiers trained on term frequency vectors or the combination of both. Interestingly, `CountVectorizer` alone resulted in the worst performance across the board. Augmenting with TF-IDF vectors only resulted in a marginal performance boost. However, employing TF-IDF vectors from `TfidfVectorizer` alone resulted in **_significantly_** better performance in every case. 

        Our results so far are: a `MultinomialNB` classifier, with `alpha=0.01` trained on TF-IDF feature vectors using default settings for `TfidfVectorizer`, achieves an accuracy of **76.74%**. On the same feature vectors, a `ComplementNB` classifier, with `alpha=0.3`, achieves an accuracy of **76.66%**. If we allow bi-grams in the feature space, these accuracies increase slightly to **77.70%** and **78.29%**, but at the cost of a near 10X dimensionality increase. If computational efficiency is highly important, it is hard to justify such a huge increase in features to achieve a 1-1.5% accuracy gain.

        So far we have focused on the accuracy score to judge the performance of our classifiers. This is a good metric for tuning parameters, since it is a single number that can be plotted against parameter values. Since we have our ideally tuned classifiers and feature sets, we can do some more detailed analysis using the tools included in the `sklearn.metrics` module. Two such tools are a classification report, which computes several statistics including the accuracy, and the confusion matrix, which is a summary of the correct and incorrect classifications made. Both tools, along with a stand-alone function for computing the accuracy score, can be imported using
        ```python
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        ```
        These functions and their outputs is covered in the **Mathematical Background** page. The output of `classification_report` is a listing of several statistics that give an overview of the performance of the classifier. The output of `confusion_matrix` is a square matrix with entries corresponding to the types of predictions made (true positives, false positives, true negatives, and false negatives). The classification report can be directly printed, or outputed as a key-value dictionary for easy access to each statistic's value. The confusion matrix is best viewed as a heatmap, similar to the ones we made when tuning the value of `ngram_range`. The confusion matrix should have large values on the main diagonal and small values elsewhere. The confusion matrix is also non-negative, meaning each value is zero or greater. When the confusion matrix is unnormalized, each value in row A column B corresponds to the number of predictions of class B for samples with a ground truth of class A. The matrix can be row (column) normalized, in which each element is divided by the sum of the elements in each row (column), or population normalized, where each element is divided by the number of predictions made. When the matrix is row normalized, the main diagonal entries correspond to the recall value for each class. When the matrix is column normalized, the main diagonal contains the precision of each class. When the matrix is normalized by population, the sum of the diagonal terms correspond to the model accuracy. In any case, a well-performing model has large values on the main diagonal and small values elsewhere. 
        
        We can include these additional metrics in our `train_and_score_classifier` function:
        ```python
        def train_and_score_classifiers(train_features, test_features, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the metrics
            metrics_dict = {}
            metrics_dict['Multinomial'] = {}
            metrics_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial = MultinomialNB(alpha=alpha_mnb)
            complement = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training features
            multinomial.fit(train_features, train_labels)
            complement.fit(train_features, train_labels)

            # Use trained classifiers to make predictions on the test set
            multinomial_predictions = multinomial.predict(test_features)
            complement_predictions = complement.predict(test_features)

            # Add the accuracies to the dictionary:
            metrics_dict['Multinomial']['accuracy'] = multinomial.score(test_features, test_labels)
            metrics_dict['Complement']['accuracy'] = complement.score(test_features, test_labels)

            # Add the classification report to the dictionary:
            metrics_dict['Multinomial']['report'] = classification_report(multinomial_predictions, test_labels)
            metrics_dict['Complement']['report'] = classification_report(complement_predictions, test_labels)

            # Add the confusion matrix to the dictionary:
            metrics_dict['Multinomial']['cm'] = confusion_matrix(multinomial_predictions, test_labels, normalize='pred')
            metrics_dict['Complement']['cm'] = confusion_matrix(complement_predictions, test_labels, normalize='pred')

            # Return the dictionary of results:
            return metrics_dict
        ```

        Below is a code block to perform more detailed analysis on our optimized classifiers. The confusion matrix heatmap figures can be enlarged by clicking the arrows in the top-right corner to see a larger view. 
        '''
    )

    # --------------------------------------------------------------
    # ----- Newsgroup classify samples into categories: 1-step -----
    # --------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from sklearn.metrics import classification_report, confusion_matrix

                # Load and split the data
                newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)

                # Extract features
                train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings)

                # Train and score classifiers
                metrics = train_and_score_classifiers(train_tfidfs, test_tfidfs, newsgroups_train.category, newsgroups_test.category)
                
                # Create labeled dataframs for the confusion matrices
                class_labels = sorted(set(newsgroups_test.category))
                metrics['Multinomial']['cm_df'] = pd.DataFrame(data=metrics['Multinomial']['cm'], columns=class_labels, index=class_labels)
                metrics['Complement']['cm_df'] = pd.DataFrame(data=metrics['Complement']['cm'], columns=class_labels, index=class_labels)

                # Initiate plots
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()

                fig1 = matrix_heatmap(metrics['Multinomial']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Multinomial classifier confusion matrix heatmap', 'Category', 'Category'), 'rotate x_tick_labels': True})

                fig2 = matrix_heatmap(metrics['Complement']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Complement classifier confusion matrix heatmap', 'Category', 'Category'), 'rotate x_tick_labels': True})

                # Show results and plots
                print('Topic classification results:')

                print('Multinomial classifier')
                print(f"Accuracy: {metrics['Multinomial']['accuracy']}")
                print('Classification report:')
                print(metrics['Multinomial']['report'])
                print('Confusion matrix:')
                print(metrics['Multinomial']['cm_df'])
                fig1.show()
                
                print('Complement classifier')
                print(f"Accuracy: {metrics['Complement']['accuracy']}")
                print('Classification report:')
                print(metrics['Complement']['report'])
                print('Confusion matrix:')
                print(metrics['Complement']['cm_df'])
                fig2.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroup_classify_samples_into_categories_run_button')
    st.subheader('Output:')
    if run_button:
        
        def load_and_split_newsgroups(training_size):
            # Load all of the data
            newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def extract_tfidfs(training_data, testing_data, opts={'norm': 'l2', 'use_idf': True, 'smooth_idf': True, 'sublinear_tf': False}):
            # Instantiate feature extractors
            tfidf_vectorizer = TfidfVectorizer(norm=opts['norm'], use_idf=opts['use_idf'], smooth_idf=opts['smooth_idf'], sublinear_tf=opts['sublinear_tf'], stop_words='english')

            # Fit the vectorizers by learning the vocabulary of the training set, then compute TF-IDFs
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into TF-IDFs
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_tfidfs, test_tfidfs
        
        def train_and_score_classifiers(train_features, test_features, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the metrics
            metrics_dict = {}
            metrics_dict['Multinomial'] = {}
            metrics_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial = MultinomialNB(alpha=alpha_mnb)
            complement = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training features
            multinomial.fit(train_features, train_labels)
            complement.fit(train_features, train_labels)

            # Use trained classifiers to make predictions on the test set
            multinomial_predictions = multinomial.predict(test_features)
            complement_predictions = complement.predict(test_features)

            # Add the accuracies to the dictionary:
            metrics_dict['Multinomial']['accuracy'] = multinomial.score(test_features, test_labels)
            metrics_dict['Complement']['accuracy'] = complement.score(test_features, test_labels)

            # Add the classification report to the dictionary:
            metrics_dict['Multinomial']['report'] = classification_report(multinomial_predictions, test_labels)
            metrics_dict['Complement']['report'] = classification_report(complement_predictions, test_labels)

            # Add the confusion matrix to the dictionary:
            metrics_dict['Multinomial']['cm'] = confusion_matrix(multinomial_predictions, test_labels, normalize='pred')
            metrics_dict['Complement']['cm'] = confusion_matrix(complement_predictions, test_labels, normalize='pred')

            # Return the dictionary of results:
            return metrics_dict

        # Load and split the data
        newsgroups_train, newsgroups_test = load_and_split_newsgroups(0.75)

        # Extract features
        train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings)

        # Train and score classifiers
        metrics = train_and_score_classifiers(train_tfidfs, test_tfidfs, newsgroups_train.category, newsgroups_test.category)
        
        # Create labeled dataframs for the confusion matrices
        class_labels = sorted(set(newsgroups_test.category))
        metrics['Multinomial']['cm_df'] = pd.DataFrame(data=metrics['Multinomial']['cm'], columns=class_labels, index=class_labels)
        metrics['Complement']['cm_df'] = pd.DataFrame(data=metrics['Complement']['cm'], columns=class_labels, index=class_labels)

        # Initiate plots
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        fig1 = matrix_heatmap(metrics['Multinomial']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Multinomial classifier confusion matrix heatmap', 'Category', 'Category'), 'rotate x_tick_labels': True})

        fig2 = matrix_heatmap(metrics['Complement']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Complement classifier confusion matrix heatmap', 'Category', 'Category'), 'rotate x_tick_labels': True})

        # Show results and plots
        st.subheader('**Topic classification results:**')
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.subheader('Multinomial classifier')
            st.write(f"**Accuracy:** {metrics['Multinomial']['accuracy']}")
            st.write('**Classification report:**')
            st.text('.  \n'+metrics['Multinomial']['report'])
            st.write('**Confusion matrix:**')
            st.write(metrics['Multinomial']['cm_df'])
            st.pyplot(fig1)
        with output_col2:
            st.subheader('Complement classifier')
            st.write(f"**Accuracy:** {metrics['Complement']['accuracy']}")
            st.write('**Classification report:**')
            st.text('.  \n'+metrics['Complement']['report'])
            st.write('**Confusion matrix:**')
            st.write(metrics['Complement']['cm_df'])
            st.pyplot(fig2)         
    st.subheader('')

    st.header('Two-step classification: a better approach?')
    st.write(
        '''
        Is there any other directions we can go to increase our classification performance? Perhaps. So far, we have been attempting to classify individual newsgroup postings into twenty categories in a single step. We have been ignoring the fact that these categories are more or less grouped into six larger sets of similar topics: Computation, Science, Recreation, Politics, Religion, and a catch-all topic Miscellaneous. By trying to predict the specific category all in one go, we are treating a mistake in prediction between related categories within the same topic the same as a mistake in prediction between completely unrelated topics!
        
        What if we changed our entire approach to classifying samples by splitting the problem into two parts. First, we classify a sample into one of the six broad topics. Then, we further classify the sample into the individual category within the broader topic. We would need _two_ independent classifiers to make a single prediction in the two-step approach.

        The first classifier would be trained with labels corresponding to the six broad topics. Since there are different numbers of categories per topic, this will imbalance the data, but would greatly increase the number of samples per topic. These topics are also quite distinct from each other, so we would expect the first classification step to have very high accuracy. The second classifier would be trained _only_ on samples from **within** the topic predicted by the first classifier, and will have _no knowledge_ of any information about samples belonging to the other topics.

        Recall from the very top of this page how the categories are divided into topics:
        '''
    )
    
    topic_df = pd.DataFrame(data={'topic': ['Computation', 'Science', 'Recreation', 'Politics', 'Religion', 'Miscellaneous'], 'categories':[['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'], ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'], ['misc.forsale']]})
    topic_df['num_categories'] = topic_df['categories'].apply(lambda x: len(x))
    st.write(topic_df)
    
    st.write(
        '''
        Note that the last topic, 'Miscellaneous', only has a single category, and therefore we do not need a second classifier to further classify any sample predicted to fall under this topic. Recall that the function used to load the data, `fetch_20newsgroups`, has a very useful argument, `categories` which allows only a select number of categories to be loaded. This makes it very convenient to train classifiers needed for the second step of the two-step process, since we can use this argument to ensure the data returned by `fetch_20newsgroups` will only contain categories from a given topic. We can modify our convenience function `load_and_split_newsgroups` to include this functionality like:
        ```python
        def load_and_split_newsgroups_categories(training_size, categories):
            # Load the data
            if len(categories) == 0:
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
            else:
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        ```
        The first thing to do is to load all of the data, and then assign each sample a new label based on its category. Then, we can see how well we can classify samples into their respective topics, which is the first step in the two-step approach.
        '''
    )
    
    # -----------------------------------------------------------------
    # ----- Newsgroup classify samples into topics: 2-step part 1 -----
    # -----------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from sklearn.metrics import classification_report, confusion_matrix

                # Create a dataframe holding the topic labels and their respective categories
                topics_list = ['Computation', 'Science', 'Recreation', 'Politics', 'Religion', 'Miscellaneous']
                categories_list = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'], ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'], ['misc.forsale']]
                topics_df = pd.DataFrame(data={'topic': topics_list, 'categories': categories_list})

                # Create dictionaries to link topics to their categories and categories to their topics
                topics_to_categories_dict = {}
                category_to_topics_dict = {}
                for i in range(len(topics_list)):
                    topics_to_categories_dict[topics_list[i]] = categories_list[i]
                    for j in range(len(categories_list[i])):
                        category_to_topics_dict[categories_list[i][j]] = topics_list[i]

                # Load and split the data
                newsgroups_train, newsgroups_test = load_and_split_newsgroups_categories(0.75)

                # Assign new topic label columns to data using topics dictionary
                newsgroups_train['topic'] = newsgroups_train['category'].apply(lambda x: category_to_topics_dict[x])
                newsgroups_test['topic'] = newsgroups_test['category'].apply(lambda x: category_to_topics_dict[x])

                # Extract features
                train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings)

                # Train and score classifiers
                metrics = train_and_score_classifiers(train_tfidfs, test_tfidfs, newsgroups_train.topic, newsgroups_test.topic)
                
                # Create labeled dataframs for the confusion matrices
                class_labels = sorted(topics_list)
                metrics['Multinomial']['cm_df'] = pd.DataFrame(data=metrics['Multinomial']['cm'], columns=class_labels, index=class_labels)
                metrics['Complement']['cm_df'] = pd.DataFrame(data=metrics['Complement']['cm'], columns=class_labels, index=class_labels)

                # Initiate plots
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()

                fig1 = matrix_heatmap(metrics['Multinomial']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Multinomial classifier confusion matrix heatmap', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

                fig2 = matrix_heatmap(metrics['Complement']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Complement classifier confusion matrix heatmap', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

                # Show results and plots
                print('Samples to topics classification results:')

                print('Multinomial classifier')
                print(f"Accuracy: {metrics['Multinomial']['accuracy']}")
                print('Classification report:')
                print(metrics['Multinomial']['report'])
                print('Confusion matrix:')
                print(metrics['Multinomial']['cm_df'])
                fig1.show()
                
                print('Complement classifier')
                print(f"Accuracy: {metrics['Complement']['accuracy']}")
                print('Classification report:')
                print(metrics['Complement']['report'])
                print('Confusion matrix:')
                print(metrics['Complement']['cm_df'])
                fig2.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroup_classify_samples_into_topics_run_button')
    st.subheader('Output:')
    if run_button:

        def load_and_split_newsgroups_categories(training_size, categories=[]):
            # Load the data
            if len(categories) == 0:
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
            else:
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def extract_tfidfs(training_data, testing_data, opts={'norm': 'l2', 'use_idf': True, 'smooth_idf': True, 'sublinear_tf': False}):
            # Instantiate feature extractors
            tfidf_vectorizer = TfidfVectorizer(norm=opts['norm'], use_idf=opts['use_idf'], smooth_idf=opts['smooth_idf'], sublinear_tf=opts['sublinear_tf'], stop_words='english')

            # Fit the vectorizers by learning the vocabulary of the training set, then compute TF-IDFs
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into TF-IDFs
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_tfidfs, test_tfidfs
        
        def train_and_score_classifiers(train_tfidfs, test_tfidfs, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the metrics
            metrics_dict = {}
            metrics_dict['Multinomial'] = {}
            metrics_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial_tfidfs = MultinomialNB(alpha=alpha_mnb)
            complement_tfidfs = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training features
            multinomial_tfidfs.fit(train_tfidfs, train_labels)
            complement_tfidfs.fit(train_tfidfs, train_labels)

            # Use trained classifiers to make predictions on the test set
            multinomial_predictions = multinomial_tfidfs.predict(test_tfidfs)
            complement_predictions = complement_tfidfs.predict(test_tfidfs)

            # Add the accuracies to the dictionary:
            metrics_dict['Multinomial']['accuracy'] = multinomial_tfidfs.score(test_tfidfs, test_labels)
            metrics_dict['Complement']['accuracy'] = complement_tfidfs.score(test_tfidfs, test_labels)

            # Add the classification report to the dictionary:
            metrics_dict['Multinomial']['report'] = classification_report(multinomial_predictions, test_labels)
            metrics_dict['Complement']['report'] = classification_report(complement_predictions, test_labels)

            # Add the confusion matrix to the dictionary:
            metrics_dict['Multinomial']['cm'] = confusion_matrix(multinomial_predictions, test_labels, normalize='pred')
            metrics_dict['Complement']['cm'] = confusion_matrix(complement_predictions, test_labels, normalize='pred')

            # Return the dictionary of results:
            return metrics_dict
        
        # Create a dataframe holding the topic labels and their respective categories
        topics_list = ['Computation', 'Science', 'Recreation', 'Politics', 'Religion', 'Miscellaneous']
        categories_list = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'], ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'], ['misc.forsale']]
        topics_df = pd.DataFrame(data={'topic': topics_list, 'categories': categories_list})

        # Create dictionaries to link topics to their categories and categories to their topics
        topics_to_categories_dict = {}
        category_to_topics_dict = {}
        for i in range(len(topics_list)):
            topics_to_categories_dict[topics_list[i]] = categories_list[i]
            for j in range(len(categories_list[i])):
                category_to_topics_dict[categories_list[i][j]] = topics_list[i]

        # Load and split the data
        newsgroups_train, newsgroups_test = load_and_split_newsgroups_categories(0.75)

        # Assign new topic label columns to data using topics dictionary
        newsgroups_train['topic'] = newsgroups_train['category'].apply(lambda x: category_to_topics_dict[x])
        newsgroups_test['topic'] = newsgroups_test['category'].apply(lambda x: category_to_topics_dict[x])

        # Extract features
        train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings)

        # Train and score classifiers
        metrics = train_and_score_classifiers(train_tfidfs, test_tfidfs, newsgroups_train.topic, newsgroups_test.topic)
        
        # Create labeled dataframs for the confusion matrices
        class_labels = sorted(topics_list)
        metrics['Multinomial']['cm_df'] = pd.DataFrame(data=metrics['Multinomial']['cm'], columns=class_labels, index=class_labels)
        metrics['Complement']['cm_df'] = pd.DataFrame(data=metrics['Complement']['cm'], columns=class_labels, index=class_labels)

        # Initiate plots
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        fig1 = matrix_heatmap(metrics['Multinomial']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Multinomial classifier confusion matrix heatmap', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

        fig2 = matrix_heatmap(metrics['Complement']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Complement classifier confusion matrix heatmap', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

        # Show results and plots
        st.subheader('**Samples to topics classification results:**')
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.subheader('Multinomial classifier')
            st.write(f"**Accuracy:** {metrics['Multinomial']['accuracy']}")
            st.write('**Classification report:**')
            st.text('.  \n'+metrics['Multinomial']['report'])
            st.write('**Confusion matrix:**')
            st.write(metrics['Multinomial']['cm_df'])
            st.pyplot(fig1)
        with output_col2:
            st.subheader('Complement classifier')
            st.write(f"**Accuracy:** {metrics['Complement']['accuracy']}")
            st.write('**Classification report:**')
            st.text('.  \n'+metrics['Complement']['report'])
            st.write('**Confusion matrix:**')
            st.write(metrics['Complement']['cm_df'])
            st.pyplot(fig2)         
    st.subheader('')
    
    st.write(
        '''
        The results so far are promising. Both the Multinomial and Complement classifiers can predict the broad topic of samples with around 85% accuracy. The next step is to subdivide the data by topic, and train classifiers on each topic to further classifier samples into categories. We loop over the different topics, loading only the categories in that topic using the `categories` argument of `fetch_20newsgroups()`. Then we perform the same analysis as we did above on each topic. Run the code below to see the results.
        '''
    )

    # --------------------------------------------------------------------
    # ----- Newsgroup classify topics into categories: 2-step part 2 -----
    # --------------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from sklearn.metrics import classification_report, confusion_matrix

                # Create a dataframe holding the topic labels and their respective categories
                topics_list = ['Computation', 'Science', 'Recreation', 'Politics', 'Religion', 'Miscellaneous']
                categories_list = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'], ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'], ['misc.forsale']]
                topics_df = pd.DataFrame(data={'topic': topics_list, 'categories': categories_list})

                # Create dictionaries to link topics to their categories and categories to their topics
                topics_to_categories_dict = {}
                category_to_topics_dict = {}
                for i in range(len(topics_list)):
                    topics_to_categories_dict[topics_list[i]] = categories_list[i]
                    for j in range(len(categories_list[i])):
                        category_to_topics_dict[categories_list[i][j]] = topics_list[i]
                
                # Loop over the topics, excluding the `Miscellaneous` topic since it only has one category
                # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
                accuracy_df = pd.DataFrame(columns=['topic', 'Classifier', 'TF-IDF accuracy'])
                
                for topic in topics_list:
                    if topic != 'Miscellaneous':
                        # Load and split the data, using the category argument and topics_to_categories_dict dictionary to extract only categories in each topic
                        newsgroups_train, newsgroups_test = load_and_split_newsgroups_categories(0.75, categories=topics_to_categories_dict[topic])

                        # Extract features
                        train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings)

                        # Train and score classifiers
                        metrics = train_and_score_classifiers(train_tfidfs, test_tfidfs, newsgroups_train.category, newsgroups_test.category)
                
                        # Create labeled dataframes for the confusion matrices
                        class_labels = sorted(topics_to_categories_dict[topic])
                        metrics['Multinomial']['cm_df'] = pd.DataFrame(data=metrics['Multinomial']['cm'], columns=class_labels, index=class_labels)
                        metrics['Complement']['cm_df'] = pd.DataFrame(data=metrics['Complement']['cm'], columns=class_labels, index=class_labels)

                        # Append row to accuracy dataframe
                        accuracy_df = accuracy_df.append({'topic': topic,
                                                        'Classifier': 'Multinomial',
                                                        'TF-IDF accuracy': metrics['Multinomial']['accuracy']},
                                                        ignore_index=True)
                        accuracy_df = accuracy_df.append({'topic': topic,
                                                        'Classifier': 'Complement',
                                                        'TF-IDF accuracy': metrics['Complement']['accuracy']},
                                                        ignore_index=True)

                        # Initiate plots
                        fig1, ax1 = plt.subplots()
                        fig2, ax2 = plt.subplots()

                        fig1 = matrix_heatmap(metrics['Multinomial']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Multinomial classifier confusion matrix heatmap', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

                        fig2 = matrix_heatmap(metrics['Complement']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Complement classifier confusion matrix heatmap', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

                        # Show results and plots
                        print(f'Topics to Categories classification results: {topic}')

                        print('Multinomial classifier')
                        print(f"Accuracy: {metrics['Multinomial']['accuracy']}")
                        print('Classification report:')
                        print(metrics['Multinomial']['report'])
                        print('Confusion matrix:')
                        print(metrics['Multinomial']['cm_df'])
                        fig1.show()
                        
                        print('Complement classifier')
                        print(f"Accuracy: {metrics['Complement']['accuracy']}")
                        print('Classification report:')
                        print(metrics['Complement']['report'])
                        print('Confusion matrix:')
                        print(metrics['Complement']['cm_df'])
                        fig2.show()

                # Create a bar chart of accuracies by topic
                fig, ax = plt.subplots()

                sns.barplotdata=accuracy_df, x='topic', y='accuracy', hue='Classifier', ax=ax)
                ax.set_title('Topic to category classification accuracies')

                print('Accuracy dataframe:')
                print(accuracy_df)
                fig.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroup_classify_topics_into_categories_run_button')
    st.subheader('Output:')
    if run_button:

        def load_and_split_newsgroups_categories(training_size, categories=[]):
            # Load the data
            if len(categories) == 0:
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
            else:
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def extract_tfidfs(training_data, testing_data, opts={'norm': 'l2', 'use_idf': True, 'smooth_idf': True, 'sublinear_tf': False}):
            # Instantiate feature extractors
            tfidf_vectorizer = TfidfVectorizer(norm=opts['norm'], use_idf=opts['use_idf'], smooth_idf=opts['smooth_idf'], sublinear_tf=opts['sublinear_tf'], stop_words='english')

            # Fit the vectorizers by learning the vocabulary of the training set, then compute TF-IDFs
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into TF-IDFs
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_tfidfs, test_tfidfs
        
        def train_and_score_classifiers(train_tfidfs, test_tfidfs, train_labels, test_labels):
            # Create a dictionary of dictionaries to hold the metrics
            metrics_dict = {}
            metrics_dict['Multinomial'] = {}
            metrics_dict['Complement'] = {}

            # Initialize the classifiers
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial_tfidfs = MultinomialNB(alpha=alpha_mnb)
            complement_tfidfs = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training features
            multinomial_tfidfs.fit(train_tfidfs, train_labels)
            complement_tfidfs.fit(train_tfidfs, train_labels)

            # Use trained classifiers to make predictions on the test set
            multinomial_predictions = multinomial_tfidfs.predict(test_tfidfs)
            complement_predictions = complement_tfidfs.predict(test_tfidfs)

            # Add the accuracies to the dictionary:
            metrics_dict['Multinomial']['accuracy'] = multinomial_tfidfs.score(test_tfidfs, test_labels)
            metrics_dict['Complement']['accuracy'] = complement_tfidfs.score(test_tfidfs, test_labels)

            # Add the classification report to the dictionary:
            metrics_dict['Multinomial']['report'] = classification_report(multinomial_predictions, test_labels)
            metrics_dict['Complement']['report'] = classification_report(complement_predictions, test_labels)

            # Add the confusion matrix to the dictionary:
            metrics_dict['Multinomial']['cm'] = confusion_matrix(multinomial_predictions, test_labels, normalize='pred')
            metrics_dict['Complement']['cm'] = confusion_matrix(complement_predictions, test_labels, normalize='pred')

            # Return the dictionary of results:
            return metrics_dict
        
        # Create a dataframe holding the topic labels and their respective categories
        topics_list = ['Computation', 'Science', 'Recreation', 'Politics', 'Religion', 'Miscellaneous']
        categories_list = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'], ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'], ['misc.forsale']]
        topics_df = pd.DataFrame(data={'topic': topics_list, 'categories': categories_list})

        # Create dictionaries to link topics to their categories and categories to their topics
        topics_to_categories_dict = {}
        category_to_topics_dict = {}
        for i in range(len(topics_list)):
            topics_to_categories_dict[topics_list[i]] = categories_list[i]
            for j in range(len(categories_list[i])):
                category_to_topics_dict[categories_list[i][j]] = topics_list[i]
        
        # Loop over the topics, excluding the `Miscellaneous` topic since it only has one category
        # Create an empty dataframe to hold the parameter values and resulting classifier accuracies
        accuracy_df = pd.DataFrame(columns=['Topic', 'Classifier', 'Accuracy'])

        # Display status message
        with st.spinner('Training individual topic to categories classifiers...'):

            # Make a streamlit progress bar
            topic_progress = 0.0
            progress_bar = st.progress(topic_progress)
            
            for topic in topics_list:
                if topic != 'Miscellaneous':
                    # Load and split the data, using the category argument and topics_to_categories_dict dictionary to extract only categories in each topic
                    newsgroups_train, newsgroups_test = load_and_split_newsgroups_categories(0.75, categories=topics_to_categories_dict[topic])

                    # Extract features
                    train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings)

                    # Train and score classifiers
                    metrics = train_and_score_classifiers(train_tfidfs, test_tfidfs, newsgroups_train.category, newsgroups_test.category)
            
                    # Create labeled dataframes for the confusion matrices
                    class_labels = sorted(topics_to_categories_dict[topic])
                    metrics['Multinomial']['cm_df'] = pd.DataFrame(data=metrics['Multinomial']['cm'], columns=class_labels, index=class_labels)
                    metrics['Complement']['cm_df'] = pd.DataFrame(data=metrics['Complement']['cm'], columns=class_labels, index=class_labels)

                    # Append row to accuracy dataframe
                    accuracy_df = accuracy_df.append({'Topic': topic,
                                                    'Classifier': 'Multinomial',
                                                    'Accuracy': metrics['Multinomial']['accuracy']},
                                                    ignore_index=True)
                    accuracy_df = accuracy_df.append({'Topic': topic,
                                                    'Classifier': 'Complement',
                                                    'Accuracy': metrics['Complement']['accuracy']},
                                                    ignore_index=True)

                    # Initiate plots
                    fig1, ax1 = plt.subplots()
                    fig2, ax2 = plt.subplots()

                    fig1 = matrix_heatmap(metrics['Multinomial']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Multinomial classifier confusion matrix heatmap', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

                    fig2 = matrix_heatmap(metrics['Complement']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Complement classifier confusion matrix heatmap', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

                    # Show results and plots
                    st.subheader(f'**Topics to Categories classification results: {topic}**')
                    output_col1, output_col2 = st.beta_columns(2)
                    with output_col1:
                        st.subheader('Multinomial classifier')
                        st.write(f"**Accuracy:** {metrics['Multinomial']['accuracy']}")
                        st.write('**Classification report:**')
                        st.text('.  \n'+metrics['Multinomial']['report'])
                        st.write('**Confusion matrix:**')
                        st.write(metrics['Multinomial']['cm_df'])
                        st.pyplot(fig1)
                    with output_col2:
                        st.subheader('Complement classifier')
                        st.write(f"**Accuracy:** {metrics['Complement']['accuracy']}")
                        st.write('**Classification report:**')
                        st.text('.  \n'+metrics['Complement']['report'])
                        st.write('**Confusion matrix:**')
                        st.write(metrics['Complement']['cm_df'])
                        st.pyplot(fig2)

                    # Close the figures to clear memory
                    plt.close(fig1)
                    plt.close(fig2)
                
                    # Update progress bar
                    topic_progress += 1
                    progress_bar.progress(topic_progress/(len(topics_list)-1))
        
        # Clear progress bar
        progress_bar.empty()

        # Create a bar chart of accuracies by topic
        fig, ax = plt.subplots()

        sns.barplot(data=accuracy_df, x='Topic', y='Accuracy', hue='Classifier', ax=ax)
        ax.set_title('Topic to category classification accuracies')

        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.write('**Accuracy dataframe:**')
            st.write(accuracy_df)
        with output_col2:
            st.pyplot(fig)
    st.subheader('')

    st.write(
        '''
        From the above results, the classification accuracy from topics into categories is about 80% across the five topics that have at least two categories. The topic with the most distinguishable categories is 'Science', with an accuracy of 89%, and the topic with the least distinguishable categories is 'Religion' with about 73% accuracy.

        The remaining task is to combine the two steps into one process. We need to be careful with how we go about doing this. First, we need to load all of the data and split it into training and testing sets. We need to make sure that the entire training process is carried out using _only_ data from the training set. This means there will be no _information leakage_ from the testing set into the training process and feature extraction. Then, we need to keep track of the feature extractors used for each topic subset. This is because we do not want information leakage between classification in different topics. The overall process can be structured like:

        **Training classifiers to predict the topic of each sample:** 

         1. Load  the data **_once_** and split into training and testing sets. Add a new `topic` column containing each sample's actual topic
         2. Extract the features of the training set using a vectorizer. Use the fit vectorizer to make feature vectors of the training and testing samples
         3. Use the feature vectors of the training set to train classifiers using the samples' `topic` as the class labels
         4. Use the trained classifiers to predict the topic of the testing set and save in a new column `predicted topic`: to be used later
        
        **Training classifiers to predict the category within each topic**

         1. Sub-divide the training set by actual `topic` (_not_ the predicted topic above)
         2. Extract the features of each training subset using a vectorizer. Use the fit vectorizer to make feature vectors of the training set
         2. Use the training subset feature vectors to train new classifiers using the `category` as the class label
         3. Keep track of the fit vectorizers for each topic: to be used later
        
        **Predicting the category of the testing set**

         1. Loop over each sample in the testing set. Each sample of the testing set should have a `predicted topic` from the steps training the topic classifiers. For each sample:
         2. Use the sample's `predicted topic` to select the corresponding fit vectorizer, use the vectorizer for that topic to create a new feature vector
         3. Use the trained classifiers for the sample's `predicted topic` to predict the category within that topic using the sample's new feature vector
         4. Save the prediction in a new column `predicted category`
        
        Once the above steps are carried out, each sample in the testing set will now have a ground truth label for `category` and `topic`. Each sample in the testing set will also have a `predicted topic` and `predicted category` label. This pairs of ground truth and predicted labels can be used to create metrics used to judge the performance of the two-step classification method.
        '''
    )
    
    # ----------------------------------------------------------------------
    # ----- Newsgroup classify topics into categories: 2-step combined -----
    # ----------------------------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                from sklearn.datasets import fetch_20newsgroups
                from sklearn.model_selection import train_test_split
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.naive_bayes import MultinomialNB, ComplementNB
                from sklearn.metrics import accuracy_report, classification_report, confusion_matrix

                # Create a dataframe holding the topic labels and their respective categories
                topics_list = ['Computation', 'Science', 'Recreation', 'Politics', 'Religion', 'Miscellaneous']
                categories_list = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'], ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'], ['misc.forsale']]
                topics_df = pd.DataFrame(data={'topic': topics_list, 'categories': categories_list})

                # Create dictionaries to link topics to their categories and categories to their topics
                topics_to_categories_dict = {}
                category_to_topics_dict = {}
                for i in range(len(topics_list)):
                    topics_to_categories_dict[topics_list[i]] = categories_list[i]
                    for j in range(len(categories_list[i])):
                        category_to_topics_dict[categories_list[i][j]] = topics_list[i]

                # Load and split the data
                newsgroups_train, newsgroups_test = load_and_split_newsgroups_categories(0.75)

                # Assign new topic label columns to data using topics dictionary
                newsgroups_train['topic'] = newsgroups_train['category'].apply(lambda x: category_to_topics_dict[x])
                newsgroups_test['topic'] = newsgroups_test['category'].apply(lambda x: category_to_topics_dict[x])

                # -------------------------------------
                # ----- Initial topic prediction: -----
                # -------------------------------------

                # Extract features for topic prediction
                train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings)

                # Initialize the classifiers for topic prediction
                alpha_mnb=0.01
                alpha_cnb=0.3
                multinomial_topic_classifier = MultinomialNB(alpha=alpha_mnb)
                complement_topic_classifier = ComplementNB(alpha=alpha_cnb)

                # Train the classifiers on the training features using the 'topic' as the class label
                multinomial_topic_classifier.fit(train_tfidfs, newsgroups_train.topic)
                complement_topic_classifier.fit(train_tfidfs, newsgroups_train.topic)

                # Use trained classifiers to make predictions on the test set
                # Create new columns in the testing datafram for topic predictions
                newsgroups_test['Multinomial predicted topic'] = multinomial_topic_classifier.predict(test_tfidfs)
                newsgroups_test['Complement predicted topic'] = complement_topic_classifier.predict(test_tfidfs)

                # -----------------------------------------------------------
                # ----- Train individual topic_to_category classifiers: -----
                # -----------------------------------------------------------

                # Create dictionaries to hold vectorizers and classifiers for each topid
                topic_to_categories_vectorizers = {}
                topic_to_categories_classifiers = {}

                # Loop over topics
                for topic in topics_list:
                    # Skip 'Miscellaneous' topic since it contains only one category
                    if topic != 'Miscellaneous':
                        
                        # Create sub-dictionaries to hold each pair of classifiers
                        topic_to_categories_classifiers[topic] = {}

                        # Select subset of training data:
                        training_subset = newsgroups_train[newsgroups_train['topic']==topic]

                        # Instantiate feature extractors
                        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))

                        # Fit the vectorizers by learning the vocabulary of the training set, then compute TF-IDFs
                        train_tfidfs = tfidf_vectorizer.fit_transform(training_subset.postings)

                        # Initialize the classifiers for topic to category prediction
                        alpha_mnb=0.01
                        alpha_cnb=0.3
                        multinomial_topic_to_category_classifier = MultinomialNB(alpha=alpha_mnb)
                        complement_topic_to_category_classifier = ComplementNB(alpha=alpha_cnb)

                        # Train the classifiers on the training subset features using the 'category' as the class label
                        multinomial_topic_to_category_classifier.fit(train_tfidfs, training_subset.category)
                        complement_topic_to_category_classifier.fit(train_tfidfs, training_subset.category)

                        # Store the trained vectorizer and classifiers in the dictionaries for later reference:
                        topic_to_categories_vectorizers[topic] = tfidf_vectorizer
                        topic_to_categories_classifiers[topic]['Multinomial'] = multinomial_topic_to_category_classifier
                        topic_to_categories_classifiers[topic]['Complement'] = complement_topic_to_category_classifier
                
                # -----------------------------------------------
                # ----- Predict categories of test samples: -----
                # -----------------------------------------------

                # Create empty lists to hold category predictions
                multinomial_category_predictions = []
                complement_category_predictions = []

                # Loop over the samples in the testing set using the dataframe index:
                for idx in range(len(newsgroups_test)):

                    # Extract the sample:
                    sample = newsgroups_test.iloc[idx]

                    # Extract the feature vector using the predicted topic to select the corresponding vectorizer:

                    # Topic predicted with the Multinomial topic classifier:
                    predicted_topic = sample['Multinomial predicted topic']

                    # If the predicted topic is 'Miscellaneous', then the predicted category is 'misc.forsale'
                    if predicted_topic == 'Miscellaneous':

                        # Append 'misc.forsale' to the category prediction lists
                        multinomial_category_predictions.append('misc.forsale')
                    
                    else:
                    
                        # Extract the features
                        sample_features = topic_to_categories_vectorizers[predicted_topic].transform([sample.postings])

                        # Predict the category using the extracted features
                        predicted_category = topic_to_categories_classifiers[predicted_topic]['Multinomial'].predict(sample_features.toarray())

                        # Append the predicted category to the respective list
                        multinomial_category_predictions.append(predicted_category[0])

                    # Topic predicted with the Complement topic classifier:
                    predicted_topic = sample['Complement predicted topic']

                    # If the predicted topic is 'Miscellaneous', then the predicted category is 'misc.forsale'
                    if predicted_topic == 'Miscellaneous':
                        
                        # Append 'misc.forsale' to the category prediction lists
                        complement_category_predictions.append('misc.forsale')
                    
                    else:
                    
                        # Extract the features
                        sample_features = topic_to_categories_vectorizers[predicted_topic].transform([sample.postings])

                        # Predict the category using the extracted features
                        predicted_category = topic_to_categories_classifiers[predicted_topic]['Complement'].predict(sample_features.toarray())

                        # Append the predicted category to the respective list
                        complement_category_predictions.append(predicted_category[0])
                
                # Create new columns in the testing datafram for category predictions
                newsgroups_test['Multinomial predicted category'] = multinomial_category_predictions
                newsgroups_test['Complement predicted category'] = complement_category_predictions

                # -------------------------------------------------------
                # ----- Compute metrics for two-step classification -----
                # -------------------------------------------------------
                
                # Create a dictionary to store the metrics
                metrics = {}
                metrics['Multinomial'] = {}
                metrics['Complement'] = {}

                # Compute accuracy scores:
                metrics['Multinomial']['accuracy'] = accuracy_score(newsgroups_test['category'], newsgroups_test['Multinomial predicted category'])
                metrics['Complement']['accuracy'] = accuracy_score(newsgroups_test['category'], newsgroups_test['Complement predicted category'])

                # Compute the classification reports:
                metrics['Multinomial']['report'] = classification_report(newsgroups_test['category'], newsgroups_test['Multinomial predicted category'])
                metrics['Complement']['report'] = classification_report(newsgroups_test['category'], newsgroups_test['Complement predicted category'])

                # Comput the confusion matrices:
                metrics['Multinomial']['cm'] = confusion_matrix(newsgroups_test['category'], newsgroups_test['Multinomial predicted category'], normalize='pred')
                metrics['Complement']['cm'] = confusion_matrix(newsgroups_test['category'], newsgroups_test['Complement predicted category'], normalize='pred')

                # Create labeled dataframes for the confusion matrices
                class_labels = sorted(set(newsgroups_test.category))
                metrics['Multinomial']['cm_df'] = pd.DataFrame(data=metrics['Multinomial']['cm'], columns=class_labels, index=class_labels)
                metrics['Complement']['cm_df'] = pd.DataFrame(data=metrics['Complement']['cm'], columns=class_labels, index=class_labels)

                # Initiate plots
                fig1, ax1 = plt.subplots()
                fig2, ax2 = plt.subplots()

                fig1 = matrix_heatmap(metrics['Multinomial']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Two-step classification confusion matrix heatmap Multinomial classifiers', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

                fig2 = matrix_heatmap(metrics['Complement']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Two-step classification confusion matrix heatmap Complement classifiers', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

                # Show results and plots
                print(f'**Two-step classification results:**')

                print('Multinomial classifiers')
                print(f"Accuracy: {metrics['Multinomial']['accuracy']}")
                print('Classification report:')
                print(metrics['Multinomial']['report'])
                print('Confusion matrix:')
                print(metrics['Multinomial']['cm_df'])
                fig1.show()
                
                print('Complement classifiers')
                print(f"Accuracy: {metrics['Complement']['accuracy']}")
                print('Classification report:')
                print(metrics['Complement']['report'])
                print('Confusion matrix:')
                print(metrics['Complement']['cm_df'])
                fig2.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='newsgroup_two_step_classification_run_button')
    st.subheader('Output:')
    if run_button:

        def load_and_split_newsgroups_categories(training_size, categories=[]):
            # Load the data
            if len(categories) == 0:
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', remove=('headers', 'footers', 'quotes'), random_state=42)
            else:
                newsgroups_all = fetch_20newsgroups(subset='all', shuffle='true', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=42)

            # Create a dataframe with each text sample and category
            newsgroups_df = pd.DataFrame(data={'postings': newsgroups_all.data, 'category': newsgroups_all.target})

            # Replace the category value with corresponding name
            newsgroups_df.category.replace({idx: cat for idx, cat in enumerate(newsgroups_all.target_names)}, inplace=True)

            # Split the data
            newsgroups_train, newsgroups_test = train_test_split(newsgroups_df, train_size=training_size, shuffle=True, random_state=42)

            # Return the training and testing subsets
            return newsgroups_train, newsgroups_test
        
        def extract_tfidfs(training_data, testing_data, opts={'norm': 'l2', 'use_idf': True, 'smooth_idf': True, 'sublinear_tf': False}):
            # Instantiate feature extractors
            tfidf_vectorizer = TfidfVectorizer(norm=opts['norm'], use_idf=opts['use_idf'], smooth_idf=opts['smooth_idf'], sublinear_tf=opts['sublinear_tf'], stop_words='english')

            # Fit the vectorizers by learning the vocabulary of the training set, then compute TF-IDFs
            train_tfidfs = tfidf_vectorizer.fit_transform(training_data)

            # Use the fit vectorizers to transform the testing set into TF-IDFs
            test_tfidfs = tfidf_vectorizer.transform(testing_data)

            # Return the feature vectors for the training and testing sets
            return train_tfidfs, test_tfidfs
        
        # Create a dataframe holding the topic labels and their respective categories
        topics_list = ['Computation', 'Science', 'Recreation', 'Politics', 'Religion', 'Miscellaneous']
        categories_list = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'], ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'], ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'], ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'], ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian'], ['misc.forsale']]
        topics_df = pd.DataFrame(data={'topic': topics_list, 'categories': categories_list})

        # Create dictionaries to link topics to their categories and categories to their topics
        topics_to_categories_dict = {}
        category_to_topics_dict = {}
        for i in range(len(topics_list)):
            topics_to_categories_dict[topics_list[i]] = categories_list[i]
            for j in range(len(categories_list[i])):
                category_to_topics_dict[categories_list[i][j]] = topics_list[i]

        # Load and split the data
        newsgroups_train, newsgroups_test = load_and_split_newsgroups_categories(0.75)

        # Assign new topic label columns to data using topics dictionary
        newsgroups_train['topic'] = newsgroups_train['category'].apply(lambda x: category_to_topics_dict[x])
        newsgroups_test['topic'] = newsgroups_test['category'].apply(lambda x: category_to_topics_dict[x])

        # -------------------------------------
        # ----- Initial topic prediction: -----
        # -------------------------------------

        # Display status message
        with st.spinner('Training initial topic classifier...'):
            # Extract features for topic prediction
            train_tfidfs, test_tfidfs = extract_tfidfs(newsgroups_train.postings, newsgroups_test.postings)

            # Initialize the classifiers for topic prediction
            alpha_mnb=0.01
            alpha_cnb=0.3
            multinomial_topic_classifier = MultinomialNB(alpha=alpha_mnb)
            complement_topic_classifier = ComplementNB(alpha=alpha_cnb)

            # Train the classifiers on the training features using the 'topic' as the class label
            multinomial_topic_classifier.fit(train_tfidfs, newsgroups_train.topic)
            complement_topic_classifier.fit(train_tfidfs, newsgroups_train.topic)

            # Use trained classifiers to make predictions on the test set
            # Create new columns in the testing datafram for topic predictions
            newsgroups_test['Multinomial predicted topic'] = multinomial_topic_classifier.predict(test_tfidfs)
            newsgroups_test['Complement predicted topic'] = complement_topic_classifier.predict(test_tfidfs)

        # -----------------------------------------------------------
        # ----- Train individual topic_to_category classifiers: -----
        # -----------------------------------------------------------

        # Display status message
        with st.spinner('Training individual topic to categories classifiers...'):

            # Make a streamlit progress bar
            topic_progress = 0.0
            progress_bar = st.progress(topic_progress)

            # Create dictionaries to hold vectorizers and classifiers for each topid
            topic_to_categories_vectorizers = {}
            topic_to_categories_classifiers = {}

            # Loop over topics
            for topic in topics_list:
                # Skip 'Miscellaneous' topic since it contains only one category
                if topic != 'Miscellaneous':
                    
                    # Create sub-dictionaries to hold each pair of classifiers
                    topic_to_categories_classifiers[topic] = {}

                    # Select subset of training data:
                    training_subset = newsgroups_train[newsgroups_train['topic']==topic]

                    # Instantiate feature extractors
                    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))

                    # Fit the vectorizers by learning the vocabulary of the training set, then compute TF-IDFs
                    train_tfidfs = tfidf_vectorizer.fit_transform(training_subset.postings)

                    # Initialize the classifiers for topic to category prediction
                    alpha_mnb=0.01
                    alpha_cnb=0.3
                    multinomial_topic_to_category_classifier = MultinomialNB(alpha=alpha_mnb)
                    complement_topic_to_category_classifier = ComplementNB(alpha=alpha_cnb)

                    # Train the classifiers on the training subset features using the 'category' as the class label
                    multinomial_topic_to_category_classifier.fit(train_tfidfs, training_subset.category)
                    complement_topic_to_category_classifier.fit(train_tfidfs, training_subset.category)

                    # Store the trained vectorizer and classifiers in the dictionaries for later reference:
                    topic_to_categories_vectorizers[topic] = tfidf_vectorizer
                    topic_to_categories_classifiers[topic]['Multinomial'] = multinomial_topic_to_category_classifier
                    topic_to_categories_classifiers[topic]['Complement'] = complement_topic_to_category_classifier

                    # Update progress bar
                    topic_progress += 1
                    progress_bar.progress(topic_progress/(len(topics_list)-1))
            
            # Clear progress bar
            progress_bar.empty()
        
        # -----------------------------------------------
        # ----- Predict categories of test samples: -----
        # -----------------------------------------------

        # Display status message
        with st.spinner('Making category predictions on the testing samples...'):

            # Make a streamlit progress bar
            sample_progress = 0
            progress_bar = st.progress(sample_progress)

            # Create empty lists to hold category predictions
            multinomial_category_predictions = []
            complement_category_predictions = []

            # Loop over the samples in the testing set using the dataframe index:
            for idx in range(len(newsgroups_test)):

                # Extract the sample:
                sample = newsgroups_test.iloc[idx]

                # Extract the feature vector using the predicted topic to select the corresponding vectorizer:

                # Topic predicted with the Multinomial topic classifier:
                predicted_topic = sample['Multinomial predicted topic']

                # If the predicted topic is 'Miscellaneous', then the predicted category is 'misc.forsale'
                if predicted_topic == 'Miscellaneous':

                    # Append 'misc.forsale' to the category prediction lists
                    multinomial_category_predictions.append('misc.forsale')
                
                else:
                
                    # Extract the features
                    sample_features = topic_to_categories_vectorizers[predicted_topic].transform([sample.postings])

                    # Predict the category using the extracted features
                    predicted_category = topic_to_categories_classifiers[predicted_topic]['Multinomial'].predict(sample_features.toarray())

                    # Append the predicted category to the respective list
                    multinomial_category_predictions.append(predicted_category[0])

                # Topic predicted with the Complement topic classifier:
                predicted_topic = sample['Complement predicted topic']

                # If the predicted topic is 'Miscellaneous', then the predicted category is 'misc.forsale'
                if predicted_topic == 'Miscellaneous':
                    
                    # Append 'misc.forsale' to the category prediction lists
                    complement_category_predictions.append('misc.forsale')
                
                else:
                
                    # Extract the features
                    sample_features = topic_to_categories_vectorizers[predicted_topic].transform([sample.postings])

                    # Predict the category using the extracted features
                    predicted_category = topic_to_categories_classifiers[predicted_topic]['Complement'].predict(sample_features.toarray())

                    # Append the predicted category to the respective list
                    complement_category_predictions.append(predicted_category[0])
                
                # Update progress bar
                sample_progress += 1
                progress_bar.progress(sample_progress/len(newsgroups_test))
            
            # Create new columns in the testing datafram for category predictions
            newsgroups_test['Multinomial predicted category'] = multinomial_category_predictions
            newsgroups_test['Complement predicted category'] = complement_category_predictions

            # Clear progress bar
            progress_bar.empty()

        # -------------------------------------------------------
        # ----- Compute metrics for two-step classification -----
        # -------------------------------------------------------
        
        # Create a dictionary to store the metrics
        metrics = {}
        metrics['Multinomial'] = {}
        metrics['Complement'] = {}

        # Compute accuracy scores:
        metrics['Multinomial']['accuracy'] = accuracy_score(newsgroups_test['category'], newsgroups_test['Multinomial predicted category'])
        metrics['Complement']['accuracy'] = accuracy_score(newsgroups_test['category'], newsgroups_test['Complement predicted category'])

        # Compute the classification reports:
        metrics['Multinomial']['report'] = classification_report(newsgroups_test['category'], newsgroups_test['Multinomial predicted category'])
        metrics['Complement']['report'] = classification_report(newsgroups_test['category'], newsgroups_test['Complement predicted category'])

        # Comput the confusion matrices:
        metrics['Multinomial']['cm'] = confusion_matrix(newsgroups_test['category'], newsgroups_test['Multinomial predicted category'], normalize='pred')
        metrics['Complement']['cm'] = confusion_matrix(newsgroups_test['category'], newsgroups_test['Complement predicted category'], normalize='pred')

        # Create labeled dataframes for the confusion matrices
        class_labels = sorted(set(newsgroups_test.category))
        metrics['Multinomial']['cm_df'] = pd.DataFrame(data=metrics['Multinomial']['cm'], columns=class_labels, index=class_labels)
        metrics['Complement']['cm_df'] = pd.DataFrame(data=metrics['Complement']['cm'], columns=class_labels, index=class_labels)

        # Initiate plots
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        fig1 = matrix_heatmap(metrics['Multinomial']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Two-step classification confusion matrix heatmap  \nMultinomial classifiers', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

        fig2 = matrix_heatmap(metrics['Complement']['cm_df'].values.tolist(), options={'x_labels': class_labels, 'y_labels': class_labels, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': True, 'vmin_vmax': (0,1), 'center': None, 'title_axis_labels': ('Two-step classification confusion matrix heatmap  \nComplement classifiers', 'Topic', 'Topic'), 'rotate x_tick_labels': True})

        # Show results and plots
        st.subheader(f'**Two-step classification results:**')
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.subheader('Multinomial classifiers')
            st.write(f"**Accuracy:** {metrics['Multinomial']['accuracy']}")
            st.write('**Classification report:**')
            st.text('.  \n'+metrics['Multinomial']['report'])
            st.write('**Confusion matrix:**')
            st.write(metrics['Multinomial']['cm_df'])
            st.pyplot(fig1)
        with output_col2:
            st.subheader('Complement classifiers')
            st.write(f"**Accuracy:** {metrics['Complement']['accuracy']}")
            st.write('**Classification report:**')
            st.text('.  \n'+metrics['Complement']['report'])
            st.write('**Confusion matrix:**')
            st.write(metrics['Complement']['cm_df'])
            st.pyplot(fig2)
    st.subheader('')


    st.write(
        '''
        The performance of the two-step classification method is slightly _worse_ than classification with a single step. The accuracies when both steps were performed be Multinomial (Complement) classifiers is 74.5% (74.2%). Adding bi-grams to the extracted features changes these to 75.4% for Multinomial classifiers and 74.1% for Complement classifiers. Comparatively, the single step results, on the other hand, were both 76.7%, and with bi-gram features added 77.7% and 78.3%.
        '''
    )

    st.header('Final conclusions')
    st.write(
        '''
        This has been a thorough walkthrough of Naive-Bayes classification of the 20 Newsgroup dataset with `sklearn`. We covered the feature extraction tools for textual data included with `sklearn` and most of their customization options. We fine-tuned the parameters affecting classification, including the proportion of data used for training, and the Laplace smoothing parameter `alpha` of the classifiers. We tried out different combinations of feature extraction parameters for `CountVectorizer` and `TfidfVectorizer`, which essentially confirmed the default settings work best for this dataset. Finally, we tried two different classification methods: the first one involved training a single classifier on all of the training data to immediately predict the category of samples. These classifiers, with the default features, had accuracies of around 75%. The second classification method attempted to take advantage of the fact that the different classes in the data were grouped into broad topics. This method introduced an intermediate step where the broad topic was first predicted with a classifier trained on the entire training set. Once the topic was predicted, a second classifier trained only on samples from the training set belonging to this topic was used to predict the individual category. We studied the performance of both steps and each had relatively high performance, with the broad topic being predicted with an accuracy around 85%, and categories within each topic predicted with accuracies ranging from 73% to almost 90%. However, combining the two steps resulted in overall performance of roughly 74%, which is essentially identical to the single step method. Sometimes trying a more clever way to approach a problem pays off in the end, but in this case, going the simple route is preferable.
        '''
    )