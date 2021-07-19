import streamlit as st

# Load example pages
from sklearn_naive_bayes_iris_example import iris_example
from sklearn_naive_bayes_20newsgroup_example import newsgroup_example

# Load helpers
from helpers import button_created, button_changed

def app_page():

    # Make title
    st.title('Naive-Bayes with sklearn')
    st.write(
        '''
        This page includes two interactive examples of analysing and classifying datasets using the tools available in `sklearn`. We will assume some prior knowledge of working with data in `pandas` DataFrames and constructing visualizations with `matplotlib` and `seaborn`. Nevertheless, the lines of code for producing these visualizations are provided. For our examples, we will use datasets already included with `sklearn` in the `sklearn.datasets` module. A list of these, with more details on each, can be found [here](https://scikit-learn.org/stable/datasets.html). Since they are included in the library, no external files need to be downloaded. To use them, `sklearn` includes convienient functions that return their respective dataset, along with some descriptive information.

        In each example, you will find code blocks
        ```python
        like this one.
        ```
        Each code block contains a hidden link in the top-right corner that appears when the mouse is hovered over it. This link allows the contents of the block to be copied to the clipboard for pasting into a code editor. Some of the code blocks are meant to illustrate the use of techniques discussed in the examples. Others are runable, in which case they appear side-by-side with an output window and a button to run the code in the block like so:
        '''
    )

    # ------------------------------
    # ----- Example code block -----
    # ------------------------------
    run_button_key = 'example_run_button'
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        st.write(
            '''
            ```python
            print('Press the "Run Code" button to print this message to the right.')

            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key=run_button_key, on_click=button_created(run_button_key))
    


    if run_button or st.session_state[run_button_key+'_dict']['was_pressed']:
        st.session_state[run_button_key+'_dict']['was_pressed'] = True
        with output_col:
            st.text('Press the "Run Code" button to print this message to the right.')
    
    st.write(
        '''
        Sometimes the code, output, or both gets rather lengthy. Luckily, Streamlit includes expandable containers to hold them like so:
        '''
    )

    # ----------------------------------------
    # ----- Expanding example code block -----
    # ----------------------------------------
    run_button_key = 'expanding_example_run_button'
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python
                print('Press the "Run Code" button to print this message below.')

                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key=run_button_key, on_click=button_created(run_button_key))
    st.subheader('Output:')
    if run_button or st.session_state[run_button_key+'_dict']['was_pressed']:
        st.session_state[run_button_key+'_dict']['was_pressed'] = True
        st.text('Press the "Run Code" button to print this message below.')
    st.subheader('')

    st.write(
        ''' 
        The output column contains a simulation of the output of the python code in the code column. It is generated with Streamlit functions, which often produce interactive versions of the plain text that would print to the screen following the execution of the code. For example, the print statement's output was reproduced with the 'text' function. If the print statement included a dataframe, the output column would display an interactive representation produced by streamlit's 'write' function. If the output is lengthy, an expanding container is used to hide the output by default. The reader can copy and paste the code to print the output to the terminal, but it won't look as nice and tidy as what appears in the output column. Finally, each code block is self-contained in the sense that the reader does not need to run them in any order to produce the output.

        The examples on this page include the **Iris** dataset, which contains fully numerical data, and the **20newsgroup** dataset, which contains fully textual data. The Iris dataset contains features that describe the physical appearance of 150 samples of three species belonging to the _iris_ genus of flowering plant. The data in this form is essentially ready for training a classifier to predict which species of iris plant the sample was taken from. The 20newsgroup dataset contains roughly twenty thousand textual samples of internet newsgroup postings. They are subdivided into twenty categories, some of which are quite similar. We use this similarity to our advantage to illustrate the limitations of the Naive-Bayes algorithm. Since this dataset consists of raw text, we will need to do some preprocessing in order to feed it into a classifier to predict categories.
        '''
    )
    
    # Create the dataset selection tool in the sidebar
    st.sidebar.subheader('Select a dataset')
    dataset = st.sidebar.selectbox('', options=['', 'Iris dataset', '20newsgroup dataset'])

    if dataset == '':
        st.write('Please select a dataset using the dropdown to the left.')
    
    elif dataset == 'Iris dataset':
        iris_example()
    
    elif dataset == '20newsgroup dataset':
        newsgroup_example()
        










  