import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import function to fetch dataset
from sklearn.datasets import load_iris

# Import Multinomial Naive-Bayes classifier
from sklearn.naive_bayes import GaussianNB

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import sklearn metrics for analysis
from sklearn.metrics import classification_report, confusion_matrix

# Import heatmap plotting function
from matrix_heatmap import matrix_heatmap

# Import custom latex display and numbering class
from latex_equation_numbering import latex_equation_numbering

# Instantiate the latex_equation_numbering class
iris_equations = latex_equation_numbering()


def iris_example():
    st.title('Classification with the Iris dataset')
    st.header('Loading the Data')
    st.write(
        '''
        To load the **Iris** dataset, we use the `load_iris()` function from the `sklearn.datasets` module:
        ```python
        from sklearn.datasets import load_iris

        iris = load_iris()
        ```

        The data features and target feature and their names are found using the `.data`, `.target`, `.feature_names`, and `.target_names` attributes. A description of the dataset can be returned using `.DESCR`. We can examine individual samples as well. Try out the following code by clicking the 'Run Code' button. The description text is lengthy, so it's placed inside an expandable container.
        '''
    )
    
    # ----------------------------------------
    # ----- load Iris dataset code block -----
    # ----------------------------------------
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        st.write(
            '''
            ```python
            from sklearn.datasets import load_iris

            iris = load_iris()

            print('Feature names:')
            print(iris.feature_names)
            
            print('Target names:')
            print(iris.target_names)

            print('Data description:')
            print(iris.DESCR)

            print('First three samples:')
            print('Feature values:')
            print(iris.data[:3])
            print('Target values:')
            print(iris.target[:3])
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='iris_load_run_button')
    if run_button:

        iris = load_iris()

        with output_col:
            st.write('**Feature names:**')
            st.text(iris.feature_names)
            
            st.write('**Target names:**')
            st.text(iris.target_names.tolist())
            
            st.write('**Data description:**')
            with st.beta_expander('Expand description'):
                st.text(iris.DESCR)
            st.text('\n  ')
            
            st.write('**First three samples:**')
            st.text('Feature values:')
            st.text(iris.data[:3].tolist())
            st.text('Target values:')
            st.text(iris.target[:3].tolist())

    st.write(
        '''
        From the feature names, we deduce that the `iris` dataset contains numerical features describing the physical characteristics of samples of the _iris_ species. From the target names, we infer three subspecies in the data. Here is an image composition of all three species from [this article](https://towardsdatascience.com/the-iris-dataset-a-little-bit-of-history-and-biology-fb4812f5a7b5). The flower petals and sepals are labeled.
        '''
    )
    col1, col2, col3 = st.beta_columns([1,3,1])
    with col2:
        st.image('./resources/images/iris_species.png')
        st.markdown(
            '''
            <div style="text-align: center;">Three species of iris in the iris dataset. (Sources: 
            <a href="https://commons.wikimedia.org/wiki/Category:Iris_setosa#/media/File:Irissetosa1.jpg" target="_blank">Left</a>
            <a href="https://en.wikipedia.org/wiki/Iris_flower_data_set#/media/File:Iris_versicolor_3.jpg" target="_blank">Center</a>
            <a href="https://en.wikipedia.org/wiki/Iris_flower_data_set#/media/File:Iris_virginica.jpg" target="_blank">Right</a>)</div>
            ''', unsafe_allow_html=True
        )

    st.header('Exploring the Data')
    st.write(
        '''
        Let's do some exploratory data analysis (EDA) for this dataset before jumping into training classifiers. To facilitate this process, we will use `pandas` to create a DataFrame of the iris dataset, and use `seaborn` for visualizations. Both packages can be imported using
        ```python
        import pandas as pd
        import seaborn as sns
        ```
        The abreviations `pd` and `sns` are conventions. By using the `as_frame=True` keyword argument in the `load_iris()` function call, `pandas` DataFrame and Series are returned for the feature data and target respectfully. Run the code to view the feature and target data, and a statistics summary of the data features.
        '''
    )

    # ------------------------------------------------------
    # ----- Iris dataset summary statistics code block -----
    # ------------------------------------------------------
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        st.write(
            '''
            ```python   
            import pandas as pd
            from sklearn.datasets import load_iris

            iris_df = load_iris(as_frame=True)

            print('iris_df.data dataframe:')
            print(iris_df.data)

            print('iris_df.target dataframe:')
            print(iris_df.target)

            print('Feature data statistics:')
            print(iris_df.data.describe())

            print('Distribution by target value:')
            print(iris_df.target.replace({0: 'setosa', 
                                          1: 'versicolor', 
                                          2: 'virginica'}).value_counts())
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='iris_stats_run_button')
    if run_button:

        iris_df = load_iris(as_frame=True)

        with output_col:
            st.write('**iris_df.data dataframe:**')
            with st.beta_expander('Expand dataframe'):
                st.write(iris_df.data)
            st.text('\n  ')

            st.write('**iris_df.target series:**')
            with st.beta_expander('Expand series'):
                st.write(iris_df.target)
            st.text('\n  ')
            
            st.write('**Feature data statistics:**')
            with st.beta_expander('Expand statistics'):
                st.write(iris_df.data.describe())
            st.text('\n  ')

            st.write('**Distribution by target value:**')
            st.write(iris_df.target.replace({0: 'setosa', 
                                             1: 'versicolor', 
                                             2: 'virginica'}).value_counts())
    
    st.write(
        '''
        From the output of the `.describe()` method on the features dataframe, we see that all of the features fall within a few centimeters of each other. The sepal length feature has the largest average value while the petal width feature the smallest. The petal length feature is the most spread out within its range (has the largest standard deviation) and the sepal width the most constrained (smallest standard deviation).

        From the output of the `.value_counts()` method on the target series (after a replacement of integer values to strings for clarity), we see that there are exactly 50 samples from each species.
        '''
    )
    
    st.subheader('One Dimensional Visualizations')
    st.write(
        '''
        The above contains textual representation of the dataset and summary statistics. We can create some visualizations to further explore the data, with the goal to determine if there is any clustering that may hint at how the different classes are differentiated. Usually the first thing we want to see is how each feature is distributed, which means looking at 1D distributions. In the iris dataset, we are fortunate that each feature is measured in the same units, with values falling in similar ranges (see the output of `.describe()` in the above runable code. Taking advantage of this, we can plot the distributions with a single vertical axis. A box or violin plot would nicely show how the data is distributed, and hint at any clustering. In a box plot, the data is divided into quartiles. The box comprises the interquartile range: the range of data that comprises the middle 50% of data samples. The median is shown with a line though the box. Outside the box are extended whiskers. These extend past the upper and lower edges of the box by 150% of the interquartile range. Outside these whiskers are outlier data points. See the following figure from [this article](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51).
        '''
    )
    ol1, col2, col3 = st.beta_columns([1,3,1])
    with col2:
        st.image('./resources/images/box_plot_diagram.png')
    st.write(
        '''
        A violin plot is very similar to a box plot, but instead of a box, whiskers, and outliers, the distribution of the data is shown as 'violin' shape with sides determined from a kernal density estimation. Wider sections indicate larger clustering of data. Inside the violin is a dark band which denotes the interquartile range. The center white dot is the median value.
        
        Select a chart type and press the run button below the code block to see the chart. 
        '''
    )
    
    # ---------------------------------------------
    # ----- 1-D distribution plots code block -----
    # ---------------------------------------------
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        plottype = st.radio('Select plot type:', options=['Box plot', 'Violin plot'], key='iris_1D_radio_button')
        if plottype == 'Box plot':
            plottype_string = 'boxplot'
        elif plottype == 'Violin plot':
            plottype_string = 'violinplot'
        st.write(
            f'''
            ```python   
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.datasets import load_iris

            iris_df = load_iris(as_frame=True)

            iris_df = load_iris(as_frame=True)
            fig, ax = plt.subplots()
            ax = sns.{plottype_string}(data=iris_df.data)
            ax.set_title('Distribution of iris features')
            ax.set_ylabel('Value (cm)')
            fig.show()
            ```
            '''
        )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='iris_1D_run_button')
    if run_button:

        iris_df = load_iris(as_frame=True)

        fig, ax = plt.subplots()
        if plottype == 'Box plot':
            ax = sns.boxplot(data=iris_df.data)
        elif plottype == 'Violin plot':
            ax = sns.violinplot(data=iris_df.data)
        ax.set_title('Distribution of iris features')
        ax.set_ylabel('Value (cm)')

        with output_col:
            st.pyplot(fig)   

    st.write(
        '''
        From the violin plot, we see some clustering in the petal length and width features. These may be good indicators of species. The sepal width feature is highly unimodal, meaning this feature is likely similar among species. The sepal length feature is elongated, meaning the data is spread out. This may mean that each species has different typical sepal lengths, or sepal lengths vary widely from sample to sample, across iris species. This means there is much overlap in these features between the different species. Since we only have three target classes, we can split the data by target label, and make the same distributions for each.
        '''
    )

    # --------------------------------------------------------------
    # ----- 1-D distribution plots split by species code block -----
    # --------------------------------------------------------------        
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        plottype = st.radio('Select plot type:', options=['Box plot', 'Violin plot'], key='iris_1D_by_species_radio_button')
        if plottype == 'Box plot':
            st.write(
                '''
                ```python   
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.datasets import load_iris

                iris_df = load_iris(as_frame=True)
                iris_df.data['species'] = iris_df.target.replace({0: 'setosa', 
                                                                  1: 'versicolor', 
                                                                  2: 'virginica'})

                df = iris_df.data.melt('species', 
                                        var_name='Iris feature',
                                        value_name='Value (cm)')

                fig, ax = plt.subplots()
                ax = sns.boxplot(data=df, x='Iris feature', y='Value (cm)', hue='species')                
                ax.set_title('Distribution of iris features by species')
                fig.show()
                ```
                '''
            )
        elif plottype == 'Violin plot':
            st.write(
                '''
                ```python   
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.datasets import load_iris

                iris_df = load_iris(as_frame=True)
                iris_df.data['species'] = iris_df.target.replace({0: 'setosa', 
                                                                  1: 'versicolor', 
                                                                  2: 'virginica'})

                df = iris_df.data.melt('species', 
                                        var_name='Iris feature',
                                        value_name='Value (cm)')

                fig, ax = plt.subplots()
                ax = sns.violinplot(data=df, x='Iris feature', y='Value (cm)', hue='species')                
                ax.set_title('Distribution of iris features by species')
                fig.show()
                ```
                '''
            )
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='iris_1D_by_species_run_button')
    if run_button:
        
        iris_df = load_iris(as_frame=True)
        iris_df.data['species'] = iris_df.target.replace({0: 'setosa', 
                                                          1: 'versicolor', 
                                                          2: 'virginica'})

        df = iris_df.data.melt('species', 
                                var_name='Iris feature', 
                                value_name='Value (cm)')

        fig, ax = plt.subplots()

        if plottype == 'Box plot':
            ax = sns.boxplot(data=df, x='Iris feature', y='Value (cm)', hue='species')
        elif plottype == 'Violin plot':
            ax = sns.violinplot(data=df, x='Iris feature', y='Value (cm)', hue='species')
        
        ax.set_title('Distribution of iris features by species')
        

        with output_col:
            st.pyplot(fig)
        
    st.write(
        '''
        From this chart, where each feature is subdivided by species, we see clear distinction between different species. The difference in the petal length and width features becomes more apparent, with the 'setosa' species is significantly separated from the other two in these features. The overlapping distributions in the petal width feature means this feature may not be vary indicitive of species. The elongated distribution of petal length, however, becomes more resolved into different distributions when split by species. The dark vertical bands indicate interquartile range, which are distinguished among species.
        '''
    )

    st.subheader('Multi-Dimensional Visualizations')
    st.write(
        '''
        Now that we've seen how the individual features are distributed, we can explore how these features are related to each other by plotting them against each other. To see all the combinations of pairs of features at once, we can use `pairplot` from the `seaborn` visualization library. This plotting style creates a 2x2 grid with size equal to the number of features. The plot in row A, column B (where A doesn't equal B) is a two-dimensional representation of the feature value in row A (on the vertical axis) plotted versus the feature value in column B (on the horizontal axis). The main diagonal contains one-dimensional visualizations of each feature. The options for the two-dimensional plots on the off-diagonal representation are a simple scatter plot, where we can use transparancy to show where many overlapping points are, a histogram, which appears as a grid with shaded cells corresponding to the number of points inside, and a kernel density estimation (KDE) with a set number of contours signifying where the KDE has a constant value. The 2-D KDE plot is similar to an elevation chart, where a peak is shown by nested curves. The options for the one-dimensional plots along the main-diagonal are regular 1-D histograms and KDEs. When the off-diagonal plotting style is a scatter plot or histogram, we can add 2-D KDEs in the lower-triangular half of the grid to guide the eye.
        
        Select the options below to change the code in the code block and press 'Run Code' to see the plot (plot will take several seconds to load). For a larger view, use the arrows in the top-right corner.  Can you identify any clustering that may hint at different species?
        '''
    )


    # ---------------------------------------------
    # ----- 2-D distribution plots code block -----
    # ---------------------------------------------
    st.subheader('Plotting options')
    code_options = st.beta_columns(3)
    with code_options[0]:
        diag_type = st.radio('1-D plotting style:', options=['Histogram', 'Kernel Density Estimation'], key='iris_2D_diag_radio_button')
    with code_options[1]:
        off_diag_type = st.radio('2-D plotting style:', options=['Scatter', 'Histogram', 'Kernel Density Estimation'], key='iris_2D_off_diag_radio_button')
    if off_diag_type != 'Kernel Density Estimation':
        with code_options[2]:
            kde_overlay = st.radio('KDE overlay:', options=['Yes', 'No'], key='iris_2D_kde_overlay_radio_button')
            
    # Define code block strings
    if diag_type == 'Histogram':
        diag_type_code = "'hist'"
    elif diag_type == 'Kernel Density Estimation':
        diag_type_code = "'kde'"

    if off_diag_type == 'Scatter':
        off_diag_type_code = "'scatter'"
        scatter_kws_string = ", plot_kws={'alpha': 0.25}"
    elif off_diag_type == 'Histogram':
        off_diag_type_code = "'hist'"
        scatter_kws_string = ''
    elif off_diag_type == 'Kernel Density Estimation':
        off_diag_type_code = "'kde'"
        scatter_kws_string = ''
        kde_overlay = 'No'
    
    if kde_overlay == 'Yes':
        kde_overlay_code = "fig.map_lower(sns.kdeplot, levels=4, color='.2')"
    elif kde_overlay == 'No':
        kde_overlay_code = ''
    
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        st.write(
            f'''
            ```python   
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.datasets import load_iris

            iris_df = load_iris(as_frame=True)
            
            fig, ax = plt.subplots()
            
            fig = sns.pairplot(data=iris_df.data, diag_kind={diag_type_code}, 
                                kind={off_diag_type_code}{scatter_kws_string})
            {kde_overlay_code}
            fig.fig.suptitle('Distribution of pairs of iris features by species', 
                                y=1.05, fontsize=16)                        
            fig.show()
            ```
            '''
        )
                 
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='iris_2D_run_button')
    if run_button:

        iris_df = load_iris(as_frame=True)

        fig, ax = plt.subplots()

        if diag_type == 'Histogram':
            if off_diag_type == 'Scatter':
                fig = sns.pairplot(data=iris_df.data, diag_kind='hist', kind='scatter', plot_kws={'alpha': 0.5})
            elif off_diag_type == 'Histogram':
                fig = sns.pairplot(data=iris_df.data, diag_kind='hist', kind='hist')
            elif off_diag_type == 'Kernel Density Estimation':
                fig = sns.pairplot(data=iris_df.data, diag_kind='hist', kind='kde')
        
        elif diag_type == 'Kernel Density Estimation':
            if off_diag_type == 'Scatter':
                fig = sns.pairplot(data=iris_df.data, diag_kind='kde', kind='scatter', plot_kws={'alpha': 0.5})
            elif off_diag_type == 'Histogram':
                fig = sns.pairplot(data=iris_df.data, diag_kind='kde', kind='hist')
            elif off_diag_type == 'Kernel Density Estimation':
                fig = sns.pairplot(data=iris_df.data, diag_kind='kde', kind='kde')

        if off_diag_type != 'Kernel Density Estimation':
            if kde_overlay == 'Yes':
                fig.map_lower(sns.kdeplot, levels=4, color='.2')
        
        fig.fig.suptitle('Distribution of pairs of iris features', y=1.05, fontsize=16)
        
        with output_col:

            st.pyplot(fig)

    
    st.write(
        '''
        There appears to be at least two clear clusters in each cell. Since we are working with labeled data, we can use different colors for each species. Here is the same plot as above, but with the target label applied.
        '''
    )

    # --------------------------------------------------------
    # ----- 2-D distribution plots by species code block -----
    # --------------------------------------------------------
    st.subheader('Plotting options')
    code_options = st.beta_columns(3)
    with code_options[0]:
        diag_type = st.radio('1-D plotting style:', options=['Histogram', 'Kernel Density Estimation'], key='iris_2D_by_species_diag_radio_button')
    with code_options[1]:
        off_diag_type = st.radio('2-D plotting style:', options=['Scatter', 'Histogram', 'Kernel Density Estimation'], key='iris_2D_by_species_off_diag_radio_button')
    if off_diag_type != 'Kernel Density Estimation':
        with code_options[2]:
            kde_overlay = st.radio('KDE overlay:', options=['Yes', 'No'], key='iris_2D_by_species_kde_overlay_radio_button')

    # Define code block strings
    if diag_type == 'Histogram':
        diag_type_code = "'hist'"
    elif diag_type == 'Kernel Density Estimation':
        diag_type_code = "'kde'"

    if off_diag_type == 'Scatter':
        off_diag_type_code = "'scatter'"
        scatter_kws_string = ", plot_kws={'alpha': 0.25}"
    elif off_diag_type == 'Histogram':
        off_diag_type_code = "'hist'"
        scatter_kws_string = ''
    elif off_diag_type == 'Kernel Density Estimation':
        off_diag_type_code = "'kde'"
        scatter_kws_string = ''
        kde_overlay = 'No'
    
    if kde_overlay == 'Yes':
        kde_overlay_code = "fig.map_lower(sns.kdeplot, levels=4, color='.2')"
    elif kde_overlay == 'No':
        kde_overlay_code = ''

    rplc_targets_string = '''iris_df.data['species'] = iris_df.target.replace({0: 'setosa', 
                                                              1: 'versicolor', 
                                                              2: 'virginica'})'''
    
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        st.write(
            f'''
            ```python   
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.datasets import load_iris

            iris_df = load_iris(as_frame=True)
            {rplc_targets_string}
            fig, ax = plt.subplots()
            
            fig = sns.pairplot(data=iris_df.data, hue='species', diag_kind={diag_type_code}, 
                                kind={off_diag_type_code}{scatter_kws_string})
            {kde_overlay_code}
            fig.fig.suptitle('Distribution of pairs of iris features by species', 
                                y=1.05, fontsize=16)                        
            fig.show()
            ```
            '''
        )
                 
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='iris_2D_by_species_run_button')
    if run_button:

        iris_df = load_iris(as_frame=True)
        iris_df.data['species'] = iris_df.target.replace({0: 'setosa', 
                                                          1: 'versicolor', 
                                                          2: 'virginica'})

        fig, ax = plt.subplots()

        if diag_type == 'Histogram':
            if off_diag_type == 'Scatter':
                fig = sns.pairplot(data=iris_df.data, hue='species', diag_kind='hist', kind='scatter', plot_kws={'alpha': 0.5})
            elif off_diag_type == 'Histogram':
                fig = sns.pairplot(data=iris_df.data, hue='species', diag_kind='hist', kind='hist')
            elif off_diag_type == 'Kernel Density Estimation':
                fig = sns.pairplot(data=iris_df.data, hue='species', diag_kind='hist', kind='kde')
        
        elif diag_type == 'Kernel Density Estimation':
            if off_diag_type == 'Scatter':
                fig = sns.pairplot(data=iris_df.data, hue='species', diag_kind='kde', kind='scatter', plot_kws={'alpha': 0.5})
            elif off_diag_type == 'Histogram':
                fig = sns.pairplot(data=iris_df.data, hue='species', diag_kind='kde', kind='hist')
            elif off_diag_type == 'Kernel Density Estimation':
                fig = sns.pairplot(data=iris_df.data, hue='species', diag_kind='kde', kind='kde')

        if off_diag_type != 'Kernel Density Estimation':
            if kde_overlay == 'Yes':
                fig.map_lower(sns.kdeplot, levels=4, color='.2')
        
        fig.fig.suptitle('Distribution of pairs of iris features by species', y=1.05, fontsize=16)
      
        with output_col:

            st.pyplot(fig)



    st.write(
        '''
        One more statistic we can look at before getting into classification is the correlation matrix. If two features are _correlated_, when one feature increases, so does the other, resulting in a correlation coeffient that is positive. The amount one feature increases relative to the increase in the other is (roughly) the correlation coefficient. If instead the second feature decreases when the first feature increases, than these two are _oppositely correlated_, and the coefficient for them is negative. If a change in one feature does not imply a change in another, than these features are _uncorrelated_ and their coefficient is zero. The correlation coeffients for each pair of features can be calculated with the `pandas` method `.corr()` called on the feature dataframe. The output is a square matrix with dimension equal to the number of features. The value in row A, column B is the correlation coeffient between features A and B. The matrix is therefore symmetric with values on the main diagonal of exactly 1. To visually compare values of a matrix with one another, a heatmap with the color of each cell scaling with the value of the matrix is a common visualization tool. Run the code below to output the correlation matrix and a heatmap of its values.
        '''
    )

    # -------------------------------------------------
    # ----- Correlation matrix heatmap code block -----
    # -------------------------------------------------
    code_col, output_col = st.beta_columns(2)
    with code_col:
        st.subheader('Code:')
        st.write(
            '''
            ```python   
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.datasets import load_iris

            iris_df = load_iris(as_frame=True)

            corr_matrix = iris_df.data.corr()

            print('Correlation matrix:')
            print(corr_matrix)

            fig, ax = plt.subplots()
            ax = sns.heatmap(corr_matrix, annot=True, fmt='0.3g', cmap='bone')
            ax.set_title('Correlation matrix heatmap')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Feature')
            fig.show()
            ```
            '''
        )
        
    with output_col:
        st.subheader('Output:')
    run_button = st.button('Run Code', key='iris_corr_heatmap_run_button')
    if run_button:
        
        iris_df = load_iris(as_frame=True)

        corr_matrix = iris_df.data.corr()

        fig, ax = plt.subplots()

        fig = matrix_heatmap(corr_matrix.values.tolist(), options={'x_labels': iris_df.feature_names,'y_labels': iris_df.feature_names, 'annotation_format': '.3g', 'color_map': 'bone', 'custom_range': True, 'vmin_vmax': (-1,1), 'center': 0, 'title_axis_labels': ('Correlation matrix heatmap', 'Feature', 'Feature'), 'rotate x_tick_labels': True})
        
        with output_col:
            st.write('**Correlation matrix:**')
            st.write(corr_matrix)
            st.pyplot(fig)

    st.write(
        '''
        From the values of the correlation matrix, and more easily from the colors in the heatmap, we see that the sepal width feature negatively correlates with all other features, whereas the other features are positively correlated with each other.

        From the visualizations in this section, we have seen that our features are good indicators for iris species, our target feature. Before we can start classification, we need to split the data to create two subsets: one that will be used for training the classifier, and the other for evaluating its performance.
        '''
    )

    

    st.header('Splitting the Data')
    st.write(
        '''
        We will make use of the the function `test_train_split()` which can be found in the `sklearn.model_selection` module. Details about this function can be found in its [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). To import it, simply use
        ```python
        from sklearn.model_selection import test_train_split
        ```

        The function call has the general form:
        ```python
        dataset_1_train, dataset_1_test, dataset_2_train, dataset_2_test, ... , dataset_N_train, dataset_N_test 
                = train_test_split(dataset_1, dataset_2, ... , dataset_N, train_size = 0.8, test_size=0.2, shuffle=True, random_state=42, stratify=dataset_j)
        ```
        Each positional arguement (the `dataset_i`'s in the function call) is a dataset to be split. The lengths of each `dataset_i` must be equal. For each one, there is a return value of `dataset_i_train` and `dataset_i_test`. The order of the arguments matches the order of the return pairs. The `train_size` and `test_size` values is the percentage of data to leave for training and testing . In the call above, each `dataset_i_train` and `dataset_i_test` will contain 80% and 20% of the data respectively. Data in the testing sets should _never_ be used in any part of the training process. The example above uses both `train_size` and `test_size` for illustration, but only one is necessary. If either is excluded, the other will be the complement of the one provided. If neither are included, the default is to use 25% of the data for testing. The `shuffle` parameter determines whether to shuffle the entries in each dataset prior to splitting. This is generally a good idea to avoid any bias in how the data was originally ordered. In our case, the iris dataset is ordered by species, and so it must be shuffled before spitting. The `random_state` parameter can be used to allow results to be replicated across multiple function calls, which is useful when one wants to tune a model without random shuffling affecting the output. Finally, `stratify` can be used with labeled data if the target feature (here `dataset_j` above) is unbalanced, meaning the distribution of classes isn't uniform. If we have `stratify=True`, then stratified sampling is used to ensure that the resulting split datasets, `dataset_i_train` and `dataset_i_test`, for each `dataset_i` will have the same proportion of classes as that in `dataset_j`. 
        
        The form of the function call above is for when all of the features and target are in different datasets. Usually, this is not the case, and the features are grouped together into one dataset `X`, and the target in another dataset `y`. In this case, one can use the simpler syntax
        ```python
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
        ```
        '''
    )

    st.header('Training and testing a classifier')
    st.write(
        '''
        Finally we are ready to start classification! First we need to decide which Naive-Bayes classifier is best for our problem. The module `sklearn.naive_bayes` includes all of the classifiers covered in the **Mathematical Background** page: `MultinomialNB`, `ComplementNB`, `BernoulliNB`, `CategoricalNB`, and `GaussianNB`. As we've seen above, the iris dataset consists of fully numerical data, and from our exploratory data analysis, each feature is roughly normally distributed in each class (determined by looking at the shape of the violins in the species-divided violin plot). This means that Gaussian Naive-Bayes is perfect for this dataset! To use this classifier, (or any of the other variations included in `sklearn.naive-bayes`), simply import it and instantiate a class instance:
        ```python
        from sklearn.naive_bayes import GaussianNB

        classifier = GaussianNB()
        ```
        
        Since the data is already in a clean and tidy form with no missing values, we can split it into training and testing sets:
        ```python
        iris_df = load_iris(as_frame=True)

        iris_features_train, iris_features_test, iris_species_train, iris_species_test 
                = train_test_split(iris_df.data, iris_df.target, test_size=0.2, shuffle=True, random_state=42)
        ```
        We don't need stratified sampling since the iris dataset is balanced (fifty samples for each species), but we do need to apply shuffling before splitting.

        To train our classifier on the training sets `iris_features_train` and `iris_species_train`, we call the `.fit()` method on `classifier`:
        ```python
        classifier.fit(iris_features_train, iris_species_train)
        ```
        After the classifier is trained, its accuracy when applied to the testing set can be found using the `.score()` method:
        ```python
        print(classifier.score(iris_features_test, iris_species_test))
        ```
        This will print the _accuracy_ of the classifier on the testing set, which is the number of times the classifier made a correct prediction divided by the total number of predictions made. To see the actual predictions determined from the testing set, we can use the `.predict()` method
        ```python
        iris_species_predict = classifier.predict(iris_species_test)
        ```
        The variable `iris_species_predict` contains a list of species predictions for each sample in the testing set. We can use `sklearn`'s built in metrics to evaluate the performance of the classifer. Useful functions in the `sklearn.metrics` module include `classification_report` and `confusion_matrix`:
        ```python
        from sklearn.metrics import classification_report, confusion_matrix
        ```
        These functions and their outputs is covered in the **Mathematical Background** page. The output of `classification_report` is a listing of several statistics that give an overview of the performance of the classifier. The output of `confusion_matrix` is a square matrix with entries corresponding to the types of predictions made (true positives, false positives, true negatives, and false negatives). The classification report can be directly printed, or outputed as a key-value dictionary for easy access to each statistic's value. The confusion matrix is best viewed as a heatmap, similar to the one made for the correlation matrix. The confusion matrix should have large values on the main diagonal and small values elsewhere. The confusion matrix is also non-negative, meaning each value is zero or greater. When the confusion matrix is unnormalized, each value in row A column B corresponds to the number of predictions of class B for samples with a ground truth of class A. The matrix can be row (column) normalized, in which each element is divided by the sum of the elements in each row (column), or population normalized, where each element is divided by the number of predictions made. When the matrix is row normalized, the main diagonal entries correspond to the recall value for each class. When the matrix is column normalized, the main diagonal contains the precision of each class. When the matrix is normalized by population, the sum of the diagonal terms correspond to the model accuracy. In any case, a well-performing model has large values on the main diagonal and small values elsewhere.

        Run the code block to split the iris dataset, train a Gaussian Naive-Bayes classifier on the training portion, and print the model accuracy, classification report, and unnormalized confusion matrix heatmap. 
        '''
    )
    
    # ------------------------------------------
    # ----- Classifier training code block -----
    # ------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python   
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.datasets import load_iris

                # Load the data
                iris_df = load_iris(as_frame=True)
                iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

                # Split the data
                iris_features_train, iris_features_test, iris_species_train, iris_species_test = train_test_split(iris_df.data, iris_df.target, train_size=0.6, shuffle=True, random_state=42)

                # Instantiate a classifier
                classifier = GaussianNB()
                
                # Train the classifier
                classifier.fit(iris_features_train, iris_species_train)

                # Compute the classification score
                print(f'Classifier accuracy: {classifier.score(iris_features_test, iris_species_test)}')

                # Compute predictions for the testing data
                iris_species_predict = classifier.predict(iris_features_test)

                print('Classification report:')
                print(classification_report(iris_species_predict, iris_species_test))

                # Create confusion matrix DataFrame
                cm_df = pd.DataFrame(data=confusion_matrix(iris_species_predict, iris_species_test), columns=iris_df.target_names, index=iris_df.target_names)

                print('Confusion matrix:')
                print(cm_df)

                # Create a heatmap of the confusion matrix
                fig, ax = plt.subplots()
                ax = sns.heatmap(cm_df.values.tolist(), annot=True, fmt='0.3g', cmap='bone')
                ax.set_title('Confusion matrix heatmap')
                ax.set_xlabel('Species')
                ax.set_ylabel('Species')
                fig.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='iris_training_run_button')
    st.subheader('Output:')
    output_col1, output_col2 = st.beta_columns(2)
    if run_button:   
        iris_df = load_iris(as_frame=True)
        iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

        iris_features_train, iris_features_test, iris_species_train, iris_species_test \
            = train_test_split(iris_df.data, iris_df.target, train_size=0.6, shuffle=True, random_state=42)
        
        classifier = GaussianNB()

        classifier.fit(iris_features_train, iris_species_train)

        iris_species_predict = classifier.predict(iris_features_test)

        # Create confusion matrix DataFrame
        cm_df = pd.DataFrame(data=confusion_matrix(iris_species_predict, iris_species_test), columns=iris_df.target_names, index=iris_df.target_names)

        # Make a heatmap of the confusion matrix
        fig, ax = plt.subplots()
        fig = matrix_heatmap(cm_df.values.tolist(), options={'x_labels': iris_df.target_names,'y_labels': iris_df.target_names, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': False, 'vmin_vmax': (-1,1), 'center': None, 'title_axis_labels': ('Confusion matrix heatmap', 'Species', 'Species'), 'rotate x_tick_labels': True})
   
        with output_col1:
            st.write(f'**Classifier accuracy:** {classifier.score(iris_features_test, iris_species_test)}')

            st.write('**Classification report:**')
            st.text('.  \n'+classification_report(iris_species_predict, iris_species_test))

            st.write('**Confusion matrix:**')
            st.write(cm_df)

        with output_col2:              
            st.pyplot(fig)
    st.subheader('')

    st.header('Tuning the model')
    st.write(
        '''
        The last step of training a machine learning algorithm is usually tuning the model to achieve the highest level of performance. In this example, the only really changeable aspect of the process is the size of the testing dataset. This is changed by either the `train_size` or the `test_size` keyword argument of `train_test_split`. Since the values of these arguments are correlated, there is only one free parameter. When tuning hyperparameters, one must have a metric in mind to compare. For this example, we don't have any real-life high costs of type-I or type-II errors, so it is sufficient to simply maximize the model's accuracy.
        
        Choose the proportion of data to train the classifier on, and the normalization type for the confusion matrix. Then run the code to see the performance.
        '''
    )

    # -------------------------------------------------
    # ----- Custom classifier training code block -----
    # -------------------------------------------------
    code_options = st.beta_columns(2)
    with code_options[0]:
        st.subheader('Proportion of training data:')
        train_size_choice = st.slider('', min_value=0.05, max_value=0.95, step=0.05, value=0.7, key='train_size_slider')
    with code_options[1]:
        st.subheader('Confusion matrix normalization style:')
        norm_choice = st.radio('', options=['None', 'Row', 'Column', 'Population'], key='norm_style_radio_button')
    
    # Define code block strings
    if norm_choice == 'None':
        norm_choice_code = None
        title_string = "'Confusion matrix heatmap: unnormalized'"
        title = 'Confusion matrix heatmap: unnormalized'
    elif norm_choice == 'Row':
        norm_choice_code = "'true'"
        title_string = "'Confusion matrix heatmap: row normalized'"
        title = 'Confusion matrix heatmap: row normalized'
    elif norm_choice == 'Column':
        norm_choice_code = "'pred'"
        title_string = "'Confusion matrix heatmap: column normalized'"
        title = 'Confusion matrix heatmap: column normalized'
    elif norm_choice == 'Population':
        norm_choice_code = "'all'"
        title_string = "'Confusion matrix heatmap: population normalized'"
        title = 'Confusion matrix heatmap: population normalized'
    
    rplc_targets_string = "{0: 'setosa', 1: 'versicolor', 2: 'virginica'}"
    accuracy_string = "{classifier.score(iris_features_test, iris_species_test)}"

    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                f'''
                ```python   
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.datasets import load_iris

                iris_df = load_iris(as_frame=True)
                iris_df.target.replace({rplc_targets_string}, inplace=True)

                iris_features_train, iris_features_test, iris_species_train, iris_species_test = train_test_split(iris_df.data, iris_df.target, train_size={train_size_choice}, shuffle=True, random_state=42)

                # Instantiate a classifier
                classifier = GaussianNB()

                # Train the classifier
                classifier.fit(iris_features_train, iris_species_train)

                # Compute the classification score
                print(f'Training size proportion: {train_size_choice}')
                print(f'Classifier accuracy: {accuracy_string}')

                # Compute predictions for the testing data
                iris_species_predict = classifier.predict(iris_features_test)

                print('Classification report:')
                print(classification_report(iris_species_predict, iris_species_test))

                # Create confusion matrix DataFrame
                cm_df = pd.DataFrame(data=confusion_matrix(iris_species_predict, iris_species_test, normalize={norm_choice_code}), columns=iris_df.target_names, index=iris_df.target_names)

                print('Confusion matrix:')
                print(cm_df)

                # Create a heatmap of the confusion matrix
                fig, ax = plt.subplots()
                ax = sns.heatmap(cm_df.values.tolist(), annot=True, fmt='0.3g', cmap='bone')
                ax.set_title({title_string})
                ax.set_xlabel('Species')
                ax.set_ylabel('Species')
                fig.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='iris_custom_training_run_button')
    st.subheader('Output:')
    output_col1, output_col2 = st.beta_columns(2)
    if run_button:

        iris_df = load_iris(as_frame=True)
        iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

        iris_features_train, iris_features_test, iris_species_train, iris_species_test \
            = train_test_split(iris_df.data, iris_df.target, train_size=train_size_choice, shuffle=True, random_state=42)
        
        classifier = GaussianNB()

        classifier.fit(iris_features_train, iris_species_train)

        iris_species_predict = classifier.predict(iris_features_test)

        if norm_choice == 'None':
            cm = confusion_matrix(iris_species_predict, iris_species_test, normalize=None)
        elif norm_choice == 'Row':
            cm = confusion_matrix(iris_species_predict, iris_species_test, normalize='true')
        elif norm_choice == 'Column':
            cm = confusion_matrix(iris_species_predict, iris_species_test, normalize='pred')
        elif norm_choice == 'Population':
            cm = confusion_matrix(iris_species_predict, iris_species_test, normalize='all')

        cm_df = pd.DataFrame(data=cm, columns=iris_df.target_names, index=iris_df.target_names)

        # Make a heatmap of the confusion matrix
        fig, ax = plt.subplots()
        fig = matrix_heatmap(cm_df.values.tolist(), options={'x_labels': iris_df.target_names,'y_labels': iris_df.target_names, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': False, 'vmin_vmax': (-1,1), 'center': None, 'title_axis_labels': (title, 'Species', 'Species'), 'rotate x_tick_labels': True})
        
        with output_col1:
            st.write(f'**Training size proportion:** {train_size_choice}')

            st.write(f'**Classifier accuracy:** {classifier.score(iris_features_test, iris_species_test)}')

            st.write('**Classification report:**')
            st.text('.  \n'+classification_report(iris_species_predict, iris_species_test))

            st.write('**Confusion matrix:**')
            st.write(cm_df)

        with output_col2:
            st.pyplot(fig)
    st.subheader('')

    st.write(
        '''
        We can perform a grid search in the space of tunable parameters to automatically find the optimum values to maximize the model's accuracy, within our grid's resolution. Since there is only one parameter to tune, the parameter space is one dimensional. The code below will fit a classifier for training sizes between 5% and 95% of the available data in step sizes of 1%. For each value, the accuracy will be found with the `.score()` method. At the end, the maximum training proportion will be selected and the model at this point in parameter space will be shown.
        '''
    )

    # -------------------------------------------------
    # ----- Hyperparameter grid search code block -----
    # -------------------------------------------------
    st.subheader('Code:')
    code_col, button_col = st.beta_columns([10,1])
    with code_col:
        code_expander = st.beta_expander('Expand code')
        with code_expander:
            st.write(
                '''
                ```python   
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.datasets import load_iris

                iris_df = load_iris(as_frame=True)
                iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

                training_sizes = [round(0.05 + x/100, 3) for x in range(0, 91)]
                accuracies = []
                training_samples = []
                testing_samples = []

                # Loop over training sizes
                for training_size in training_sizes:
                    iris_features_train, iris_features_test, iris_species_train, iris_species_test 
                        = train_test_split(iris_df.data, iris_df.target, train_size=training_size, shuffle=True, random_state=42)

                    # Instantiate and fit a classifier
                    classifier = GaussianNB().fit(iris_features_train, iris_species_train)

                    # Calculate the accuracy score and append to the lists
                    accuracies.append(classifier.score(iris_features_test, iris_species_test))
                    training_samples.append(len(iris_features_train))
                    testing_samples.append(len(iris_features_test))

                # Find training sizes with maximum accuracy
                max_accuracy = max(accuracies)
                max_accuracy_training_sizes = [size for size, value in zip(training_sizes, accuracies) if value == max_accuracy]
                accuracy_df = pd.DataFrame(data={'train_size': training_sizes, 
                                                'accuracy': accuracies, 
                                                'training samples': training_samples, 
                                                'testing_samples': testing_samples})

                print('Accuracy vs. train_size results:')
                print(accuracy_df)

                # Make a line plot of accuracy vs train_size
                fig1, ax1 = plt.subplots()
                sns.lineplot(data=accuracy_df, x='train_size', y='accuracy', palette=['blue'], ax=ax1)
                ax1.plot(max_accuracy_training_sizes[0][0], max_accuracy_training_sizes[0][1], 'y*', markersize=10)
                ax1.set_title('Classifier accuracy versus training size')
                ax1.legend(labels=['Accuracy', 'Ideal training size'], loc=(0.6, 0.10))

                # Use minimum training size for maximum accuracy to train classifier
                iris_features_train, iris_features_test, iris_species_train, iris_species_test 
                    = train_test_split(iris_df.data, iris_df.target, train_size=max_accuracy_training_sizes[0], shuffle=True, random_state=42)
                
                # Instantiate and fit a classifier
                classifier = GaussianNB().fit(iris_features_train, iris_species_train)

                # Compute predictions for the testing data
                iris_species_predict = classifier.predict(iris_features_test)
                
                # Calculate score and classification report
                print(f'Minimum train_size for maximum accuracy: {max_accuracy_training_sizes[0]}')
                print(f'Classifier accuracy: {classifier.score(iris_features_test, iris_species_test)}')
                print('Classification report:')
                print(classification_report(iris_species_predict, iris_species_test))

                # Create confusion matrix DataFrame
                cm_df = pd.DataFrame(data=confusion_matrix(iris_species_predict, iris_species_test, normalize=None), columns=iris_df.target_names, index=iris_df.target_names)

                print('Confusion matrix:')
                print(cm_df)

                # Create a heatmap of the confusion matrix
                fig2, ax2 = plt.subplots()
                ax2 = sns.heatmap(cm.values.tolist(), annot=True, fmt='0.3g', cmap='bone')
                ax2.set_title(f'Confusion matrix heatmap: unnormalized, Training size: {max_accuracy_training_sizes[0]})
                ax2.set_xlabel('Species')
                ax2.set_ylabel('Species')
                fig2.show()
                ```
                '''
            )
    with button_col:
        run_button = st.button('Run Code', key='iris_hyperparameter_tuning_run_button')
    st.subheader('Output:')
    if run_button:
        
        iris_df = load_iris(as_frame=True)
        iris_df.target.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)

        training_sizes = [round(0.05 + x/100, 3) for x in range(0, 91)]
        accuracies = []
        training_samples = []
        testing_samples = []

        # Loop over training sizes
        for training_size in training_sizes:
            iris_features_train, iris_features_test, iris_species_train, iris_species_test \
                = train_test_split(iris_df.data, iris_df.target, train_size=training_size, shuffle=True, random_state=42)
        
            classifier = GaussianNB()

            classifier.fit(iris_features_train, iris_species_train)

            accuracies.append(classifier.score(iris_features_test, iris_species_test))
            training_samples.append(len(iris_features_train))
            testing_samples.append(len(iris_features_test))
        
        # Find training sizes with maximum accuracy
        max_accuracy = max([accuracy for accuracy in accuracies if accuracy < 1])
        max_accuracy_training_sizes = [(size, value) for size, value in zip(training_sizes, accuracies) if value == max_accuracy]
        accuracy_df = pd.DataFrame(data={'train_size': training_sizes, 
                                         'accuracy': accuracies, 
                                         'training samples': training_samples, 
                                         'testing samples': testing_samples})

        # Make a line plot of accuracy vs train_size
        fig1, ax1 = plt.subplots()
        sns.lineplot(data=accuracy_df, x='train_size', y='accuracy', palette=['blue'], ax=ax1)
        ax1.plot(max_accuracy_training_sizes[0][0], max_accuracy_training_sizes[0][1], 'y*', markersize=10)
        ax1.set_title('Classifier accuracy versus training size')
        ax1.legend(labels=['Accuracy', 'Ideal training size'], loc=(0.6, 0.10))

        # Use minimum training size for maximum accuracy to train classifier
        iris_features_train, iris_features_test, iris_species_train, iris_species_test \
            = train_test_split(iris_df.data, iris_df.target, train_size=max_accuracy_training_sizes[0][0], shuffle=True, random_state=42)
        
        classifier = GaussianNB()

        classifier.fit(iris_features_train, iris_species_train)

        iris_species_predict = classifier.predict(iris_features_test)

        cm = confusion_matrix(iris_species_predict, iris_species_test, normalize=None)

        cm_df = pd.DataFrame(data=cm, columns=iris_df.target_names, index=iris_df.target_names)

        
        # Make a heatmap of the confusion matrix
        fig2, ax2 = plt.subplots()
        fig2 = matrix_heatmap(cm_df.values.tolist(), options={'x_labels': iris_df.target_names,'y_labels': iris_df.target_names, 'annotation_format': '.3g', 'color_map': 'bone_r', 'custom_range': False, 'vmin_vmax': (-1,1), 'center': None, 'title_axis_labels': (f'Confusion matrix heatmap: unnormalized\n  Training size: {max_accuracy_training_sizes[0][0]}', 'Species', 'Species'), 'rotate x_tick_labels': True})
        
        st.header('Loop over `train_size` values:')
        output_col1, output_col2 = st.beta_columns(2)
        with output_col1:
            st.write('**Accuracy vs. train_size results:**')
            st.write(accuracy_df)
        
        with output_col2:
            st.pyplot(fig1)

        st.header('Classification results at ideal training size:')
        output_col3, output_col4 = st.beta_columns(2)
        with output_col3:
            st.write(f'**Minimum** `train_size` **for maximum accuracy:** {max_accuracy_training_sizes[0][0]}')

            st.write(f'**Classifier accuracy:** {classifier.score(iris_features_test, iris_species_test)}')

            st.write('**Classification report:**')
            st.text('.\n  '+classification_report(iris_species_predict, iris_species_test))

            st.write('**Confusion matrix:**')
            st.write(cm_df)
        
        with output_col4:
            st.pyplot(fig2)
    st.subheader('')

    st.header('Conclusions and outlook')
    st.write(
        '''
        In this guided walkthough of Naive-Bayes classification of the Iris dataset, we demonstrated a typical workflow for beginning from loading the data, all the way through to tuning model parameters for optimal classification performance. We covered many types of Exploratory Data Analysis including examining feature clustering and correlation, with suitable visualizations for each. We covered tuning the free parameters of the model for optimal accuracy.

        Looking at the values of our model accuracy vs. training size dataframe and plot, we see that even with a training size of about 15% yields an accuracy over 90%! Our ideal training size was chosen as the _least_ value whose accuracy was a maximum, while still being under 100% to stay realistic. The value turned out to be 44%, which yielded an amazing 98.8% classification accuracy with a Gaussian Naive-Bayes classifier. The Iris dataset features are highly clustered and separable so that they are very indicative of species. This presents an ideal situation for our classifier and we do not expect such high performance in less ideal settings.
        '''
    )
