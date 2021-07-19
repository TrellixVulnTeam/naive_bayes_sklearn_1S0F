import streamlit as st

# Import custom latex display and numbering class
from latex_equation_numbering import latex_equation_numbering

def app_page():

    # Instantiate the latex_equation_numbering class
    math_equations = latex_equation_numbering()

    # Make title
    st.title('The Mathematics of Naive-Bayes Classification')

    # ---------------------------------------------
    # ----- Introduction to probability ideas -----
    # ---------------------------------------------
    st.write(
        '''
        Naive-Bayes classification is a simple, yet powerful technique for assigning labels or classes to data. Since its use requires the data to be labeled, or categorized, it is an example of a _supervised_ machine learning algorithm. It works by utilizing a powerful result from probability: Bayes Theorem. Bayes Theorem is a statement about the equivalence of ways of calculating probabilities of two or more events happening together. The statement can be derived as follows. 
        '''
    )
    st.header('The Bayes-ics')
    st.write(
        '''
        Consider two events $A$ and $B$. We can write the _probability_ of event $A$ occuring as $P(A)$ and likewise for event $B$. Since $P(A)$ is a probability, it must fall in the range $0 \leq P(A) \leq 1$, and further, the sum of $P(A)$ and $P(\lnot A)$ (the probability of $A$ _not_ happening) must be unity: $P(A) + P(\lnot A) = 1$, i.e. you can state with 100% certainty that either $A$ has happened or $A$ has not happened. Now consider another event, $B$, that in some way _affects_ the chances of event $A$ happening. How would you leverage you knowledge of whether event $B$ happened to calculate how likely event $A$ will happen?
        '''
    )
    st.subheader('Conditional probability')
    st.write(
        r'''
        Before getting to Bayes Theorem, we must introduce the concept of _conditional probability_: i.e. probability that is _conditional_ on the knowledge of prior events having or having not already occured. We can denote 'the probability of event $A$ happening _given_ event $B$ has happened' by $P(A|B)$, which one typically reads as 'the probability of $A$ given $B$'. This probability is also refered to as a _posterior probability_ because it uses prior knowledge of event $B$ to inform knowledge of event $A$. The word 'posterior' comes from the Latin _a posteriori_ which, in its adverb usage, means 'in a way based on reasoning from known facts or past events rather than by making assumptions or predictions.'

        Now suppose you wish to know the probability of two events occuring, denoted as $P(A \cap B)$ (the symbol $\cap$ is the _intersection_ of two things, visually, one can think of the overlaping region in a Venn diagram). You first observe that event $A$ has happened, with a probability of $P(A)$. Then for event $B$ to happen, you would multiply by the _conditional probability_ of $B$ happening _given_ the prior knowledge that $B$ has happened: $P(B|A)$. Thus you arrive at the expression 
        $$
        P(A \cap B) = P(B|A)P(A)\;.
        $$

        Now wait a minute. What if you had instead first observed that event $B$ has happened? The you would calculate the quantity $P(B \cap A)$ the same way as above using different conditional probabilities. You would first take the probability that event $B$ has occured, $P(B)$, and multiply with the probability that event $A$ will occur, _given_ $B$ has already occured, or $P(A|B)$. Then you would have the expression
        $$
        P(B \cap A) = P(A|B)P(B) \;.
        $$
        Can we reconcile these two expressions? Of course! If you have studied set theory, or have drawn a Venn diagram or two, you know that the overlapping region is always the same whether set $A$ is on the left and $B$ on the right or vice-versa. Mathematically, this is simply the statement that $P(A \cap B) = P(B \cap A)$, or in words, the probability of $A$ and $B$ happening is the same as $B$ and $A$ happening. This is the key that links the two expressions.
        '''
    )
    st.subheader('Bayes Theorem')
    st.write(
        '''
        Since we have the result that the two expressions are equal, we can equate their right-hand-sides respectivly, and arrive at 
        $$
        P(A|B)P(B) = P(B|A)P(A) \;.
        $$
        Now, if we further assume that either events $A$ and $B$ can actually occur, meaning that one or both of $P(A)$ and $P(B)$ are non-zero, then we can divide by that factor to arrive at Bayes Theorem:
        '''
    )
    math_equations.add_equation('bayes_theorem_1', r'P(A|B) = \frac{P(B|A)P(A)}{P(B)} \;,', numbered=True)
    math_equations.display_equation('bayes_theorem_1')
    st.write(
        f'''
        with the analagous expression for $P(B|A)$ found by dividing by $P(A)$ instead. 
        
        The form of Bayes Theorem in Eq. ({math_equations.eqref('bayes_theorem_1')}) is good and well, but what happens if we don't know the probability of one of the events happening at all, say $P(B)$? Are you out of luck? Well no, because we can rephrase the probability of $B$ happening using conditional probabilities based on whether event $A$ has or has not occured. In words, the probability of $B$ occuring is the probability of $B$ _given_ $A$ having occured, times the probability of $A$ happening in the first place, in addition to the probability of $B$ occuring given $A$ _not_ occuring, times the probability of $A$ _not_ happening at all. Using our notations for conditional probabilites, this is expressed with the equation
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('prob_B', r'P(B) = P(B|A)P(A) + P(B|\lnot A)P(\lnot A) \;.', numbered=True)
    math_equations.display_equation('prob_B')
    st.write(
        '''
        Inserting this into our first expression for Bayes Thoerem gives us another way to calculate $P(A|B)$:
        '''
    )
    math_equations.add_equation('bayes_theorem_2', r'P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|\lnot A)P(\lnot A)} \;.', numbered=True)
    math_equations.display_equation('bayes_theorem_2')
    st.write(
        f'''
        Whether we use our original expression in Eq. ({math_equations.eqref('bayes_theorem_1')}) or this result depends on what information we know.

        Now the question is, how does one take Bayes Theorem, and turn it into an algorithm for classifying data? Let's looks a common use-case for Bayes classifiers.
        ''', unsafe_allow_html=True
    )

    # -------------------------------------
    # ----- Spam email filter example -----
    # -------------------------------------
    st.header('Binary classification with Bayes: filtering spam emails')
    st.write(
        '''
        To describe how one employs Bayes Theorem for classification, we will use a familiar example, the spam filter in your email. How does this filter decide whether an email is either spam or not spam? Well we can leverage our knowledge of spam emails to arrive at a simple method for filtering them.

        Let's start with a well known example of a spam email: a plea from a Nigerian Prince to assist in transfering their inheritance outside their country. Since there are many variations of these so-called [419 scams](https://www.psychologytoday.com/us/blog/out-the-ooze/201808/why-we-still-fall-the-nigerian-prince-scam), for the sake of illustration, let's pare it down to filtering emails based solely on whether the word 'inheritance' appears. Let's say you know the following three estimates (based on some [2021 spam statistics](https://dataprot.net/statistics/spam-statistics/#:~:text=Nearly%2085%25%20of%20all%20emails%20are%20spam.,-(Spamlaws)&text=That%20translates%20into%20an%20average,show%20that%20it's%20currently%20declining.)):
         
         1. A whopping 85% of emails are spam
         2. Roughly 25% of spam concerns financial matters, so lets say a fifth of these contain the word 'inheritance'
         3. 'inheritance' appears in maybe 0.1% of non-spam emails (Alright, I made this one up but unless you happen to be an attorney practicing estate law, this low rate is reasonable, and is probably quite conservative)

        Given these numbers, just how efficient would a simple filter on the word 'inheritance' actually be?
        '''
        +
        r'''
        We can identify event $A$ as an email being spam, and event $B$ being an email contains the word 'inheritance'. We want to therefore calculate $P(A|B) = P(\text{spam}|\text{inheritence})$, i.e. the probability of an email being spam given that it contains the word 'inheritance', using Bayes Theorem. To do so, we can identify three other probabilities from our three estimates above:

         1. $P(A) = P(\text{spam}) = 0.85$
         2. $P(B|A) = P(\text{inheritance}|\text{spam}) = 0.05$
         3. $P(B| \lnot A) = P(\text{inheritance}|\lnot \text{spam}) = 0.001$

        To use Bayes Theorem, we only need one more probability, $P(\lnot A) = P(\lnot \text{ spam})$. Since an email is either spam or not spam, we have $P(\lnot \text{ spam}) = 1 - P(\text{spam}) = 0.15$. Notice how we make use of our second formulation of Bayes Theorem. In this case, we don't know $P(B) = P(\text{inheritance})$. In fact, it would be quite difficult to estimate too. How often does the word 'inheritance' appear in emails, spam and not spam alike? Unless you could hack into Gmail and scan through millions of emails to get a decent sample size to calculate it directly, the error in this probability would be substantial. Thankfully we can infer it using $P(B) = P(B|A)P(A) + P(B|\lnot A)P(\lnot A)$:
        $$
        P(\text{inheritance}) = P(\text{inheritance}|\text{spam})P(\text{spam}) + P(\text{inheritance}|\lnot \text{ spam})P(\lnot\text{ spam}) \;,
        $$
        $$
        = 0.05 \times 0.85 + 0.001 \times 0.15 \;,
        $$
        $$
        = 0.04265 \;.
        $$
        Thus finally using Bayes Theorem, we have the desired conditional probability:
        $$
        P(\text{spam}|\text{inheritence}) = \frac{P(\text{inheritance}|\text{spam})P(\text{spam})}{P(\text{inheritance})} \;,
        $$
        $$
        = \frac{0.05 \times 0.85}{0.04265} \;,
        $$
        $$
        = 0.9965 \;.
        $$
        '''
        +
        '''
        Thus we have determined that an email containing the word 'inheritance' can be labeled as the class 'spam' with a probability of 0.9965, and since there are only two outcomes, as the class 'not spam' with a probability of 1-0.9965 = 0.0035.

        **_Amazing!_** Using our prior knowledge about spam and how often the word 'inheritance' appears in spam and non-spam emails, we can create a simple filter that classifies any email with this word as spam, we would be correct 99.65% of the time!
        '''
    )
    st.subheader('A more robust filter')
    st.write(
        r'''
        You may be wondering, "A filter that is 99.65% correct, what's the catch?" The catch is that a spam filter than only filters by one word is pretty limited. Since we estimated that only 5% of spam emails contain the word 'inheritance', our filter would be useless 95% of the time! How do we go about making our filter more robust? Wouldn't it be great if we could take _each_ word in a given email and use how often that word apears in spam emails to come to an informed conclusion on the whole email? To achieve this, we would need a large dataset of sample emails labeled 'spam' or 'not spam' and find how often each word in our suspect email appears in our database as 'spam' or 'not spam'.

        Let's consider we have a suspect email, and we want to find $P(\text{spam}|\textbf{email})$, i.e. the probability that given an specific email, it is spam. I've put the word 'email' in bold, because it represents a _specific_ email in the entire space of possible emails. For now, we can think of it as a list of words: $\textbf{email} = [\text{word}_1, \text{word}_2, \text{word}_3, \ldots , \text{word}_N]$, where $N$ is the number of words in our specific email. Using Bayes Theorem, we can express our desired probability as
        '''
    )
    math_equations.add_equation('spam_filter_bayes_1-spam', r'P(\text{spam}|\textbf{email}) = \frac{P(\textbf{email}|\text{spam})P(\text{spam})}{P(\textbf{email})} \;.', numbered=True)
    math_equations.display_equation('spam_filter_bayes_1-spam')
    st.write(
        r'''
        Let's break this apart. If we didn't already have $P(\text{spam}) = 0.85$ from before, we could easily estimate it from our dataset using **_maximum likelihood estimation (MLE)_**. This amounts to estimating the probability of receiving 'spam' and 'not spam' based on their frequencies in our dataset:
        '''
    )
    math_equations.add_equation('prob_spam', r"P(\text{spam}) = \frac{\text{number emails labeled `spam' in dataset}}{\text{number emails in dataset}}")
    math_equations.display_equation('prob_spam')
    math_equations.add_equation('prob_not-spam', r"P(\lnot\text{ spam}) = \frac{\text{number emails labeled `not spam' in dataset}}{\text{number emails in dataset}}")
    math_equations.display_equation('prob_not-spam')
    st.write(
        r'''
        Note the above definitions imply one very important caveat to our estimates of the probabilities of receiving spam emails. The probabilities calculated above are based solely on the distribution of the labeled data in our dataset. If half of the emails are labeled spam, then we will have $P(\text{spam}) = P(\lnot\text{ spam}) = 0.5$, quite different from the real-world finding that $P(\text{spam}) \sim 0.85$! This means that it is important to use a dataset that reflects the real world, if one wants to apply this algorithm outside of predicting classes within the dataset. Of course, one can always specify the actual priors of each class themselves, instead of calcualting them via the expressions above.

        We now need $P(\textbf{email}|\text{spam})$, which in words means, given an email is spam, what is the probability that the email is this _specific_ email. Finally, $P(\textbf{email})$ is simply the probability of receiving this _specific_ email at all. **_How does one calculate such probabilities?_** To do so, we need to make one **big** assumption.
        '''
    )
    st.subheader('What\'s naive about Naive-Bayes?')
    st.write(
        r'''
        To simplify calculating the probabilities of receiving _specific_ emails, we need to make the assumption of **_feature independence_**. That is, the appearance or not of one feature in our data does not affect the appearance or not of any other featre. This assumption is why the entire classification algorithm is refered to as _Naive_-Bayes classification. We are naively assuming all the features that make up the data, in this case all the words in the email, are uncorrelated. Of course this is not true. If we received a spam email containing the word 'inheritance', we can assume the word 'money' will likely also show up too. However, this assumption in practice doesn't hurt the classification algorithm. In fact, it is this very simplification that makes this algorithm so powerful. 

        Armed with our assumption of independence of data features, we can express the conditional probability $P(\textbf{email}|\text{spam})$ as a product of probabilities:
        $$
        P(\textbf{email}|\text{spam}) = P([\text{word}_1, \text{word}_2, \text{word}_3, \ldots , \text{word}_N]|\text{spam}) \;,
        $$
        $$
        = P(\text{word}_1|\text{spam}) \times P(\text{word}_2|\text{spam}) \times P(\text{word}_3|\text{spam}) \times \ldots \times P(\text{word}_N|\text{spam}) \;,
        $$
        $$
         = \prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\text{spam}) \;.
        $$
        '''
        +
        r'''
        The notation $\prod_{i \;\in\; \{1,\; \ldots \;, N\}}$ in this example means include _only_ values of $i$ in $1,\; \ldots \;, N$ where the feature, in this case $\text{word}_i$, appears in the sample $\textbf{email}$. Since for now we are only considering a singular email, the distinction isn't necessary, but becomes important later when we want to classify many samples, for which not every feature will be present in every sample.'''+f''' Meanwhile, the denominator of Eq. ({math_equations.eqref('spam_filter_bayes_1-spam')})'''+r''', $P(\textbf{email})$, can be expressed a sum of conditional probabilities (recall $P(B) = P(B|A)P(A) + P(B|\lnot A)P(\lnot A)$) each of which can expressed as products:
        $$
        P(\textbf{email}) = P(\textbf{email}|\text{spam})P(\text{spam}) + P(\textbf{email}|\lnot\text{ spam})P(\lnot\text{ spam}) \;,
        $$
        $$
        = P(\text{spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\text{spam}) + P(\lnot\text{ spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\lnot\text{ spam}) \;.
        $$
        Thus ultimately we can write our desired probability as
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('spam_filter_bayes_2-spam', r'P(\text{spam}|\textbf{email}) = \frac{P(\text{spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\text{spam})}{P(\text{spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\text{spam}) + \prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\lnot\text{ spam})P(\text{word}_i|\lnot\text{ spam})} \;.', numbered=True)
    math_equations.display_equation('spam_filter_bayes_2-spam')
    st.write(
        r'''
        Let us reflect on what we have achieved with our assumption of feature independence. On the left side of this equation, we have $P(\text{spam}|\textbf{email})$, the probability that, given a _specific_ email, how likely is it spam? On the left hand side, we have a somewhat complicated expression, but are saved by the fact that the only dependence on the original email are the specific words, or features, it contains! All of the complexity of language, sentence structure, phrasing, is distilled and simplified to calculating probabilities of individual words.


        Thus we have broken down our difficult task of calculating the probability of our _specific_ email being spam ($P(\text{spam}|\textbf{email})$) into finding, for each word, the probability of that word appearing in a spam email or not ($P(\text{word}_i|\text{spam})$ and $P(\text{word}_i|\lnot\text{ spam})$). To find these, we examine a large set of emails labeled 'spam' and 'not spam' and simply count how often individual words appear. Then we calculate the probabilites as
        '''
    )
    math_equations.add_equation('prob_word_in_spam', r"P(\text{word}_i|\text{spam}) = \frac{\text{number of times `}\text{word}_i\text{' appears in all emails labeled `spam'}}{\text{number of words in all emails labeled `spam'}} \;.")
    math_equations.display_equation('prob_word_in_spam')
    math_equations.add_equation('prob_word_in_not_spam', r"P(\text{word}_i|\lnot\text{ spam}) = \frac{\text{number of times `}\text{word}_i\text{' appears in all emails labeled `not spam'}}{\text{number of words in all emails labeled `not spam'}} \;.")
    math_equations.display_equation('prob_word_in_not_spam')
    st.write(
        r'''
        Once all of the probabilities are calculated for each unique $`\text{word}_i$' in the entire dataset, they can be stored in an appropriate data structure for easy reference. This task is referred to as _training_ the algorithm and the large set of labeled emails is the _training_ set. To make a classification on a new email, we simply pull out the probabilities for each word in the new email and multiply them together in the mannor of '''+ f'''Eq. ({math_equations.eqref('spam_filter_bayes_2-spam')}).'''+r''' To make the classification prediction, we need to compare the probabilities of the email being spam, $P(\text{spam}|\textbf{email})$, against the email being _not_ spam, $P(\lnot\text{ spam}|\textbf{email})$.''' + f''' The latter probability has the same form as Eq. ({math_equations.eqref('spam_filter_bayes_1-spam')}):
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('spam_filter_bayes_1-not_spam', r'P(\lnot\text{ spam}|\textbf{email}) = \frac{P(\textbf{email}|\lnot\text{ spam})P(\lnot\text{ spam})}{P(\textbf{email})} \;.', numbered=True)
    math_equations.display_equation('spam_filter_bayes_1-not_spam')
    st.write(
        '''
        After applying our assumption of feature independence, we can follow the same steps to distill the specific details of our email sample to its underlying words:
        '''
    )
    math_equations.add_equation('spam_filter_bayes_2-not_spam', r'P(\lnot\text{ spam}|\textbf{email}) = \frac{P(\lnot\text{ spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\lnot\text{ spam})}{P(\text{spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\text{spam}) + P(\lnot\text{ spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\lnot\text{ spam})} \;.', numbered=True)
    math_equations.display_equation('spam_filter_bayes_2-not_spam')
    st.write(
        f''' Now that we have both probabilities, Eqs. ({math_equations.eqref('spam_filter_bayes_2-spam')}) and ({math_equations.eqref('spam_filter_bayes_2-not_spam')}), we note that their only difference is in the _numerators_, and the complicated expression in the denominators is the same. Since all of the quanities appearing in both equations are probabilities, which are constrained to be positive, we can make our classification based solely on which numerator is larger. Namely:
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('spam_filter_classification', r'\textbf{email}\text{ is spam if } \left(P(\text{spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\text{spam})\right) > \left(P(\lnot\text{ spam})\prod_{i \;\in\; \{1,\; \ldots \;, N\}} P(\text{word}_i|\lnot\text{ spam})\right) \text{ else, }\textbf{email}\text{ is not spam}')
    math_equations.display_equation('spam_filter_classification')

    st.subheader('Handling the rare word edge case: smoothing')
    st.write(
        r'''
        We have now laid out how a simple Naive-Bayes spam filter can be constructed and trained and used to make predictions on new emails. There is one small but important adjustment to make before this filter can be deployed. It deals with handling edge cases where we want to classify an email that contains a word that _never_ appeared in our training set. Recall how we calculate $P(\text{word}_i|\text{spam})$ and $P(\text{word}_i|\lnot\text{ spam})$ in '''+f'''Eqs. ({math_equations.eqref('prob_word_in_spam')}) and ({math_equations.eqref('prob_word_in_not_spam')}). If our new rare word never appeared in our training set, the numerators of each of these expressions is **zero** and our overall probabilities for the new email to be spam or not spam are zero. We need to modify '''+f'''Eqs. ({math_equations.eqref('prob_word_in_spam')}) and ({math_equations.eqref('prob_word_in_not_spam')}) to return a small but non-zero probability for very rare words that haven't shown up in our training set. This can be done via adding terms to the numerator and denominator as:
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('prob_word_in_spam-smoothed', r"P(\text{word}_i|\text{spam}) = \frac{\text{number of times `}\text{word}_i\text{' appears in all emails labeled `spam'} \textcolor{red}{+ \alpha}}{\text{ number of words in all emails labeled `spam'} \textcolor{red}{+ \alpha\times\text{ number of unique words in training set}}} \;")
    math_equations.display_equation('prob_word_in_spam-smoothed')
    math_equations.add_equation('prob_word_in_not_spam-smoothed', r"P(\text{word}_i|\lnot\text{ spam}) = \frac{\text{number of times `}\text{word}_i\text{' appears in all emails labeled `not spam'} \textcolor{red}{+ \alpha}}{\text{number of words in all emails labeled `not spam'} \textcolor{red}{+ \alpha\times\text{ number of unique words in training set}}} \;,")
    math_equations.display_equation('prob_word_in_not_spam-smoothed')
    st.write(
        r'''
        where $0 \leq \alpha \leq 1$. When $\alpha > 0$, when we encounter a new word that our algorithm hasn't seen before, we can still make a meaningful classification by estimating the probability of the rare word with a small, but fixed, number. The process of modifying the estimate for the posterior probability is called _smoothing_. The technique demonstrated above is called 'additive smoothing.' The case when $\alpha = 1$ is called _Laplace smoothing_, and $\alpha < 1$ _Lidstone smoothing_. It essentially means we apply a uniform prior to all possible features in the data (in this case words that _may_ (or _may not_) appear in emails). For words that _are_ in the training set, this fixed uniform prior is _updated_ via the expressions above.
        '''
    )

    # -----------------------------------
    # ----- Multi-class Naive-Bayes -----
    # -----------------------------------
    st.header('Beyond \'spam\' and \'not spam\': multi-class Naive-Bayes')
    st.write(
        r'''
        So far, we have only discussed Bayes Theorem in the case of _binary classification_. What happens when there are more classes? It turns out, the extension of our simple spam/not spam filter example to any number of classes is straightforward. In fact we've already done this when we took our expression for $P(\text{spam}|\textbf{email})$'''+f''' in Eq. ({math_equations.eqref('spam_filter_bayes_2-spam')})'''+r''' to write a similar expression for $P(\lnot\text{ spam}|\textbf{email})$ in'''+f''' Eq. ({math_equations.eqref('spam_filter_bayes_2-not_spam')}).
        '''
        +
        r'''
        Let's introduce new notation to generalize beyond two classes. To be concrete, suppose we have $M$ _classes_, indexed by $k = 1, 2, \ldots , M$. In the case of our spam filter example, $M = 2$. We represent a specific class by the symbol $C_k$. Next, suppose we have $N$ _features_, indexed by $i = 1, 2, \ldots, N$. In our spam filter, $N$ is the number of unique words in our training set. We can represent a data sample with an $N$-dimensional vector $\bm{x}$ with a component $x_i$ for each feature. (A perceptive reader may realize at this point why we have chosen to put the word 'email' in bold font in all of our prior math_equations. We were really imagining the email as a vector in the many-dimensional vector space defined by the vocabulary of our training set.) 

        Now that we defined symbols for classes ($C_k$'s) and data samples ($\bm{x}$'s with components $x_i$), we can rewrite our formulas for calculating probabilities. The probability of a data sample $\bm{x}$ having a class $C_k$ is written as $P(C_k | \bm{x})$. The probability of a given class containing our data sample is written as $P(\bm{x}|C_k)$. Probabilities of each class is written as $P(C_k)$. The probability of our data sample is written as $P(\bm{x})$.'''+f''' Using Eq.({math_equations.eqref('prob_B')}),'''+r''' $P(\bm{x})$ can be expended in terms of conditional probabilities given each class:
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('prob_x', r'P(\bm{x}) = \sum_{k=1}^M P(\bm{x}|C_k)P(C_k) \;.')
    math_equations.display_equation('prob_x')
    st.write(
        r'''
        The probability of our data sample $\bm{x}$ having class $C_k$ can then be written using Bayes Theorem as
        '''
    )
    math_equations.add_equation('prob_x_is_class_k', r'P(C_k|\bm{x}) = \frac{P(\bm{x}|C_k)P(C_k)}{P(\bm{x})} = \frac{P(C_k)\prod_{i=1}^N P(x_i|C_k)}{\sum_{k=1}^M P(C_k)\prod_{i=1}^N P(x_i|C_k)} \;,')
    math_equations.display_equation('prob_x_is_class_k')
    st.write(
        r'''
        where we have used our assumption that all of our features $x_i$ are independent of one another to replace $P(\bm{x}|C_k)$ with $\prod_{i=1}^N P(x_i|C_k)$. Notice we have switched to the notation $\prod_{i=1}^N$ instead of $\prod_{i \;\in\; \{1,\; \ldots \;, N\}}$ because in the general case, we include all probabilities, whether the feature appears in the given sample or not. When we want to ignore features that do not appear, we will build that in to how $P(x_i|C_k)$ is defined, instead of limiting the terms appearing in the product. To classify a data point $\bm{x}$, we calculate'''+ f''' Eq. ({math_equations.eqref('prob_x_is_class_k')})'''+r''' for every class $k$ and find which class has the largest probability. This may seem daunting, since the expression looks quite complicated. However, as we have noticed in the case of the spam filter, we do not need to calculate the entire expression. If we take a close look at '''+ f''' Eq. ({math_equations.eqref('prob_x_is_class_k')}),'''+r''' we notice that only the _numerator_ depends on the actual value of the class index $k$, and that the denominator is a constant that depends on the components of the feature vector $\bm{x}$. Hence the value of $k$ that cooresponds to the maximum probability is the same as the value that maximizes the numerator. So we can take our expression and define the classification scheme as
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('multiclass-classification', r'\text{data point }\bm{x}\text{ is class }C_{\hat{k}}\text{ where }\hat{k} = \argmax_{k \;\in\; \{1,\; \ldots \;, M\}} \left( P(C_k)\prod_{i=1}^N P(x_i|C_k) \right) \;.')
    math_equations.display_equation('multiclass-classification')
    st.write(
        fr'''
        The $\argmax$ function in Eq. ({math_equations.eqref('multiclass-classification')})'''+r''' means 'chose the value of $k$ which maximizes the argument' (quantity in the big parenthesis). This type of classification rule is also known as **_maximum a posteriori (MAP) estimation_**, because it maximizes the posterior probability $P(\bm{x}|C_k)$.
        '''
        +
        f'''Although perfectly mathematically correct, Eq. ({math_equations.eqref('multiclass-classification')}) is _not quite_ the optimal form of the classification scheme for the purposes of implementing the algorithm on a computer. Indeed, the multiplication of many small probabilities together can result in unexpected behavior based on how those numbers are represented in memory. After a small modification that leaves the _result_ of the classification unchanged, we will have an expression that is much more computationally safe than that in Eq. ({math_equations.eqref('multiclass-classification')}).
        ''', unsafe_allow_html=True
    )
    st.subheader('Transforming products into sums: the Logarithm')
    st.write(
        f'''
        The classification scheme in Eq. ({math_equations.eqref('multiclass-classification')})'''+r''' is actually _not_ ideal for implementing this algorithm on a computer. This is because it relies on multiplying many numbers together, each of which fall in the range of zero to one. (Actually the true range does not include either zero or one since we have applied _smoothing_ to our estimates of the posterior probabilities.) This means that, often we will be multiplying many numbers together that are individually very small, with the result being a really, _really_, **_really_** small number. We may even run into a computational issue known as **_underflow_**, which means we have a probability that is smaller in magnitude than the smallest number that can be represented in memory by the computer's CPU. For example, a standard floating point number has a smallest magnitude of about $10^{-38}$. In text classification where we have hundreds or thousands of features (unique words), and data samples containing significant amounts of features, we may easily be computing products involving numbers smaller than this (indeed possibly even smaller than the corresponding limits on double- or quaduple-precision floats). We need to find a way around this computational difficulty in a way in which we can avoid ever calculating such products in the first place, making the numerical implementation safe no matter how the numbers are stored in memory.
        
        To get around this, we use the _logarithm_ function, which has the very useful property that logarithms of numbers that are very large or small in magnitude become much more manageable, roughly the same magnitude as their exponent in scientific notation, e.g. $\log_{10}(10^{-50}) = -50$. The logarithm also has the handy property wherefore logarithms of products of factors become sums of logarithms of the factors: $\log(ab) = \log (a) + \log (b)$. This means that taking the logarithm of the product in '''+f'''Eq. ({math_equations.eqref('multiclass-classification')})'''+r''' results in a sum over logarithms of each probability, and we never have to multiply them together at all! Now, we can't simply apply a function to the right-hand-side and expect our classification scheme using $\argmax$ to still work. Recall, we select the class with the _maximum_ probability, and therefore the hierarchy of values is crucial. Applying some function to the maximum probability does not guarantee that the resulting number will still be the maximum (for example consider negation, which would take the _most_ likely class and turn it into the _least_ likely class!). However, one final property of the logarithm saves us here: it is _monotonically increasing_. This means for $0 < x_1 < x_2$, $\;\log(x_1) < \log(x_2)$. In otherwords, the heirarchy of our class probabilities is preserved, and the most likely class will remain most likely.

        Now that we have a work-around to possible underflow issues, let us modify our main result into the form typically used in implementations of Naive-Bayes classification. Taking the logarithm of the argument in the right-hand-side of '''+f'''Eq. ({math_equations.eqref('multiclass-classification')}) and using the property of logarithms of products, we have
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('multiclass-classification-log_1', r' P(C_k)\prod_{i=1}^N P(x_i|C_k) \to \log\left( P(C_k)\prod_{i=1}^N P(x_i|C_k) \right) = \log\left(P(C_k)\right) + \sum_{i = 1}^N\log\left(P(x_i|C_k)\right)\;.')
    math_equations.display_equation('multiclass-classification-log_1')
    st.write('Therefore our classification scheme is now')
    math_equations.add_equation('multiclass-classification-log_2', r'\text{data point }\bm{x}\text{ is class }C_{\hat{k}}\text{ where }\hat{k} = \argmax_{k \;\in\; \{1,\; \ldots \;, M\}} \left[\log\left(P(C_k)\right) + \sum_{i = 1}^N\log\left(P(x_i|C_k)\right)\right] \;.')
    math_equations.display_equation('multiclass-classification-log_2')
    st.write(
        r'''
        The $\log(P(x_i|C_k))$ terms are refered to as _log-likelihoods_ and the $\log(P(C_k))$ terms as _log-priors_. The new form of our classification scheme in '''+f''' Eq. ({math_equations.eqref('multiclass-classification-log_2')})'''+r''' has the following intuitive meaning. The log-priors give a baseline for how likely a given class is correct: classes with high probabilities (occuring more frequently in the _existing_ data) are, by nature, more likely to be the true class of _new_ data. Then, the information from the actual sample adds more or less support for this class. Each feature $x_i$ that shows up in the data sample $\bm{x}$ _adds evidence for_ a class $C_k$ when that feature is more associated with that class so that $\log(P(x_i|C_k))$ is large.

        One final note, you may be wondering, since we turned a product of probabilities into a sum of logs of probabilities, if we still need to incorporate smoothing for rare features. Before, if we had a feature appearing in a new data sample that was not present in the training data, we would arrive at a conditional probability for each class of zero, which would make the entire probability of the new sample having any class zero, since we multiply all of the probabilities together. If we are instead adding them, does it matter that we may have zero probabilities in the summation? Actually, by taking the logarithm, we have made this edge case even more dangerous. This is because the logarithm of zero is _undefined_. In fact, the limit of $\log x$ as $x$ approaches zero from the right is negative infinity! A computer may return unpredictable results if we asked it to calculate this. Therefore, we still must apply smoothing to the data by choosing a uniform prior so that we don't introduce infinities into our computation.
        ''', unsafe_allow_html=True
    )

    # --------------------------------
    # ----- Types of Naive-Bayes -----
    # --------------------------------
    st.header(r'Flavors of Naive-Bayes: estimating $P(x_i|C_k)$')
    st.write(
        r'''
        All Naive-Bayes classifiers use the same classification scheme. (Actually this is not _exactly_ true, the example of Complement Naive-Bayes described below uses a slightly modified version of '''+f''' Eq. ({math_equations.eqref('multiclass-classification-log_2')}), but the principle of maximum a posteriori estimation remains.) However, depending on the specific type of data we are classiying, and how it is distributed within each class, will inform how to estimate $P(x_i|C_k)$. One should ask questions such as: _Are the features numerical or categorical?_ If numeric: _Are the features discrete or continuous?_ If categorical: _Are the features binary or not?_ The answers to these questions should point to which distribution one should chose when estimating the conditional probabilities. 
        
        Since we are ultimately interested in implementing the Naive-Bayes algorithm using the python library _sklearn_, we will now cover the different options when choosing a Naive-Bayes classifier included in the `sklearn.naive_bayes` module.
        ''', unsafe_allow_html=True
    )

    # ----- Multinomial Naive-Bayes -----
    st.subheader('Multinomial Naive-Bayes: `MultinomialNB()`')
    st.write(
        r'''
        For data with features that are multinomially distributed, the probability of a feature $i$ given a class $k$ is calculated by dividing the frequency of that feature in the given class, divided by the total number of features in the given class (subscript 'M' for multinomial):
        '''
    )
    math_equations.add_equation('prob_xi_is_class_k-multinomial', r'P_M(x_i \neq 0|C_k) = \frac{N_{ik}}{N_k} \;,')
    math_equations.display_equation('prob_xi_is_class_k-multinomial')
    st.write(
        r'''
        where $N_{ik} = \sum_{\text{class }k} x_i$ is the sum of feature $x_i$ appearing in all samples $\bm{x}$ labeled with class $k$ and $N_k = \sum_{i=1}^N N_{ik}$ is the sum of all features from all data with class $k$. Notice this is exactly how we calculated the probabilities of words in spam emails in in'''+f''' Eq. ({math_equations.eqref('prob_word_in_spam')})'''+r''' with "number of times word${}_i$ appears in all emails labeled 'spam'" corresponding to $N_{ki}$ and "number of words in all emails labeled 'spam'" corresponding to $N_k$. Note, we specify $x_i \neq 0$ since for now we are addressing only features that actually appear in the given data sample.'''
        +
        f'''
        
        Now, Eq. ({math_equations.eqref('prob_xi_is_class_k-multinomial')})'''+r''' suffers from the 'rare feature' limitation, meaning that if it is used to classify a data sample containing a feature not present in the training data, the numerator will be zero. This problem was solved with smoothing of the data, meaning all possible features are given a small, uniform prior probability. For features that are actually present in the training data, this uniform probability is updated using
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('prob_xi_is_class_k-multinomial-smoothed', r'P_M(x_i \neq 0|C_k) = \frac{N_{ik} + \alpha}{N_k + \alpha N} \;,')
    math_equations.display_equation('prob_xi_is_class_k-multinomial-smoothed')
    st.write(
        r'''
        where $0 \leq \alpha \leq 1$ is fixed, and $N$ is the number of unique features in the data. This is the same form as the smoothed probabilities in '''+f'''Eq. ({math_equations.eqref('prob_word_in_spam-smoothed')}), '''+r''' with the new term $N$ corresponding to "number of unique words in training set."
        
        Multinomial Naive-Bayes is suitable when the features used to classify the data are _discrete_, for example word counts in text classification where the values of the sample components $x_i$ are positive integers including zero. However, in the case that $x_i = 0$, i.e. feature $i$ is absent in the sample, we _define_ $P_M(x_i = 0|C_k)\equiv 1$. This is a _definition_, and **_does not_** imply a probability of 100%. Indeed, for a different sample where feature $i$ _is_ present, then we have $x_i \neq 0$, and $P_M(x_i \neq 0|C_k)$ is calculated as in '''+f''' Eq. ({math_equations.eqref('prob_xi_is_class_k-multinomial-smoothed')}) above.'''+r''' The _reason_ for defining $P_M(x_i = 0|C_k)\equiv 1$ is so that it has **_no effect_** in the classification scheme. If we are using the form involving a product of probabilities in '''+f'''Eq. ({math_equations.eqref('multiclass-classification')}), a term with a probability that equals one does not affect the product. Similarly, if we are using the form involving a sum of logarithms of probabilities in Eq. ({math_equations.eqref('multiclass-classification-log_2')}), a term with a probability that equals one does not affect the sum, since $\log 1 = 0$. Note, this special case of treating features that are absent in the data sample we are trying to classify is unique to Multinomial Naive-Bayes. The other variations of the Naive-Bayes classifier will not use a separate treatment. Because of this special treatment, Multinomial Naive-Bayes may struggle with shorter text samples, where features that are indicitive of a given class may be absent.'''
        +r'''

        Finally, as evidenced in the expressions for $P_M(x_i\neq 0| C_k)$ in this section, Multinomial Naive-Bayes relies heavily on information contained in a class to make estimates about probabilities of data belonging to that class. The method therefore suffers when any given class is not well represented in the training data. This situation occurs if the training data is unbalanced. A way to handle this deficiency for unbalanced training data, Multinomial Naive-Bayes can be modified to instead use data from all classes _except_ a given class, to make classifications about that class. This modified version is called Complement Naive-Bayes, and is discussed below.
        ''', unsafe_allow_html=True
    )

    # ----- Complement Naive-Bayes -----
    st.subheader('Complement Naive-Bayes: `ComplementNB()`')
    st.write(
        r'''
        Complement Naive-Bayes is an adaptation of Multinomial Naive-Bayes, introduced in the article [Tackling the poor assumptions of naive bayes text classifiers](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) from MIT's Artificial Intelligence Laboratory by Rennie et al. (2003). Complement Naive-Bayes gets its name from the notion of a _complement of a set_, which is all of the elements in a domain that are not included in the set. Regular Multinomial Naive-Bayes uses data from a given class $k$ to determine whether a data sample belongs to that class. Coorespondingly, the Complement Naive-Bayes model uses data from all classes _except_ $k$ to make the determination. The method for calculating $P_{M^C}(x_i|C_k)$ (subscript '$M^C$' for complement of multinomial) in the Complement Naive-Bayes model is similar to that in Multinomial Naive-Bayes:
        '''
    )
    math_equations.add_equation('prob_xi_is_class_k-complement-smoothed', r'P_{M^C}(x_i \neq 0|C_k) = \frac{\tilde{N}_{ik} + \alpha}{\tilde{N}_k + \alpha N} \;,')
    math_equations.display_equation('prob_xi_is_class_k-complement-smoothed')
    st.write(
        r'''
        where $\tilde{N}_{ik} = \sum_{\text{class }\neq\; k} x_i$ is the sum of feature $x_i$ appearing in all samples $\bm{x}$ labeled with all classes _other than_ $k$ and $\tilde{N}_k = \sum_{i=1}^N \tilde{N}_{ik}$ is the sum of all features from all data with classes _other than_ $k$. To account for the use of data _not_ in class $k$, we need to make a small adjustment to the classification scheme. Instead of _adding_ the logarithms of $P_{M^C}(x_i|C_k)$, we instead _subtract_ them:
        '''
    )
    math_equations.add_equation('multiclass-classification-complement', r'\text{data point }\bm{x}\text{ is class }C_{\hat{k}}\text{ where }\hat{k} = \argmax_{k \;\in\; \{1,\; \ldots \;, M\}} \left[\log\left(P(C_k)\right) - \sum_{i = 1}^N\log\left(P_{M^C}(x_i|C_k)\right)\right] \;.')
    math_equations.display_equation('multiclass-classification-complement')
    st.write(
        r'''
        The minus sign ensures that we assign to class $k$ documents that _most poorly_ match the _complement_ of class $k$, meaning small values of $P_{M^C}(x_i|C_k)$. Finally, as we did with regular Multinomial Naive-Bayes, we make the definition $P_{M^C}(x_i = 0|C_k) \equiv 1$, which effectly ignores all features that are not present in the given data sample.
        '''
    )

    # ----- Bernoulli Naive-Bayes -----
    st.subheader('Bernoulli Naive-Bayes: `BernoulliNB()`')
    st.write(
        r'''
        Like Multinomial and Complement Naive-Bayes, Bernoulli Naive-Bayes is also suitable for discrete features. However, in the Bernoulli model, feature components are _binary_, taking on the value of either zero or one. A value of one corresponds to the feature being present in the sample, and a value of zero corresponds to a feature being absent. The conditional probabilities $P_B(x_i|C_k)$ (subscript 'B' for Bernoulli) are calculated as any of the following equivalent definitions
        '''
    )
    math_equations.add_equation('prob_xi_is_class_k-bernoulli', r'P_B(x_i|C_k) = p_{ik}^{x_i}(1-p_{ik})^{1-x_i} = p_{ik}x_i + (1-p_{ik})(1-x_i) = \begin{cases} p_{ik} \text{ for } x_i = 1 \\ 1-p_{ik} \text{ for } x_i = 0 \end{cases}\;,')
    math_equations.display_equation('prob_xi_is_class_k-bernoulli')
    st.write(
        r'''
        where $p_{ik}$ is the probability of feature $i$ appearing in class $k$, i.e. the fraction of samples with class label $k$ where feature $i$ is present. In a slight abuse of notation, we can represent this fraction in much the same way as $P_M(x_i|C_k)$ in'''+f''' Eq. ({math_equations.eqref('prob_xi_is_class_k-multinomial')}):
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('p_ik', r'p_{ik} = \frac{N_{ik}}{N_k} \;,')
    math_equations.display_equation('p_ik')
    st.write(
        r'''
        where in this expression, $N_{ik}$ is the _number of samples with class label_ $k$ _where feature_ $i$ _is present_ and $N_k$ is the _number of samples with class label_ $k$. Note that $p_{ik}$ can be zero (and thus also $P_B(x_i|C_k)$), when we encounter a feature not in the training set. This means we need to also apply smoothing to $p_{ik}$ in the same mannor as done for $P_M(x_i|C_k)$ in '''+f'''Eq. ({math_equations.eqref('prob_xi_is_class_k-multinomial-smoothed')}):
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('p_ik-smoothed', r'p_{ik} = \frac{N_{ik} + \alpha}{N_k + \alpha N} \;,')
    math_equations.display_equation('p_ik-smoothed')
    st.write(
        r'''
        where in this expression, $\alpha$ and $N$ have the same meaning as in '''+f'''Eq. ({math_equations.eqref('prob_xi_is_class_k-multinomial-smoothed')}).'''+r'''

        One important feature of the Bernoulli model is that it penalizes both the _lack of appearance_ of abundant features, and the _appearance_ of rare ones. Looking at how $P_B(x_i|C_k)$ is calculated in '''+f'''Eq. ({math_equations.eqref('prob_xi_is_class_k-bernoulli')}),'''+r''' if $p_{ik}$ is close to one, meaning feature $i$ occurs in most instances of the class $k$, and the given data sample _does not_ contain this feature, i.e. $x_i = 0$, the probability $P_B(x_i|C_k)$ is _reduced_, being equal to $1-p_{ik}$ which is small since $p_{ik}$ is large. In the opposite situation, were $p_{ik}$ is close to zero, meaning feature $i$ occurs rarely in instances of the class $k$, but the given data sample _does_ contain this feature, i.e. $x_i = 1$, the probability $P_B(x_i|C_k)$ is _also reduced_, being equal to $p_{ik}$ which is also small since $p_{ik}$ is small. This is an important distinction between the Bernoulli model and the Multinomial model. Both are useful for text classification, but in the case of the Multinomial model, when a feature is absent and we have $x_i = 0$, we use the special case of $P_M(x_i = 0|C_k) \equiv 1$, which actually negates any effect this feature has on the classification scheme, as explained in the section on Multinomial Naive-Bayes. This distinction makes the Bernoulli model more effective for shorter text samples such as tweets or academic abstracts, where the absence of terms associated with a class or topic is more noticable and important. The Multinomial model is better for longer text samples when there is enough data to use the frequency of terms. 
        ''', unsafe_allow_html=True
    )

    # ----- Categorical Naive-Bayes -----
    st.subheader('Categorical Naive-Bayes: `CategoricalNB()`')
    st.write(
        r'''
        For data with features that are categorical, i.e. contain values that are all of a specific type or category, one can use categorical distributions to estimate $P(x_i|C_k)$. Categorical data includes features like 'color' (with values such as 'red', 'blue', 'green', ...), 'size' (with values such as 'small', 'medium', 'large', ...), 'genre' (with values such as 'action', 'comedy', 'drama', ...) etc. The values for the features are pulled from a set of options that do not imply any inherent heirarchy or ordering. This is different from features that contain an inherit ordering such as distances, frequency, weight, etc. A data point with 'blue' as its value for 'color' would not have an implied relation of larger or smaller than, say, another data point with a value of 'red' for 'color'. Since the feature components $x_i$ are now categorical values, instead of numerical ones, we have the notation $P_C(x_i = t|C_k)$ (subscript 'C' for categorical) which reads 'probability feature $i$ is catagory $t$, given class $k$. It is calculated as: 
        '''
    )
    math_equations.add_equation('prob_xi_is_class_k-categorical', r'P_C(x_i = t|C_k) = \frac{N_{tik} + \alpha}{N_k + \alpha n_i} \;,')
    math_equations.display_equation('prob_xi_is_class_k-categorical')
    st.write(
        r'''
        where $N_{tik}$ is the number of times feature $i$ is category $t$ when $\bm{x}$ has class $k$, $N_k$ is the number of samples of class $k$, and $n_i$ is the number of different values of the category for feature $i$ (for example if feature $i$ is color, with the option of three different colors, then $n_i = 3$). The smoothing parameter $\alpha$ is the same one we have used in the other variations of Naive-Bayes. Although $P_C(x_i=t|C_k)$ is quite similar to $P_M(x_i \neq 0|C_k)$, the latter relies on the numerical values of $x_i$ to correspond to appearences of a given feature in the data. This is very different when $x_i$ is categorical, _even if_ it has a numerical value: $x_i = 30$ _does not_ mean the feature $x_i$ appeared thirty in the sample, it means $x_i$ had the _category_ of 30, for example if feature $i$ was a pants waist measurement.

        When implementing Categorical Naive-Bayes on a computer, it is useful to preprocess the discrete values of each categorical feature into integers using a process known as _encoding_. Encoding simply means assiging to $x_i$, for each feature $i$, an integer in the range $0, 1, \ldots, n_i-1$. For the example where $x_i$ is a color, taking the values of 'red', 'green', or 'blue', one can assign 0 to 'red', 1 to 'green', and 2 to 'blue'. This type of encoding is called _ordinal encoding_. Note, although ordinal encoding results values that are numerically ordered, it _does not_ imply the same ordering on the original feature values. An encoded value of $x_i = 2$ (meaning 'blue' for the color example) is to be interpreted as _greater than_ an encoded value of 0 (meaning 'red' in the color example), it is simply a method of indexing the different choices for each category. Ordinal encoding can be achieved with the `OrdinalEncoder()` function from the `sklearn.preprocessing' module.

        Alternatively, one could use a different encoding process called _one-hot encoding_, which takes each categorical feature $i$ and turns it into $n_i$ _binary_ features taking a value of $0$ and $1$, corresponding to $x_i$ having the specific category or not. For the color example, one-hot encoding would turn the 'color' feature into three binary features: 'color-red', 'color-green', 'color-blue' where a value of 'color-red' of 1 means that $x_i$ had the value 'red'. These new features are also called _dummy variables_ or _dummy features_. Having done this, one could then use Bernoulli Naive-Bayes, since each feature has become binary. This process works fine, but theoretically, it severely violates the assumption of _feature independence_, since a value for a one-hot encoded feature of 1 _implies_ the others are zero: when 'color-red' is one, then 'color-green' and 'color-blue' are _necessarily_ zero. Despite this lack of feature independence among the dummy variables, using them in this way still gives high accuracy. One-hot encoding can be achieved with the `OneHotEncoder()` function from the `sklearn.preprocessing' module.
        
        One should take care when deciding how to encode categorical features whether to use normal encoding by assigning each feature category an integer or to create dummy features through one-hot encoding. If a given feature category contains _many_ different values, will using one-hot encoding result in too many features causing overfitting? Is there enough data so that all of the dummy variables are populated? If $n_i$ is large for a given feature, it may make more sense to use Ordinal Encoding to avoid creating too many features and risk overfitting.
        '''
    )
    
    # ----- Gaussian Naive-Bayes -----
    st.subheader('Gaussian Naive-Bayes: `GaussianNB()`')
    st.write(
        r'''
        When the data is numerical, but has _continuous_ values, then one cannot use the above variations of Naive-Bayes estimation. This would be the case for features such as 'distance' or 'weight', where values fall on a spectrum. For data that follows a normal, or Gaussian, distribution, the conditional probabilities $P_G(x_i|C_k)$ (subscript 'G' for Gaussian) are calculated as
        '''
    )
    math_equations.add_equation('prob_xi_is_class_k-gaussian', r'P_G(x_i|C_k) = \frac{1}{\sqrt{2\pi}\sigma_k}\exp\left(-\frac{(x_i - \mu_k)^2}{2\sigma_k^2}\right) \;,')
    math_equations.display_equation('prob_xi_is_class_k-gaussian')
    st.write(
        r'''
        where the feature mean $\mu_k$ and standard deviation $\sigma_k$ are found from maximum liklihood estimation via
        '''
    )
    math_equations.add_equation('mu_k-MLE', r'\mu_k = \frac{1}{N}\sum_{\text{class }k} x_i \;,')
    math_equations.display_equation('mu_k-MLE')
    st.write(
        '''
        and
        '''
    )
    math_equations.add_equation('sigma_k-MLE', r'\sigma_k = \sqrt{\frac{1}{N}\sum_{\text{class }k} (x_i - \mu_k)^2} \;.')
    math_equations.display_equation('sigma_k-MLE')
    st.write(
        r'''
        Gaussian Naive-Bayes is suitable for data classified using features that are _continuous_, and normally distributed. An advantage of Gaussian Naive-Bayes, is that if the features are _not individually_ normally distributed, we can still make use of Gaussian Naive-Bayes by calculating the _mean_ of the features. According to the Central Limit Theorem, the distribution of the _mean_ of individual random variables tends toward a normal distribution when many samples are included. Another advantage of Gaussian Naive-Bayes is that $P_G(x_i|C_k)$ is never zero, and thus we don't need to introduce any smoothing or uniform priors.
        '''
    )
    
    # -----------------------------------------
    # ----- Naive-Bayes algorithm outline -----
    # -----------------------------------------
    st.header('Putting it all together')
    st.write(
        r'''
        Using the Naive-Bayes Classification Algorithm in `sklearn` typically involves the following process:
         - **Model identification:** Examine the individual features and determine which variation of Naive-Bayes to use. Guiding thoughts include
            - Is the data numerical or categorical?
                - If numerical, are the values discrete or continuous?
                - If categorical, are the values binary? If not binary, should one use ordinal encoding or one-hot encoding?
            - If the data contains mixed numerical and categorical features, should the categorical features be dropped or converted to numerical values through encoding?

         - **Data splitting:** Decide how to split the data into training, testing, and validation (if needed) sets. The validation set is usually used for tuning hyperparameters, such as the smoothing parameter $\alpha$ when we have discrete, numerical data features. Note, since Naive-Bayes classification is a _supervised_ algorithm, all of the data in these sets are already labeled with a class. This labeling is used during training to calculate the posterior probabilities, and is used during testing and validation to compare the trained classifier's predicted labels with the already known true labels. The function `train_test_split` in the `sklearn.model_selection` module can be used to split the data into training and testing portions. If a validation set is needed, it can be used again to further split the testing set into testing and validation. This function takes several arguments to control how the splitting is performed, and it is best to illustrate these with concrete examples.
        
        - **Training:** Once one has chosen a classifier type, it is trained on the training data. In `sklearn`, the following steps are accomplished with the `.fit()` method.
            1. Index all of the classes in the dataset with the index $k$
            2. Index all of the features in the dataset that are used to make classifications with the index $i$
            4. Compute $P(C_k)$, the probability of each class
            3. Calculate all of the posterior probabilities $P(x_i|C_k)$, for each class $k$, for each feature $i$ in the training dataset

        - **Testing:** Once the classifier is trained, it is evaluated on the testing data. In `sklearn`, the following steps are accomplished with the `.predict()` method.
            1. Given a new sample $\bm{x}$, for each class $k$, extract all relevant posterior probabilities that were calculated during training and multiply them together with the class probability $P(C_k)$
            2. Compare the resulting probability for each class and find the maximum
            3. Assign to $\bm{x}$ the class $k$ associated with the maximum probability
        
        - **Analysis:** Once the classifier is trained, various statistics can be used to evaluate its performance using built-in tools from the  `sklearn.metrics` module. The next section covers these methods and their interpretation in more detail.

        A note on Naive-Bayes predictions: Naive-Bayes is a powerful classification tool, but is known as a poor _estimation_ tool. This means that one can use `sklearn`s `.predict()` method on new data to make classifications, but the output of the `.predict_proba()` method, which contains the probabilities for each class, should not be taken seriously.
        ''', unsafe_allow_html=True
    )

    # --------------------------------------
    # ----- sklearn evaluation metrics -----
    # --------------------------------------
    st.header('Model analysis in sklearn')
    st.write(
        '''
        The module `sklearn.metrics` contains many useful tools for analysing the performance of a machine learning model and is not unique to naive-bayes classification. Once a classifier has been trained on the training set using the `.fit()` method, the first tool for its evaluation is the `.score()` method. This method uses the data in the training set to make predictions. Then those predictions are compared with the base truth labels and the _accuracy_ is returned. The accuracy is a simple ratio of number of correct predictions versus total number of predictions:
        '''
    )
    math_equations.add_equation('accuracy_description', r'\text{accuracy} = \frac{\text{Number of correct predictions}}{\text{number of predictions}}')
    math_equations.display_equation('accuracy_description')
    st.write(
        '''
        Before moving on, let us discus the meaning of `correct predictions` in more detail and introduce some new terms. There are four types of predictions a model can make, depending on the prediction and ground truth values. These are easiest to understand for binary classification, when a prediction has only two outcomes. For now, let us use the terms 'positive' and 'negative' to refer to them. The variations of predictions are then:
         - **True Positives (TP)**: ground truth is positive and the model predicted positive
         - **False Positives (FP)**: ground truth is negative and the model predicted positive
         - **True Negatives (TN)**: ground truth is negative and the model predicted negative
         - **False Negatives (FN)**: ground truth is positive and the model predicted negative
        
        For multiclass classification, these definitions change with each class. For a given class 'A', we have
         - **True Positives (TP)**: ground truth is class 'A' and the model predicted class 'A'
         - **False Positives (FP)**: ground truth is any class except 'A' and the model predicted class 'A'
         - **True Negatives (TN)**: ground truth is any class except 'A' and the model predicted any class except 'A'
         - **False Negatives (FN)**: ground truth is class 'A' and the model predicted any class except 'A'

        The term 'true negative' loses its meaning for multiclass classification, and is really just meant as the predictions that are not relevant to the given class at hand. A given prediction will change categories, depending on which class is relevant. For example, a prediction that is a true positive for class 'A' could become a false positive, true negative, or false negative relative to a different class 'B'.
        
        By counting the number of each type of prediction, different statistics are calculated. The accuracy, described in words in '''+f'''Eq. ({math_equations.eqref('accuracy_description')}), is more concretely defined as        
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('accuracy', r'\text{accuracy} = \frac{TP + TN}{TP + FP + FN + TN} \;.')
    math_equations.display_equation('accuracy')
    st.write(
        '''
        Another statistic is _precision_, which is the fraction of correctly predicted positive results out of all positive predictions made:
        '''
    )
    math_equations.add_equation('precision', r'\text{precision} = \frac{TP}{TP + FP} \;.')
    math_equations.display_equation('precision')
    st.write(
        '''
        A third statistic that can be calculated is the _recall_, which is the fraction of correctly predicted positive results out of all of the samples with a positive ground truth:
        '''
    )
    math_equations.add_equation('recall', r'\text{recall} = \frac{TP}{TP + FN} \;.')
    math_equations.display_equation('recall')
    st.write(
        '''
        Remember, false negatives have positive ground truth.  Recall differs from precision in that recall answers the question 'of the samples that are actually positive, which were correctly identified?' while precision answers the question 'of the times a positive result was predicted, how many of those were actually positive?' A visual explaination for this is shown in the diagram below from [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall):
        '''
    )
    # Display centered precision-vs-recall diagram
    col1, col2, col3 = st.beta_columns(3)
    with col2:
        st.subheader('Precision vs Recall:')
        st.image('./resources/images/Precisionrecall.png')
    st.write(
        '''
        A fourth statistic that can be calculated is called the $F_1$ _score_. It is related to precision and recall by
        '''
    )
    math_equations.add_equation('f1_score', r'F_1\text{ score} = \frac{2}{1/\text{precision} + 1/\text{recall}} = 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}} = \frac{TP}{TP + (FP+FN)/2} \;.')
    math_equations.display_equation('f1_score')
    st.write(
        r'''
        The $F_1$ score is the _harmonic mean_ of precision and recall. The F1 score can at most be 1, when both precision and recall are 1 (perfect classification), and at least be zero, when either precision or recall is zero. In '''+f'''Eq. ({math_equations.eqref('f1_score')}), the weight of precision and recall are equal. If one wants to weighs one or the other, the general form is
        ''', unsafe_allow_html=True
    )
    math_equations.add_equation('fbeta_score', r'F_\beta\text{ score} = (1+\beta^2)\cdot\frac{\text{precision}\cdot\text{recall}}{\beta^2\cdot\text{precision}+\text{recall}} = \frac{(1+\beta^2)\cdot TP}{(1+\beta^2)\cdot TP + FP+\beta^2\cdot FN} \;,')
    math_equations.display_equation('fbeta_score')
    st.write(
        r'''
        where $\beta > 1$ weighs recall over precision and $\beta < 1$ the opposite. 
        '''
    )
    st.subheader('Prioritizing a metric: costs of Type-I and Type-II errors')
    st.write(
        r'''
        False positives and false negatives are also often referred to as _Type-I_ and _Type-II_ errors. False positives (type-I) are the rejection of a true null hypothesis (an innocent person was found guilty) while false negatives (type-II) are failures to reject false null hypothesis (a guilty person was not convicted). Often it isn't the best practice to simply adjust hyperparameters until the accuracy is a maximum. The relative importance of accuracy, precision, recall, and $F_1$ score depends on the application of the model and the real-life consequences of type-I and type-II errors. For example, in a spam email filter, a **false positive** would mean a non-spam email is identified as spam. This could have a high cost to the end user since it means a possibly important email ends up in their spam folder which is rarely checked, while the cost of a **false negative** is low, since at worse the user would see the spam email in their inbox and proceed to delete it. In this case, a high _precision_ would be important, since it measures 'of the emails classified as spam, how many were actually spam?'. If precision is prioritized, fewer non-spam emails would be directed to the spam folder. On the other hand, if the classifier is predicting an illness on the basis of patient symptoms or medical scans, then the cost of a **false negative** would be very high. The patient would actually have the illness but be classified as well! Here, one would want to prioritize _recall_ since it measures 'of all the patients who are actually ill, how many were diagnosed?' A doctor would much rather prefer an algorithm hedge on the side of safety and predict an illness, after which follow-up proceedures would be recommended which would rule out the initial diagnosis, over prediction of no illness and no follow-up. Even if there were many false positives, the cost of a single false negative (patient goes home and believes they are well, meanwhile the illness could progress and their health deteriorate) is much higher than a false positive (the patient goes through further testing which finds no illness). If both false positives and false negatives are equally costly, one could prioritize the $F_1$ score, which is high when both precision and recall is high, and low when either is low. Finally, could also use a weighted $F_\beta$ score as well if either preicision or recall is favored, but both are important.
        '''
    )

    st.subheader('Viewing the metrics: Classification report and Confusion matrix')

    st.write(
        r'''
        Once one has given the real-world impact of type-I and II errors some thought, the actual value of the statistics measures discussed above can be easily obtained with the `classification_report()` function. First, one must generate the actual predictions of the model on the testing set as an imput for the report. This is already done with the `.score()` method, but to save the predictions, one can use the `.predict()` method, which takes just the data features in the testing set and returns the predicted classes. Then, with the predictions, one can generate a summary of various metrics with the `classification_report` function. This function takes the predictions and ground truth labels and calculates, in addition to accuracy, precision, recall, and $F_1$ score, the following:
         - **Macro Avg**: is the unweighted mean value of precision and recall. 
         - **Weighted Avg**: is the weighted mean value of precision and recall by the support values for each class.
         - **Support**: is the number of observations in each class to predict.
        
        Note that all of the metrics in the classification report are directly calculated from the individual numbers of true and false positives, and true and false negatives. The output of the classification report is a formatted string for easy viewing on a screen, but one can choose a dictionary output to access the individual values easier through key-value pairs.
        '''
        +
        '''
        A second metric that is directly calculated from the predictions vs ground truth values is the _confusion matrix_, calculated with the `confusion_matrix()` function that takes the same arguments as the classification report. The confusion matrix is a square matrix with size being the number of classes. Each row corresponds to a ground truth class, while each column corresponds to a predicted class. The number in each cell of the matrix has a different meaning depending on its location. Here is a diagram of the meaning of each cell for a binary and multiclass classification problems (sources: [left](https://docs.wso2.com/display/ML110/Model+Evaluation+Measures), [right](https://towardsdatascience.com/roc-curve-explained-using-a-covid-19-hypothetical-example-binary-multi-class-classification-bab188ea869c)):        
        '''
    )
    # Display centered binary confusion matrix explaination figure
    col1, col2, col3 = st.beta_columns([1,1,1.5])
    with col1:
        st.subheader('Binary classification:')
        st.image('./resources/images/Binary_Classification_Matrix_Definition.png', use_column_width=True)
    # Display centered multi-class confusion matrix explaination figure
    with col3:
        st.subheader('Multiclass classification:')
        st.image('./resources/images/confusion_matrix_structure_color.jpeg', output_format='jpeg')
    st.write(
        r'''
        The binary case is very simple. Diagonal terms are correct predictions (either positive or negative), and off-diagonal terms are type-I (lower-half) and type-II (upper-half) errors. The values for $TP$, $FP$, $TN$, and $FN$ are then simply the value in the single corresponding cell. From these values, overall accuracy, precision, recall, and $F_1$ score is determined. For multiclass classification, the case is slighty more complicated. The entries of the confusion matrix are described as 'number of samples with ground truth class A (in row A) that were predicted as class B (in column B). For each class A, the true positives is the number in the cell along the main diagonal in row A. False negatives are the other cells in that row. False positives are the cells in column A, not including the cell along the main diagonal. Finally, true negatives are all of the remaining cells. Then, the values of $TP$, $FP$, $TN$, and $FN$ are the _sums_ of the cells in the corresponding positions. From these, _class specific_ values of precision, recall, and $F_1$ score are calculated. There is no class-specific version of accuracy, since the accuracy for a multiclass model is the sum of the true positives divided by the total number of predictions. To be concrete, if an element in row $A$, column $B$ of the confusion matrix is labeled $C_{AB}$, then we have
        '''
    )
    math_equations.add_equation('class_specific_precision', r'\text{precision}_i = \frac{C_{ii}}{C_{ii} + \sum_{j\neq i} C_{ij}}')
    math_equations.display_equation('class_specific_precision')
    math_equations.add_equation('class_specific_recall', r'\text{recall}_i = \frac{C_{ii}}{C_{ii} + \sum_{j\neq i} C_{ji}}')
    math_equations.display_equation('class_specific_recall')
    math_equations.add_equation('class_specific_f1_score', r'F_1\text{ score}_i = \frac{C_{ii}}{C_{ii} + \frac{1}{2}\sum_{j\neq i}(C_{ji} + C_{ij})}')
    math_equations.display_equation('class_specific_f1_score')
    math_equations.add_equation('multiclass_accuracy', r'\text{accuracy} = \frac{\sum_i C_{ii}}{\sum_i\sum_j C_{ij}}')
    math_equations.display_equation('multiclass_accuracy')
    st.write(
        '''
        When the entries of the confusion matrix correspond to counts of prediction types, the matrix is said to be _unnormalized_. This means it's values correspond to bare counts and will depend on the size of the testing set. To allow for easier tuning of hyperparameters, one can _normalize_ the confusion matrix. The choices for normalization are _row normalization_, _column normalization_, or _population normalization_. When the matrix is normalized by row, each element in a row (corresponding to samples with a single ground truth class) are divided by the sum of elements in the row. Then, the elements along the main diagonal exactly correspond to the _recall_ of the class (true positives divided by true positives plus false negatives). When the matrix is normalized by column, each element in a column (corresponding to samples with a single predicted class) are divided by the sum of elements in the column. Then, the elements along the main diagonal exactly correspond to the _precision_ of the class (true positives divided by true positives plus false positives). Finally, if the matrix is normalized by population, each cell is divided by the same number, namely the total number of predictions. In this case, the sum of the diagonal entries corresponds to the model accuracy. No matter the choice of (or lack of) normalization, the confusion matrix of a highly performing model will have large values along the main diagonal and small values elsewhere.
        '''
    )


    # ---------------------------
    # ----- Reference notes -----
    # ---------------------------
    st.header('References and resources')
    st.write(
        '''
        The discussion of conditional probability and Bayes Theorem contains well known results which are available in any introductory statistics textbook, or Wikipedia.

        The spam filter example is a common pedegogical tool for illustrating the Naive-Bayes classification algorithm and similar incarnations can be found in various teaching resourses. 

        The details of the variations of Naive-Bayes classifiers implemented in `sklearn` can be found in the [User Guide](https://scikit-learn.org/stable/modules/naive_bayes.html). I have taken the formulae there and clarified the notations and expanded the explainations.

        More details about encoding categorical data using `sklearn` can be found from the [sklearn.preprocessing documentation](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features).
        '''
    )