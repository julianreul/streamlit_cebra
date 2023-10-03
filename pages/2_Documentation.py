import streamlit as st
import pandas as pd
import numpy as np

st.title('Documentation MO|DE.behave')

st.markdown("___")

st.header("Discrete Choice Modelling")

st.markdown(
    """
    **Discrete Choice Theory:**
    Discrete choice theory is used to study the decision-making process
    of individuals or aggregate groups.
    The core idea of discrete choice theory is, that decision-makers choose
    the choice alternative among a set of available choices (choice set),
    which has the highest *utility* for the decision-maker.
    The utility of a choice option is determined by the attributes of the 
    choice option, the (socio-economic) attributes of the decision-maker,
    and the importance, that a decision-maker assigns to a certain attribute. 
    We call this importance of attributes the *preferences* of the decision-maker.
    During the estimation of discrete choice models, those preferences
    are elicited via optimization algorithms, based on survey data.
    
    **Mixed Logit:**
    A mixed logit model is a multinomial logit model (MNL), in which the coefficients 
    do not take a single value, but are distributed over a parameter space. 
    Within this package, the mixed logit models 
    are estimated on a discrete parameter space, which is specified by the researcher (nonparametric design).
    The discrete subsets of the parameter space are called classes, 
    analogously to latent class models (LCM). The goal of the estimation procedure
    is to estimate the optimal share, i.e. weight, of each class within the discrete parameter space.
    The algorithm roughly follows the procedure below:

    1. Estimate initial coefficients of a standard multinomial logit model.
    2. Specify a continuous parameter space for the random coefficients with
       the mean and the standard deviation of each initially calculated random coefficient. 
       (The standard deviation can be calculated from a k-fold cross-validation.)
       Alternatively, the parameter space can be defined via the absolute values
       of the parameters.
    3. Draw points (maximum number of point = -max_shares-) from the parameter space via latin hypercube sampling.
    3. Estimate the optimal share for each drawn point with an expectation-maximization (EM) algorithm. (see Train, 2009)

          
    Further reading:

    * Train, K. (2009): "Mixed logit", in Discrete choice methods with simulation (pp. 76–93), Cambridge University Press
    * Train, K. (2008): "EM algorithms for nonparametric estimation of mixing distributions", in Journal of Choice Modelling, 1(1), 40–69, https://doi.org/10.1016/S1755-5345(13)70022-8
    * Train, K. (2016): "Mixed logit with a flexible mixing distribution", in Journal of Choice Modelling, 19, 40–53, https://doi.org/10.1016/j.jocm.2016.07.004
    * McFadden, D. and Train, K. (2000): "Mixed MNL models for discrete response", in Journal of Applied Econometrics, 15(5), 447-470, https://www.jstor.org/stable/2678603 

    """
            )

st.markdown("___")

st.header("Correct Input Format")

st.markdown("Exemplary input data:")

df = pd.DataFrame(
   np.random.randn(10, 15),
   columns=[
       "choice_0_0",
       "choice_1_0",
       "choice_2_0",
       "av_0_0",
       "av_1_0",
       "av_2_0",
       "attr1_0_0",
       "attr1_1_0",
       "attr1_2_0",
       "attr2_0_0",
       "attr2_1_0",
       "attr2_2_0",
       "attr3_0_0",
       "attr3_1_0",
       "attr3_2_0",
       ]
   )

df["av_0_0"] = 1
df["av_1_0"] = 1
df["av_2_0"] = 1
df["choice_0_0"] = 1
df["choice_1_0"] = 0
df["choice_2_0"] = 0

st.table(df)

st.markdown(
    """
    - **Each row** represents a **single observation** from an observed- or stated-preference survey.
    - There are three **types of columns**:
        1. Choice-columns (choice_x_y), indicating, which choice alternative has been chosen in the respective observation.
        2. Availability-columns (av_x_y), indicating, if a choice alternative was available during the observation.
        3. Attribute-columns (attribute_name_x_y), indicating the value of the respective attribute of the choice option or the decision maker.
    - **Each column** has **two suffixes**:  
        1. The first suffix (_x) counts the choice alternative.
        2. The second suffix (_y) counts the number of equal choice alternatives.
        Exemplary choice options: 2 apples and 1 banana.
        This would result in the following choice- and av-columns:
        - choice_0_0 (first apple), choice_0_1 (second apple), choice_1_0 (first banana)
        - av_0_0 (first apple), av_0_1 (second apple), av_1_0 (first banana)
    - The values in **choice- and av-columns** are either 0 (not chosen/not available) or 1 (has been chosen/was available). 
    - The values in the **attribute-colums** either describe the choice options (e.g. color of an apple) or the decision-maker (e.g. age of the decision-maker).  
      An attribute which describes the decision-maker is equal across all choice options.
      Exemplary choice options: 2 apples and 1 banana.
      - e.g., columns for attribute age: age_0_0 = 28, age_0_1 = 28, age_1_0 = 28,
        Indicating, that the age of the decision-maker was 28 for all choice options, since it was the same person.
    - **Summary**: 
        - The input data should contain **three types of columns**: Choice-, availability-, and attribute-columns.
        - There should be **one row for each survey-observation**.
        - The input data should contain the following **number of choice- and availability columns**: no_choice_alternatives*max(no_equal_choice_alternatives)
        - The input data should contain the following **number of attribute-columns**: no_choice_alternatives*max(no_equal_choice_alternatives)*no_attributes
    """
    )
