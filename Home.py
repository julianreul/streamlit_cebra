import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os

from functionality import Function 

st.set_option('deprecation.showPyplotGlobalUse', False)

sep = os.sep
PATH_IMAGE = "images" + sep + "logo_orange_cebra.png"
image = Image.open(PATH_IMAGE)
st.image(image, width = 300)

add_text = st.sidebar.text(
    "OCEBRA - Data-driven analyses\
    \nJulian Reul\
    \nE-Mail: j.reul@ocebra.de\
    "
    )

st.markdown("OCEBRA is a data-consultancy based in Cologne and with roots at the Forschungszentrum Jülich.")
   
st.markdown("""
    We offer data analytics with a focus on interpretable neural networks and web-based data visualizations.
    """)
    
st.markdown("""
    At the core of our analysis, we employ the open-source model MO|DE.behave, which was recently developed at the Institute for Energy and Climate Research - Techno-economic Systems Analysis (IEK-3) at the Forschungszentrum Jülich.
	This model originally stems from the field of behavioral modeling to explore the choice behavior of individuals and groups. 
	The model's unique feature is that it retains the interpretability of its underlying neural-network structure.
	Conventional neural networks remain uninterpretable black-boxes due to their inherent complexity. 
	This makes them great algorithms for applications such as image- or voice-recognition, but limits their capability of delivering insights into the how and why data-inputs and -outputs are causally linked.
    """)
    
st.markdown("""
    MO|DE.behave is a general data analytics tool, which can be applied to any field of analysis, with roots in behavioral analyses.
    Typical applications include the analysis of survey data to derive behavioral preferences of individuals or groups.\n
    OCEBRA's services are a great fit, if you plan to conduct a survey and you are now looking for a partner conducting science-based data analytics to gain deeper insights.
            """)

st.markdown("""---""")

st.markdown("Gain FIRST INSIGHTS already ONLINE:")

#uploading data from local directory
uploaded_file = st.file_uploader("Upload your survey data as .csv-files", help="Have a look at the -Documentation- page for information on the correct data-format.")

st.markdown("HINT: Have a look at our documentation page for preparing your survey data in the right way.")

st.markdown("___")

PATH_HOME = os.path.dirname(__file__)
sep = os.path.sep


if uploaded_file is not None:

    try:
        dataframe = pd.read_csv(uploaded_file)
    except:
        raise AttributeError("Data in wrong format: Check file-format (.csv) and separator (,)")
         
    #Check column names to derive attributes
    col_names = dataframe.columns.values
    col_names_reduced = []
    for c in col_names:
        c_red = c[:-4]
        if c_red in ["choice", "av"]:
            continue
        else:
            col_names_reduced.append(c_red)
        
    col_names_reduced = list(set(col_names_reduced))    

    choice_cols = [col for col in col_names if col.startswith("choice")]
    #number of choice alternatives
    alt_temp = max([int(c[7]) for c in choice_cols])+1
    #number of equal choice alternatives
    equal_alt_temp = max([int(c[9]) for c in choice_cols])+1

    with st.form(key='Selecting Columns'):
        
        st.markdown("Which type of analysis do you want to conduct?")

        mnl_model = st.checkbox(
            "Estimate Multinomial Logit Model", 
            help = "A multinomial logit model estimates a single, average set of choice preferences for the whole base population."
            )
        
        mxl_model = st.checkbox(
            "Estimate nonparametric Mixed Logit Model",
            help = "A nonparametric mixed logit model estimates 1000 preference sets (in this case) and weights them according to their relative importance in the base population."
            )
        consumer_groups = st.checkbox(
            "Identify consumer groups",
            help = "kmeans clustering is performed upon the mixed logit results to identify more homogeneous consumer/preference groups among the 1000 previously estimated preference sets.")

        options = st.multiselect(
            'Which attributes do you want to consider as model parameters?',
            col_names_reduced
            )                
        
        k_temp = st.number_input(
            'How many preference groups do you want to analyze?',
            help = "Only relevant for mixed logit models and if consumer groups shall be identified.",
            value=2
            )

        
        submit_button = st.form_submit_button(label='Confirm selection')

    if submit_button:
                
        st.text("Model estimation starts...")
        
        #derive param_temp and declare all attributes and random and variable.
        param_fixed = []
        param_random = options

        param_temp = {'constant': 
                          {
                           'fixed':[],
                           'random':[]
                           },
                      'variable':
                          {
                           'fixed': [],
                           'random':[]
                           }
                      }

        param_temp['variable']['fixed'] = param_fixed
        param_temp['variable']['random'] = param_random
        
        if mxl_model:
            model_type = "MXL"
        else:
            model_type = "MNL"
        
        function_ = Function(
            dataframe, 
            alt_temp, 
            equal_alt_temp, 
            param_temp, 
            k_temp, 
            True,
            model_type
            )
        
        function_.estimate_model()
        
        LL_MNL, LL_MXL = function_.get_likelihood()
    
        #Evaluation of MNL- and MXL-model
        text_temp = "LL-Ratio of MNL-model: " + str(LL_MNL)
        st.text(text_temp)
        if mxl_model:
            text_temp = "LL-Ratio of MXL-model: " + str(LL_MXL)
            st.text(text_temp)
        
        #IDENTIFY CONSUMER GROUPS
        if consumer_groups:
            function_.get_consumer_groups()
                    
        #DOWNLOAD RESULTS
        if mxl_model:
            logit_csv, mixed_logit_csv = function_.export_data(model_type = "MXL")
            
            st.download_button(
                label="Download MNL-estimates as CSV",
                data=logit_csv,
                file_name='MNL_estimates.csv',
                mime='text/csv'
                )      
            
            st.download_button(
                label="Download MXL-estimates as CSV",
                data=mixed_logit_csv,
                file_name='MXL_estimates.csv',
                mime='text/csv'
                )     
            
            if consumer_groups:
                consumer_groups_csv = function_.export_consumer_groups()    
                
                st.download_button(
                    label="Download consumer groups as CSV",
                    data=consumer_groups_csv,
                    file_name='consumer_groups.csv',
                    mime='text/csv'
                    )     
            
        else:
            logit_csv = function_.export_data(model_type = "MNL")
            
            st.download_button(
                label="Download MNL-estimates as CSV",
                data=logit_csv,
                file_name='MNL_estimates.csv',
                mime='text/csv'
                )
