# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:30:37 2023

@author: j.reul
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

import mode_behave_public as mb

class Function:
    """
    This class provides the functionality within the Home.py
            
    """
    
    def __init__(self, data_in, alt, equal_alt, param, k, numeric_analysis, model_type, **kwargs):
                                            
        #Initialize model
        self.model = mb.Core(
            param=param, 
            data_in=data_in, 
            alt=alt, 
            equal_alt=equal_alt
            )

        #number of clusters to analyze
        self.k = int(round(k, 0))
        
        self.numeric = numeric_analysis
        
        self.PATH_HOME = os.path.dirname(__file__)
        self.sep = os.path.sep

        self.model_type = model_type
        
    def estimate_model(self):

        if self.numeric:
            t_stats_out = True
        else:
            t_stats_out = False
            
        if self.model_type == "MNL":
            #estimate logit model
            self.model.estimate_logit(stats=t_stats_out)
        else:
            #estimate mixed logit model        
            self.model.estimate_mixed_logit(
                min_iter=10, 
                max_iter=1000,
                tol=0.01,
                space_method = 'abs_value',
                scale_space = 2,
                max_shares = 1000,
                bits_64=True,
                t_stats_out=t_stats_out
                )

    def get_consumer_groups(self):
        #conduct clustering
        res_clustering = self.model.cluster_space("kmeans", self.k)
        self.cluster_center = res_clustering[0]
        
        #get cluster sizes
        cluster_labels_pd = pd.DataFrame(columns=['labels', 'weights'])
        cluster_labels_pd['labels'] = res_clustering[1]
        #assign weights
        cluster_labels_pd['weights'] = self.model.shares
                    
        self.cluster_sizes = np.array(
            [cluster_labels_pd.loc[cluster_labels_pd['labels'] == i, 'weights'].sum() for i in range(self.k)]
            )

    def get_likelihood(self):
        if self.model_type == "MNL":
            LL_MNL = self.model.loglike_MNL()[0]
            LL_MXL = False
        else:
            LL_MNL = self.model.loglike_MNL()[0]
            LL_MXL = self.model.loglike_MXL()
        
        return LL_MNL, LL_MXL


    def conduct_visual_analysis(self):
        save_fig_path_temp = self.PATH_HOME + self.sep + "data"
                
        #visualize the preferences distribution   
        fig_space = self.model.visualize_space(
            k=self.k, 
            scale_individual=True, 
            cluster_method='kmeans', 
            external_points=np.array([self.model.initial_point]),
            bw_adjust=0.03,
            save_fig_path=save_fig_path_temp,
            return_figure=True
            )
        
        fig_forecast = self.model.forecast(method='LC', 
                    k=self.k,
                    cluster_method='kmeans',
                    name_scenario='clustering',
                    return_figure=True
                    )
        
        return fig_space, fig_forecast
    
    def export_consumer_groups(self):
                        
        param_name_list = []
        choice_alternative_list = []
        param_index_list = []
        cluster_index_list = []
        cluster_param_list = []
        cluster_size_list = []

        for k_temp in range(self.k):
            
            cluster_param_list.extend(list(self.model.initial_point[:self.model.count_c-1]))
            cluster_param_list.extend(list(self.cluster_center[k_temp]))        

            for c in range(self.model.count_c):
                if c == 0:
                    continue
                else:
                    param_name_list.append("ASC_" + str(c))
                    choice_alternative_list.append(c)
                    param_index_list.append(0)
                    cluster_index_list.append(k_temp)
                    cluster_size_list.append(self.cluster_sizes[k_temp])
            
            len_con_fix = len(self.model.param["constant"]["fixed"])
            len_con_ran = len(self.model.param["constant"]["random"])
            len_var_fix = len(self.model.param["variable"]["fixed"])
            len_var_ran = len(self.model.param["variable"]["random"])
            
            for c in range(self.model.count_c):
                for a, attr in enumerate(self.model.param["constant"]["fixed"]):
                    param_name_list.append(attr + "_" + str(c))
                    choice_alternative_list.append(c)
                    param_index_list.append(1+a)
                    cluster_index_list.append(k_temp)
                    cluster_size_list.append(self.cluster_sizes[k_temp])
            
                for a, attr in enumerate(self.model.param["constant"]["random"]):
                    param_name_list.append(attr + "_" + str(c))
                    choice_alternative_list.append(c)
                    param_index_list.append(1+a+len_con_fix)
                    cluster_index_list.append(k_temp)
                    cluster_size_list.append(self.cluster_sizes[k_temp])
            
                for a, attr in enumerate(self.model.param["variable"]["fixed"]):
                    param_name_list.append(attr + "_" + str(c))
                    choice_alternative_list.append(c)
                    param_index_list.append(1+a+len_con_fix+len_con_ran)
                    cluster_index_list.append(k_temp)
                    cluster_size_list.append(self.cluster_sizes[k_temp])
            
                for a, attr in enumerate(self.model.param["variable"]["random"]):
                    param_name_list.append(attr + "_" + str(c))
                    choice_alternative_list.append(c)
                    param_index_list.append(1+a+len_con_fix+len_con_ran+len_var_fix)
                    cluster_index_list.append(k_temp)
                    cluster_size_list.append(self.cluster_sizes[k_temp])
            
        t_stats_pandas = pd.DataFrame(
            index=range(len(self.model.initial_point)*self.k), 
            columns=["Param_Name", "Param_Value", "Param_Index", 
                     "Choice_Alternative", "Cluster_Size", "Cluster_Index"]
            )

        t_stats_pandas["Param_Name"] = param_name_list
        t_stats_pandas["Param_Value"] = cluster_param_list
        t_stats_pandas["Param_Index"] = param_index_list
        t_stats_pandas["Choice_Alternative"] = choice_alternative_list
        t_stats_pandas["Cluster_Size"] = cluster_size_list
        t_stats_pandas["Cluster_Index"] = cluster_index_list
        
        return t_stats_pandas.to_csv()        
        
    
    def export_data(self, model_type):
        return self.model.export_estimates(model_type=model_type)
    