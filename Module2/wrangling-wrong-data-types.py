# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:32:57 2016

Reading and changing the datatype of html files

@author: Arthur Gouveia
"""
import pandas as pd


def prepare_anos_df(uclhtml):
    df = uclhtml[4].iloc[1:].copy()
    df2 = uclhtml[5].iloc[1:].copy()
    cnames = uclhtml[4].ix[0].copy()
    cnames[5] = 'Semifinalista1'
    cnames[6] = 'Semifinalista2'
    df.columns = cnames
    df2.columns = cnames
    df['Era'] = 'Taça dos campeões'
    df2['Era'] = 'Champions League'
    df = df.append(df2)
    df['Era'].fillna('Champions League', inplace=True)
    df.reset_index(drop=True, inplace=True)
    del(df2)
    del(cnames)

    return df


def prepare_aprov_finais(uclhtml):
    df = uclhtml[7].iloc[1:].copy()
    df.columns = uclhtml[7].ix[0]
    df.reset_index(drop=True, inplace=True)

    return df


def correcttypes(df):
    df.Rank = pd.to_numeric(df.Rank, errors='coerce')
    df['Nº de Títulos'] = pd.to_numeric(df['Nº de Títulos'], errors='coerce')
    df['Nº de Vice'] = pd.to_numeric(df['Nº de Vice'], errors='coerce')
    df.Aproveitamento = df.Aproveitamento.apply(
                                           lambda s: s[0:-1].replace(',', '.'))
    df.Aproveitamento = pd.to_numeric(df.Aproveitamento, errors='coerce')

    return None


html = pd.read_html(
       'https://pt.wikipedia.org/wiki/Liga_dos_Campe%C3%B5es_da_UEFA')

anos = prepare_anos_df(html)
aprov_finais = prepare_aprov_finais(html)

print('Dtypes before')
print(aprov_finais.dtypes)

correcttypes(aprov_finais)

print('Dtypes after')
print(aprov_finais.dtypes)
