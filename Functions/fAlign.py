# ======================================================================================================================
"""
This script creates a function used to align dataframes.

By Sebastian Jensen
Jan, 2024
Aarhus University
CREATES
"""
# ======================================================================================================================
# Importing libraries
import numpy as np


# Creating function
def fAlign(gdp, pop, defl, ppp, emis):
    """
    This function is used to align dataframes.

    ARGUMENTS
        * gdp:  GDP
        * pop:  Population
        * defl: GDP deflator
        * ppp:  Purchasing power parity conversion factor
        * emis: Emissions

    Returns
        * gdp_pc:    Aligned version of transformed GDP
        * emis_pc:   Aligned verion of transformed emissions
        * pop_out:   Aligned version of population
        * countries: List of countries that are not all NANs
    """

    countries = list(emis.columns.intersection(gdp.columns))

    gdp_same = gdp[countries]
    pop_same = pop[countries]
    defl_same = defl[countries]
    ppp_same = ppp[countries]
    emis_same = emis[countries]

    year = np.where(gdp_same.index == 2005)[0][0]

    real_gdp = gdp_same / defl_same
    gdp_2005 = real_gdp * np.reshape(np.array(defl_same.iloc[year, :]), (1, -1), order='F')
    gdp_usd_2005 = gdp_2005 / np.reshape(np.array(ppp_same.iloc[year, :]), (1, -1), order='F')
    gdp_usd_2005 = gdp_usd_2005 / 1e9

    where_gdp = gdp_usd_2005.isna()
    where_emis = emis_same.isna()
    where_pop = pop_same.isna()

    where_mat = where_gdp | where_emis | where_pop

    gdp_out = gdp_usd_2005[~where_mat]
    emis_out = emis_same[~where_mat]
    pop_out = pop_same[~where_mat]

    junk1 = np.shape(where_mat)[0] - np.sum(where_mat, axis=0) > 0
    idx1 = np.where(~junk1)[0]

    junk2 = np.shape(where_mat)[1] - np.sum(where_mat, axis=1) > 0
    idx2 = np.where(~junk2)[0]

    gdp_out = gdp_out.drop(gdp_out.columns[idx1], axis=1)
    gdp_out = gdp_out.drop(gdp_out.index[idx2], axis=0)

    emis_out = emis_out.drop(emis_out.columns[idx1], axis=1)
    emis_out = emis_out.drop(emis_out.index[idx2], axis=0)

    pop_out = pop_out.drop(pop_out.columns[idx1], axis=1)
    pop_out = pop_out.drop(pop_out.index[idx2], axis=0)

    countries = gdp_out.columns

    return gdp_out, emis_out, pop_out, countries
