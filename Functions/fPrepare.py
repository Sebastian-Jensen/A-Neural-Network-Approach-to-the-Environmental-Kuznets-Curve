# ======================================================================================================================
"""
This script creates a function used to prepare the data.

Required subroutines:
* fAlign.py

By Sebastian Jensen
Jan, 2024
Aarhus University
CREATES
"""
# ======================================================================================================================

# Importing libraries
import numpy as np
from fAlign import fAlign


def fPrepare(GDP, POP, DEF, PPP, GHG):
    """
     This function is used to prepare the data.

     ARGUMENTS
         * GDP:         GDP
         * POP:         Population
         * DEF:         GDP deflator
         * PPP:         Purchasing power parity conversion factor
         * GHG:         Emissions

     Returns
         * gdp:         Aligned and prepared version of transformed GDP
         * ghg:         Aligned and prepared verion of transformed emissions
         * pop:         Aligned and prepared version of population
     """

    # Aligning and log-transforming data
    GDP_aligned, GHG_aligned, POP_aligned, Countries = fAlign(GDP, POP, DEF, PPP, GHG)

    countries = {}
    pop = {}
    gdp = {}
    ghg = {}

    regions = ['OECD', 'REF', 'Asia', 'MAF', 'LAM']

    for ii in range(len(regions)):
        if regions[ii] == 'OECD':
            countries[regions[ii]] = list(
                {'ALB', 'AUS', 'AUT', 'BEL', 'BIH', 'BGR', 'CAN', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
                 'DEU', 'GRC', 'GUM', 'HUN', 'ISL', 'IRL', 'ITA', 'JPN', 'LVA', 'LTU', 'LUX', 'MLT', 'MNE', 'NLD',
                 'NZL', 'NOR', 'POL', 'PRT', 'PRI', 'ROU', 'SRB', 'SVK', 'SVN', 'ESP', 'SWE', 'CHE', 'MKD', 'TUR',
                 'GBR', 'USA'}.intersection(set(Countries)))

            gdp[regions[ii]] = GDP_aligned[countries[regions[ii]]]
            ghg[regions[ii]] = GHG_aligned[countries[regions[ii]]]
            pop[regions[ii]] = POP_aligned[countries[regions[ii]]]

            gdp[regions[ii]].sort_index(axis=1, inplace=True)
            ghg[regions[ii]].sort_index(axis=1, inplace=True)
            pop[regions[ii]].sort_index(axis=1, inplace=True)

            country = 'USA'

            tmp = gdp[regions[ii]][country]
            gdp[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            gdp[regions[ii]].insert(0, country, tmp)

            tmp = ghg[regions[ii]][country]
            ghg[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            ghg[regions[ii]].insert(0, country, tmp)

            tmp = pop[regions[ii]][country]
            pop[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            pop[regions[ii]].insert(0, country, tmp)

        elif regions[ii] == 'REF':
            countries[regions[ii]] = list(
                {'ARM', 'AZE', 'BLR', 'GEO', 'KAZ', 'KGZ', 'MDA', 'RUS', 'TJK', 'TKM', 'UKR', 'UZB',
                 'XKX'}.intersection(set(Countries)))

            gdp[regions[ii]] = GDP_aligned[countries[regions[ii]]]
            ghg[regions[ii]] = GHG_aligned[countries[regions[ii]]]
            pop[regions[ii]] = POP_aligned[countries[regions[ii]]]

            gdp[regions[ii]].sort_index(axis=1, inplace=True)
            ghg[regions[ii]].sort_index(axis=1, inplace=True)
            pop[regions[ii]].sort_index(axis=1, inplace=True)

            country = 'RUS'

            tmp = gdp[regions[ii]][country]
            gdp[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            gdp[regions[ii]].insert(0, country, tmp)

            tmp = ghg[regions[ii]][country]
            ghg[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            ghg[regions[ii]].insert(0, country, tmp)

            tmp = pop[regions[ii]][country]
            pop[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            pop[regions[ii]].insert(0, country, tmp)

            gdp[regions[ii]].loc[:1986] = np.nan
            ghg[regions[ii]].loc[:1986] = np.nan
            pop[regions[ii]].loc[:1986] = np.nan

        elif regions[ii] == 'Asia':
            countries[regions[ii]] = list(
                {'AFG', 'BGD', 'BTN', 'BRN', 'KHM', 'CHN', 'PRK', 'KOR', 'FJI', 'PYF', 'IND', 'IDN', 'LAO', 'MYS',
                 'MDV', 'FSM', 'MNG', 'MMR', 'NPL', 'NCL', 'PAK', 'PNG', 'PHL', 'WSM', 'SGP', 'SLB', 'LKA', 'THA',
                 'TLS', 'VUT', 'VNM', 'TUV', 'MAC', 'MHL', 'PLW', 'HKG', 'TON', 'KIR'}.intersection(set(Countries)))

            gdp[regions[ii]] = GDP_aligned[countries[regions[ii]]]
            ghg[regions[ii]] = GHG_aligned[countries[regions[ii]]]
            pop[regions[ii]] = POP_aligned[countries[regions[ii]]]

            gdp[regions[ii]].sort_index(axis=1, inplace=True)
            ghg[regions[ii]].sort_index(axis=1, inplace=True)
            pop[regions[ii]].sort_index(axis=1, inplace=True)

            country = 'CHN'

            tmp = gdp[regions[ii]][country]
            gdp[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            gdp[regions[ii]].insert(0, country, tmp)

            tmp = ghg[regions[ii]][country]
            ghg[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            ghg[regions[ii]].insert(0, country, tmp)

            tmp = pop[regions[ii]][country]
            pop[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            pop[regions[ii]].insert(0, country, tmp)

        elif regions[ii] == 'MAF':
            countries[regions[ii]] = list(
                {'DZA', 'AGO', 'BHR', 'BEN', 'BWA', 'BFA', 'BDI', 'CMR', 'CPV', 'CAF', 'TCD', 'COM', 'COD', 'COG',
                 'CIV', 'DJI', 'EGY', 'GNQ', 'ERI', 'ETH', 'GAB', 'GMB', 'GHA', 'GIN', 'GNB', 'IRN', 'IRQ', 'ISR',
                 'JOR', 'KEN', 'KWT', 'LBN', 'LSO', 'LBR', 'LBY', 'MDG', 'MWI', 'MLI', 'MRT', 'MUS', 'MAR', 'MOZ',
                 'NAM', 'NER', 'NGA', 'PSE', 'OMN', 'QAT', 'RWA', 'SAU', 'SEN', 'SLE', 'SOM', 'ZAF', 'SSD', 'SDN',
                 'SWZ', 'SYR', 'TGO', 'TUN', 'UGA', 'ARE', 'TZA', 'YEM', 'ZMB', 'ZWE', 'SYC',
                 'STP'}.intersection(set(Countries)))

            gdp[regions[ii]] = GDP_aligned[countries[regions[ii]]]
            ghg[regions[ii]] = GHG_aligned[countries[regions[ii]]]
            pop[regions[ii]] = POP_aligned[countries[regions[ii]]]

            gdp[regions[ii]].sort_index(axis=1, inplace=True)
            ghg[regions[ii]].sort_index(axis=1, inplace=True)
            pop[regions[ii]].sort_index(axis=1, inplace=True)

            country = 'ZAF'

            tmp = gdp[regions[ii]][country]
            gdp[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            gdp[regions[ii]].insert(0, country, tmp)

            tmp = ghg[regions[ii]][country]
            ghg[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            ghg[regions[ii]].insert(0, country, tmp)

            tmp = pop[regions[ii]][country]
            pop[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            pop[regions[ii]].insert(0, country, tmp)

        elif regions[ii] == 'LAM':
            countries[regions[ii]] = list(
                {'ARG', 'ABW', 'BHS', 'BRB', 'BLZ', 'BOL', 'BRA', 'CHL', 'COL', 'CRI', 'CUB', 'DOM', 'ECU', 'SLV',
                 'GRD', 'GTM', 'GUY', 'HTI', 'HND', 'JAM', 'MEX', 'NIC', 'PAN', 'PRY', 'PER', 'SUR', 'TTO', 'VIR',
                 'URY', 'VEN', 'VCT', 'CUW', 'LCA', 'ATG', 'DMA', 'BMU'}.intersection(set(Countries)))

            gdp[regions[ii]] = GDP_aligned[countries[regions[ii]]]
            ghg[regions[ii]] = GHG_aligned[countries[regions[ii]]]
            pop[regions[ii]] = POP_aligned[countries[regions[ii]]]

            gdp[regions[ii]].sort_index(axis=1, inplace=True)
            ghg[regions[ii]].sort_index(axis=1, inplace=True)
            pop[regions[ii]].sort_index(axis=1, inplace=True)

            country = 'MEX'

            tmp = gdp[regions[ii]][country]
            gdp[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            gdp[regions[ii]].insert(0, country, tmp)

            tmp = ghg[regions[ii]][country]
            ghg[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            ghg[regions[ii]].insert(0, country, tmp)

            tmp = pop[regions[ii]][country]
            pop[regions[ii]].drop(labels=[country], axis=1, inplace=True)
            pop[regions[ii]].insert(0, country, tmp)

    return gdp, ghg, pop