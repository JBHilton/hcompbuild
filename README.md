# hcompbuild
Python tools for constructing risk-stratified household compartmental infection models.

## Data sources

This repo comes prepackaged with demographic data and contact matrices which can be used to parameterise models.
* The contact matrices in *uk_contacts_all.csv*, *uk_contacts_home.csv* and *uk_contacts_other.csv* are taken from the supplementary material to [Prem et al.](https://doi.org/10.1371/journal.pcbi.1005697).
* The following files contain aggregated data on household composition from the 2011 UK census:
  * *eng_and_wales_adult_child_composition_dist.csv*
  * *eng_and_wales_adult_child_composition_list.csv*
  * *eng_and_wales_adult_child_vuln_composition_dist.csv*
  * *eng_and_wales_adult_child_vuln_composition_list.csv*
  * *uk_composition_dist.csv*
  * *uk_composition_list.csv*
* The population pyramid (age distribution by sex) in *uk_pop_pyramid.csv* comes from the [Office for National Statistics](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/articles/ukpopulationpyramidinteractive/2020-01-08).
