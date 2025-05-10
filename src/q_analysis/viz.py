"""
Q-analysis Package
Copyright (C) 2024 Nikita Smirnov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# TODO: allow user to pass their figure and specify whether to show plot or not
# TODO: add docs saying that kwargs are for sns
def plot_q_analysis_vectors(structure_vectors_df, pvalues_df=None, **kwargs):
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    plot_parameters = dict(
        x='q', 
        y='Value', 
        col='Vector', 
        col_wrap=3, 
        data=structure_vectors_df,
        kind='point', 
        sharey=False, 
        dodge=0.5,
        estimator='median',
        height=4,
        aspect=1.2,
        legend_out=True,
        markers=['o', 's', 'D', '^', 'v', 'p']
    )
    plot_parameters.update(**kwargs)

    g = sns.catplot(**plot_parameters)

    hue = kwargs.get('hue')
    if pvalues_df is not None and hue is not None:
        hue_categories = structure_vectors_df[hue].unique()
        pair = tuple(hue_categories)

    for i, ax in enumerate(g.axes.flat):
        # Make x-axis tick labels vertical to prevent overlap
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        if pvalues_df is not None and hue is not None:
            hue_categories = structure_vectors_df[hue].unique()
            
            pair = tuple(hue_categories)
            
            current_vector = g.col_names[i]
            filtered_pvals = pvalues_df[pvalues_df['Vector'] == current_vector]
            p_values = [
                (q_val, pvalues_df.query(f'Vector == "{current_vector}" and q == {q_val}')['Value'].iloc[0])
                for q_val in filtered_pvals['q'].unique()
                if pvalues_df.query(f'Vector == "{current_vector}" and q == {q_val}')['Value'].iloc[0] < 0.05
            ]
            if len(p_values) == 0:
                continue
            box_pairs = [
                [(q_val, pair[0]), (q_val, pair[1])]
                for q_val, _ in p_values
            ]
            p_values = [p_value for _, p_value in p_values]
            annotator = Annotator(
                ax,
                box_pairs,
                data=structure_vectors_df[
                    structure_vectors_df["Vector"] == current_vector
                ],
                x="q",
                y="Value",
                hue=hue,
            )
            annotator.configure(test=None, text_format='star', loc='outside')
            annotator.set_pvalues(p_values)
            annotator.annotate()
            

    g.set_titles("{col_name}")
    g.tight_layout()