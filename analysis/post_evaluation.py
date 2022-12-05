import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)


def neg_occurrence_plot(no_occurrences_bert_base, count_neg_names_bert_base, no_occurrences_bert_large,
                       count_neg_names_bert_large):
    """
    produces a bar plot which shows for each model size (BERT base and large) how often a NEG pair winner (negative sample name) won across the 9 different contexts.
    """

    fig = go.Figure(data=[
        go.Bar(name="BERT BASE", x=no_occurrences_bert_base, y=count_neg_names_bert_base, marker_color="#26b4b4"),
        go.Bar(name='BERT LARGE', x=no_occurrences_bert_large, y=count_neg_names_bert_large, marker_color="#266db4")
    ])
    fig.update_layout(title="Frequency of pair wins by negative sample name", barmode='group', template="ggplot2")
    fig.update_xaxes(title="Number of pair wins", dtick=1)
    fig.update_yaxes(title="Number of contexts")

    path_fig = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/scripts/plots/"
    fig.write_image(path_fig + "exp_v1_numberNegativeSampleWins_baseAndLarge.png")


def winner_neg(df):
    """
    This function counts the wins of negative samples per name across contexts
    :param df: analysis file
    """
    winner_neg = df.loc[(df['is_in_train'] == 0) & (df['pair_result'] == 'first')]

    neg_winner_count1 = winner_neg.groupby(['name']).size().to_dict()
    neg_winner_count2 = winner_neg['name'].value_counts()

    no_occurrences = []
    count_neg_names = []
    for occur in neg_winner_count2.unique():
        print("There are {} negative sample names that occur {}-times as winners".format(
            neg_winner_count2.value_counts()[occur], occur))
        no_occurrences.append(occur)
        count_neg_names.append(neg_winner_count2.value_counts()[occur])

    list_9_time_winners = []
    for k, v in neg_winner_count1.items():
        if v == 9:
            print('{} has 9 wins'.format(k))
            list_9_time_winners.append(k)

    return no_occurrences, count_neg_names, list_9_time_winners


def avg_score_per_name(df):
    """
    :return: a dataframe with averaged score values per name
    """
    df = df.loc[df['score'] != 'ND']
    df = df.astype({'score': 'float', 'pair_diff_norm': 'float'})

    avg_df = pd.DataFrame(columns=["pair_ID", "is_in_train", "name", "avg_score", "avg_pair_diff_norm"])

    for name in df.name.unique():
        df_temp = df.loc[df['name'] == name]
        avg_score_temp = df_temp.score.mean()
        avg_diff_temp = df_temp.pair_diff_norm.mean()
        pair_id_temp = df_temp.pair_ID.unique().tolist()
        is_in_train_temp = df_temp.is_in_train.unique().tolist()

        temp_dict = {'pair_ID': pair_id_temp[0], 'is_in_train': is_in_train_temp[0], 'name': name,
                     'avg_score': avg_score_temp, 'avg_pair_diff_norm': avg_diff_temp}
        avg_df = avg_df.append(temp_dict, ignore_index=True)

    #compare avg score between negative and positive samples
    avg_df = avg_df.astype({'avg_score': 'float', 'avg_pair_diff_norm': 'float'})
    avg_df_0 = avg_df.loc[avg_df['is_in_train'] == 0]
    avg_df_0 = avg_df_0.drop(['pair_ID', 'is_in_train', 'name'], axis=1)
    avg_df_1 = avg_df.loc[avg_df['is_in_train'] == 1]
    avg_df_1 = avg_df_1.drop(['pair_ID', 'is_in_train', 'name'], axis=1)

    print("Comparison of statistical values:")
    print("NEGATIVE samples:")
    print(avg_df_0.describe())
    print("-----")
    print("POSITIVE samples:")
    print(avg_df_1.describe())

    return avg_df


def avg_scores_box_plot(df, NEG_COLOR, POS_COLOR, model_size):
    """
    produces a box plot which shows the distribution of average scores for negative and positive samples and per model size (BERT base and large)
    """
    fig = px.box(df, x="is_in_train", y="avg_score", color="is_in_train", points="outliers",
                 color_discrete_map={0: NEG_COLOR, 1: POS_COLOR},
                 width=800, height=1200)

    fig.update_layout(title="BERT " + model_size + " : Average score per name", template='ggplot2')

    path_fig = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/scripts/plots/"
    fig.write_image(path_fig + "exp_v1_distributionOfAVGScores_Bert" + model_size + ".png")


def main():

    ### path to results
    path = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/experiment_data/results/"

    results_bert_base = pd.read_csv(path + "results_expv1_bert_base_PLUS.csv", encoding='utf-8', index_col=0)
    results_bert_large = pd.read_csv(path + "results_expv1_bert_large_PLUS.csv", encoding='utf-8', index_col=0)
    pd.set_option('display.max_columns', None)

    ### color definition for positive (green) and negative samples (red)
    POS_COLOR = '#6d9c8f'
    NEG_COLOR = '#9C6D7A'

    print("BERT BASE:")
    no_occurrences_bert_base, count_neg_names_bert_base, winner9_bert_base = winner_neg(results_bert_base)
    avg_results_bert_base = avg_score_per_name(results_bert_base)
    avg_results_bert_base.to_csv(path + "results_expv1_bert_base_AVG.csv", encoding='utf-8')
    avg_scores_box_plot(avg_results_bert_base, NEG_COLOR, POS_COLOR, "BASE")

    print("***" * 10)

    print("BERT LARGE:")
    no_occurrences_bert_large, count_neg_names_bert_large, winner9_bert_large = winner_neg(results_bert_large)
    avg_results_bert_large = avg_score_per_name(results_bert_large)
    avg_results_bert_large.to_csv(path + "results_expv1_bert_large_AVG.csv", encoding='utf-8')
    avg_scores_box_plot(avg_results_bert_large, NEG_COLOR, POS_COLOR, "LARGE")

    ### visualize frequency of neg winners + comparison of models
    neg_occurrence_plot(no_occurrences_bert_base, count_neg_names_bert_base, no_occurrences_bert_large,
                       count_neg_names_bert_large)

    ### compare the 9 times neg winners of the different model sizes
    neg_winners_both_models = [x for x in winner9_bert_base if x in winner9_bert_large]
    print("BERT BASE has {} 9-time winners, BERT LARGE has {} 9-time winners. There are {} 9-time neg winners in both models.".format(
            len(winner9_bert_base), len(winner9_bert_large), len(neg_winners_both_models)))

    print("9-time NEG winners that appear in both models: ", neg_winners_both_models)

if __name__ == "__main__":
    main()
