import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def exact_name_match(df):
    df['exact_match'] = df.apply(lambda x: "match" if x['output_name'] == x['name'] else "mismatch", axis=1)

    return df


def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())


def pair_winner(df):
    """
    :return: input dataframe with three new columns: "pair_result", "pair_diff", "pair_diff_norm".
    - pair_result: based on the scores of the pair; binary; values: first --> higher score/second --> lower score;
    - pair_diff: based on the score of the pair; float; absolute difference;
    - pair_diff_norm: based on the pair_diff of a pair; float; normalization of calculated differences via Min-Max-Scaling over entire population;
    """
    df['pair_result'] = np.NaN
    df['pair_diff'] = np.NaN

    for id in df.pair_ID.unique():
        for s in df.context_sentence.unique():
            df_temp = df.loc[(df['pair_ID'] == id) & (df['context_sentence'] == s)]
            index_neg = df_temp.index[df_temp['is_in_train'] == 0].tolist()[0]
            index_pos = df_temp.index[df_temp['is_in_train'] == 1].tolist()[0]
            score_neg = df_temp.at[index_neg, 'score']
            score_pos = df_temp.at[index_pos, 'score']
            try:
                score_neg = float(score_neg)
                score_pos = float(score_pos)
                if score_neg < score_pos:
                    diff = abs(score_pos - score_neg)
                    df.at[index_neg, 'pair_result'] = "second"
                    df.at[index_pos, 'pair_result'] = "first"
                    df.at[index_neg, 'pair_diff'] = diff
                    df.at[index_pos, 'pair_diff'] = diff

                elif score_neg > score_pos:
                    diff = abs(score_neg - score_pos)
                    df.at[index_neg, 'pair_result'] = "first"
                    df.at[index_pos, 'pair_result'] = "second"
                    df.at[index_neg, 'pair_diff'] = diff
                    df.at[index_pos, 'pair_diff'] = diff
            except:
                continue

    df['pair_diff_norm'] = min_max_scaling(df['pair_diff'])

    return df


def investigate_ND(df):
    """
    ND stands for not Not Detected and means insatcnes in which the name was not recognizeD by the model as PER.
    :return: dataframe without -ND- values in score column
    """
    for s in df.context_sentence.unique():
        print("For sentence: ", s)
        temp = df.loc[df['context_sentence'] == s]
        print("Total: ", len(temp))
        temp_new = temp.loc[temp['score'] != 'ND']
        print("Without ND: ", len(temp_new))

        if len(temp) != len(temp_new):
            temp_nd = temp.loc[temp['score'] == 'ND']
            print(temp_nd)

        print("---" * 10)

    return df.loc[df['score'] != 'ND']


def winner_box_plot(df, NEG_COLOR, POS_COLOR, model_size):
    """
    produces a box plot which compares the normalized pair differences of pair winners per context sentence;
    """
    winner_total = df.loc[df['pair_result'] == 'first']
    fig = px.box(winner_total, x="is_in_train", y="pair_diff_norm", color="is_in_train", points="outliers",
                 facet_row='context_sentence',
                 color_discrete_map={0: NEG_COLOR, 1: POS_COLOR},
                 width=1000, height=5000)

    fig.update_layout(title="BERT " + model_size, template='ggplot2')

    path_fig = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/scripts/plots/"
    fig.write_image(path_fig + "exp_v1_distributionOfPairDiffNorm_Bert" + model_size + ".png")


def winner_bar_plot(df, NEG_COLOR, POS_COLOR, model_size):
    """
    produces a bar plot which compares the number of winners per context sentence. The plot differentiates between negative and positive samples.
    """
    fig = go.Figure(data=[
        go.Bar(name='Negative samples', x=df.context_sentence, y=df.num_winner_neg, marker_color=NEG_COLOR),
        go.Bar(name='Positive Samples', x=df.context_sentence, y=df.num_winner_pos, marker_color=POS_COLOR)
    ])
    fig.update_layout(title="Distribution of pair winners - BERT " + model_size, barmode='group', template="ggplot2")

    path_fig = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/scripts/plots/"
    fig.write_image(path_fig + "exp_v1_distributionOfWinners_Bert" + model_size + ".png")


def compare_pair_winner(df, model_size):
    """
    :param df: result model output
    :param model_size: size of used model
    This function compares the results (scores) for negative and positive samples provided to the BERT NER model for the 9 different contexts.
    """
    POS_COLOR = '#6d9c8f'
    NEG_COLOR = '#9C6D7A'

    winner_neg = df.loc[(df['is_in_train'] == 0) & (df['pair_result'] == 'first')]
    winner_pos = df.loc[(df['is_in_train'] == 1) & (df['pair_result'] == 'first')]

    winner_box_plot(df, NEG_COLOR, POS_COLOR, model_size)

    winner_dist = pd.DataFrame(columns=['context_sentence', 'num_winner_neg', 'num_winner_pos'])
    for s in df.context_sentence.unique():
        print("For sentence: ", s)
        temp_neg = winner_neg.loc[winner_neg['context_sentence'] == s]
        temp_pos = winner_pos.loc[winner_pos['context_sentence'] == s]
        print("There are {} pair winners for the NEGATIVE samples.".format(len(temp_neg)))
        print("There are {} pair winners for the POSITIVE samples.".format(len(temp_pos)))
        temp_dict = {'context_sentence': s, 'num_winner_neg': int(len(temp_neg)), 'num_winner_pos': int(len(temp_pos))}
        winner_dist = winner_dist.append(temp_dict, ignore_index=True)
        print("---" * 10)

    winner_bar_plot(winner_dist, NEG_COLOR, POS_COLOR, model_size)

    print("IN TOTAL:")
    print("There are {} pair winners for the NEGATIVE samples.".format(len(winner_neg)))
    print("There are {} pair winners for the POSITIVE samples.".format(len(winner_pos)))


def main():

    path_for_experiment_data = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/datasets/experiment_data/"
    bert_base_results = pd.read_csv(path_for_experiment_data + "results/results_expv1_bert_base.csv", encoding='utf-8',
                                    index_col=0)
    bert_large_results = pd.read_csv(path_for_experiment_data + "results/results_expv1_bert_large.csv",
                                     encoding='utf-8',
                                     index_col=0)

    # show all columns
    pd.set_option('display.max_columns', None)

    ### BERT BASE ###
    print("BERT BASE:")
    bert_base_results = exact_name_match(bert_base_results)
    bert_base_results_noND = pair_winner(bert_base_results)
    compare_pair_winner(bert_base_results, "BASE")
    # save processed file with additional info (exact match of name /output name; pair winner/loser; difference in scores per pair)
    bert_base_results.to_csv(path_for_experiment_data + "results/results_expv1_bert_base_PLUS.csv", encoding='utf-8')

    print("***" * 10)

    ### find ND instances in data
    """
    bert_base_results_no_ND = investigate_ND(bert_base_results)
    bert_base_results_no_ND[['pair_ID', 'score']] = bert_base_results_no_ND[['pair_ID', 'score']].apply(pd.to_numeric)
    bert_base_results_no_ND[['is_in_train', 'pair_result']] = bert_base_results_no_ND[['is_in_train', 'pair_result']].astype('string')

    # Visualization

    for s in bert_base_results_no_ND['context_sentence'].unique():
        temp_df = bert_base_results_no_ND.loc[bert_base_results_no_ND['context_sentence'] == s]

        fig = px.scatter(temp_df, x="pair_ID", y="score", color="is_in_train", symbol="pair_result",
                         color_discrete_map={"0": "#954b5e", "1":  "#4b9582"},
                         symbol_map={"first":"star-diamond", "second":"circle"},
                         hover_name="name", hover_data=["score", "pair_diff"],
                         opacity = 0.8,
                         width=1200, height=600,
                         template="plotly_white",
                         title=s
                         )

        fig.update_yaxes(title_text='PER tag score')
        fig.update_xaxes(title_text='Pair ID')
        temp_title = re.sub(r'[^\w\s]', '', s)

        path_fig = "C:/Users/z003zewu/Desktop/UNI/MAThesis/coding/scripts/plots/"
        fig.write_html(path_fig + "exp_v1_winner_pairs_"+temp_title+"_.html")
        fig.write_image(path_fig + "exp_v1_winner_pairs_"+temp_title+"_.png")
    """

    ### BERT LARGE ###
    print("BERT LARGE:")
    bert_large_results = exact_name_match(bert_large_results)
    bert_large_results_noND = pair_winner(bert_large_results)
    compare_pair_winner(bert_large_results, "LARGE")
    # save processed file with additional info (exact match of name /output name; pair winner/loser; difference in scores per pair)
    bert_large_results.to_csv(path_for_experiment_data + "results/results_expv1_bert_large_PLUS.csv", encoding='utf-8')


if __name__ == "__main__":
    main()