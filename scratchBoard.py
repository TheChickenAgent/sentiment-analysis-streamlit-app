    def create_pairs(sentence_list):
        sentence_1 = sentence_list[0]
        sentence_2 = sentence_list[1]
        sentence_pair = (sentence_1, sentence_2)
        sentence_list.append(sentence_pair)

        dict = {"sentence": sentence_list} 
        df_pairs = pd.DataFrame(dict)
        return df_pairs

    def wide_dataframe(dataframe):
        col = list(range(2*768))
        list_of_double_lists = []

        #Go over each value in the embedding and put each value in its own column
        for row in tqdm(range(0, len(dataframe["sentence_pairs"]))):
            l = []
            for i in range(2):
                l.extend(dataframe["sentence_pairs"][row][i])
            list_of_double_lists.append(l)

        dataframe_new = pd.DataFrame(columns = col, data = list_of_double_lists)

        return dataframe_new

    df_pairs_train = create_pairs(sentence_list = sentences)