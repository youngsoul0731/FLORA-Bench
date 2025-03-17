def fix_score(data_dict):
    prediction = data_dict['prediction']
    score = float(data_dict['score'])
    answer = data_dict['answer']
    if answer in prediction and score == 0.0:
        return 1.0
    else:
        return score