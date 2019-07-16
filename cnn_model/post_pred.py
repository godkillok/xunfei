from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import logging
import os
import tensorflow as tf
def top_2_label_code(test_preds_prob,test_y):
    test_preds = []
    for prob in test_preds_prob:
        test_preds.append(list(prob.argsort()[-2:][::-1]))

    test_y_name = []
    test_preds_code= []
    for real, pred in zip(test_y, test_preds):
        prd = pred[0]
        # print(real, pred)
        for pr in pred:
            if real == pr:
                prd = real
        test_y_name.append(real)
        test_preds_code.append(prd)

    return test_y_name,test_preds_code

def top_2_acc(test_preds_prob,test_y):
    test_y_name, test_preds_code=top_2_label_code(test_preds_prob, test_y)
    return accuracy_score(test_y_name, test_preds_code)
def post_eval(path_label,output_results,model_dir='',history_dir=''):
    if model_dir!='' and model_dir[-1]=='/':
        model_dir=model_dir[:-1]
    with open(path_label, 'r', encoding='utf8') as f:
        lines = f.readlines()
        id2label = {i: l.strip().split("\x01\t")[0] for i, l in enumerate(lines)}

    # predict
    predict_label_list = []
    true_label_list = []
    prob_list = []
    true_label_code = []
    guids = []

    for prediction in output_results:
        predict_label_id = prediction["predict_label_ids"]
        true_label_id = prediction["true_label_ids"]
        prob_list.append(prediction["probabilities"])
        guids.append(prediction["guid"])
        predict_label = id2label[predict_label_id]
        true_label = id2label[true_label_id]
        predict_label_list.append(predict_label)
        true_label_code.append(true_label_id)
        true_label_list.append(true_label)

    test_y_name, test_preds_code = top_2_label_code(prob_list, true_label_code)
    acc2 = accuracy_score(test_y_name, test_preds_code)
    acc1=accuracy_score(true_label_list, predict_label_list)
    logging.info(
    "The total program acc1 {} and top2 acc is {}".format(acc1, acc2))
    if model_dir!='':
        root, folder_name = os.path.split(model_dir)
        if acc2 <= 0.7:
            cmd = "cd {} && rm -rf {}".format(root,folder_name)
            logging.info("==========")
            logging.info(cmd)
        #os.system(cmd)
    return acc2,acc1

def post_pred(path_label,model_dir,history_dir,output_results,acc2):
    if model_dir[-1]=='/':
        model_dir=model_dir[:-1]
    with open(path_label, 'r', encoding='utf8') as f:
        lines = f.readlines()
        id2label = {i: l.strip().split("\x01\t")[0] for i, l in enumerate(lines)}

    # predict
    predict_label_list = []
    prob_list = []
    guids = []

    for prediction in output_results:
        predict_label_id = prediction["predict_label_ids"]
        prob_list.append(prediction["probabilities"])
        guids.append(prediction["guid"])
        predict_label = id2label[predict_label_id]

        predict_label_list.append(predict_label)

    root, folder_name = os.path.split(model_dir)

    output_eval_file = os.path.join(history_dir, "p_{}_results.txt".format(acc2))
    try:
        os.makedirs(history_dir)
    except:
        pass

    logging.info(model_dir)
    if acc2>0.74:
        cmd = "cd {} && mv {} {}_{}".format(root, folder_name,folder_name, acc2)
        logging.info("==========")
        logging.info(cmd)
        os.system(cmd)
    logging.info("total pred len is {} and top2{}".format(len(prob_list),acc2))
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        for guid, prob in zip(guids, prob_list):
            writer.write('{},{} \n'.format(guid, ','.join([str(pr) for pr in prob])))