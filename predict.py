import numpy as np
import tensorflow as tf
import pathlib
import sklearn.metrics

def _main():
    idg = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.0
	# 学習時のスケールと合わせる必要がある
    )
    gen_validation = idg.flow_from_directory(
        directory='validation',
        target_size=(32, 32),
        batch_size=128,
        shuffle=False
    )

    # 学習済みのモデルをロード
    model = tf.keras.models.load_model('model_compe_3layers_cnn_train.h5')
    preds = model.predict_generator(gen_validation)

    # 検証データは熊、虫、ネズミの３種類が各100毎ずつ
    labels = ['bear', 'beetle', 'mouse']
    # 正解ラベルの作成
    y_label = []
    for label in labels:
        for i in range(100):
            y_label.append(label)

    # 推論ラベルの作成
    y_preds = []
    for p in preds:
        # 確信度が最大の値を推論結果とみなす
        label_idx = p.argmax()
        y_preds.append(labels[label_idx])
    
    # 混合行列を取得
    val_mat = sklearn.metrics.confusion_matrix(y_label, y_preds, labels=labels)
    print('[混合行列]')
    print(f'      {labels[0]: >6} {labels[1]: >6} {labels[2]: >6}')
    for i, row in enumerate(val_mat):
        print(f'{labels[i]: >7} {val_mat[i][0]: >4} {val_mat[i][1]: >6} {val_mat[i][2]: >6}')
    print()

    rec_score = sklearn.metrics.recall_score(y_label, y_preds, average=None)
    print('再現率： ',rec_score)

    pre_score = sklearn.metrics.precision_score(y_label, y_preds, average=None)
    print('適合率： ', pre_score)

    acc_score = sklearn.metrics.accuracy_score(y_label, y_preds)
    print('正解率： ', acc_score)
    print()

    f1_score = sklearn.metrics.f1_score(y_label, y_preds, average=None)
    print('F値   ： ', f1_score)

    rec_score_avg = sklearn.metrics.recall_score(y_label, y_preds, average="macro")
    print('再現率(平均)： ', rec_score_avg)
    pre_score_avg = sklearn.metrics.precision_score(y_label, y_preds, average="macro")
    print('適合率(平均)： ', pre_score_avg)
    f1_score_avg = sklearn.metrics.f1_score(y_label, y_preds, average="macro")
    print('F値(平均)   ： ', f1_score_avg)

if __name__ == '__main__':
    _main()
